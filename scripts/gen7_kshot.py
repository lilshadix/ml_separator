"""Few-shot calibration on a genuinely new chemotype (brief §12).

Zero-shot remains the primary objective, but the deployment question a separations
chemist actually asks is different: *if I measure this new ligand two or three
times, how much does the model improve?*  gen5's k-shot study answered a version
of this on the pair task and found the uncomfortable result that from k ≈ 2 the
model added nothing over a no-model affine fit.  This re-asks it on the level task,
under the chemotype hold-out, and with that null carried explicitly.

Protocol, per held-out ligand and per k:

1. draw ``k`` of its test rows as the "measured" set — **stratified across the
   ligand's series** so k=2 is not two points of the same titration, which would
   flatter every method;
2. the remaining rows are scored;
3. four estimators are compared on exactly those remaining rows:

   ``zero_shot``    the model's own prediction, untouched;
   ``offset_only``  the model's prediction plus the mean residual over the k rows
                    — the k-shot correction the brief describes;
   ``no_model``     the mean of the k measured values.  No model at all.  This is
                    the null that killed the gen5 story and it must be beaten, not
                    omitted;
   ``affine``       a slope-and-intercept fit of truth on prediction over the k
                    rows (needs k >= 2), the strongest calibration available.

Everything is repeated over ``--draws`` random draws of the measured set, and the
result is reported per k with its spread.  A method that only wins on one draw of
one ligand has not won.
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

K_VALUES: tuple[int, ...] = (0, 1, 2, 3, 5, 8)


def stratified_draw(block: pd.DataFrame, k: int, rng: np.random.Generator) -> np.ndarray:
    """Positional indices of k rows spread over the ligand's series, not clustered."""
    if k <= 0:
        return np.empty(0, dtype=int)
    if k >= len(block):
        return np.arange(len(block))
    series = block["series_id"].astype(str).to_numpy() if "series_id" in block.columns \
        else np.zeros(len(block), dtype=int)
    order: list[int] = []
    buckets: dict = {}
    for position, key in enumerate(series):
        buckets.setdefault(key, []).append(position)
    keys = list(buckets)
    rng.shuffle(keys)
    for values in buckets.values():
        rng.shuffle(values)
    while len(order) < k:
        progressed = False
        for key in keys:
            if buckets[key]:
                order.append(buckets[key].pop())
                progressed = True
                if len(order) == k:
                    break
        if not progressed:
            break
    return np.asarray(order[:k], dtype=int)


def evaluate(oof: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for (model, split_seed), block_all in oof.groupby(["model", "split_seed"], sort=True):
        for ligand, block in block_all.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            if len(block) < 4:
                continue
            y = block["log_D"].to_numpy(float)
            p = block["prediction"].to_numpy(float)
            for k in K_VALUES:
                if k >= len(block):
                    continue
                n_draws = 1 if k == 0 else draws
                for draw in range(n_draws):
                    measured = stratified_draw(block, k, rng)
                    mask = np.ones(len(block), dtype=bool)
                    mask[measured] = False
                    if not mask.any():
                        continue
                    y_hold, p_hold = y[mask], p[mask]
                    row = {"model": model, "split_seed": int(split_seed), "extractant": ligand,
                           "k": k, "draw": draw, "n_scored": int(mask.sum()),
                           "zero_shot": float(np.abs(p_hold - y_hold).mean())}
                    if k > 0:
                        shift = float(np.mean(y[measured] - p[measured]))
                        row["offset_only"] = float(np.abs(p_hold + shift - y_hold).mean())
                        row["no_model"] = float(np.abs(np.mean(y[measured]) - y_hold).mean())
                    if k >= 2 and np.ptp(p[measured]) > 1e-9:
                        slope, intercept = np.polyfit(p[measured], y[measured], 1)
                        row["affine"] = float(np.abs(slope * p_hold + intercept - y_hold).mean())
                    records.append(row)
    return pd.DataFrame(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True,
                        help="an oof_predictions.parquet written by run_gen7_experiment")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--draws", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "runs" / "gen7_architecture" / "kshot")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(args.oof)
    if args.models:
        oof = oof[oof["model"].isin(set(args.models))]
    if oof.empty:
        raise SystemExit("no rows after model filter")

    detail = evaluate(oof, draws=args.draws, seed=args.seed)
    detail.to_parquet(args.out / "kshot_detail.parquet", index=False)

    methods = ["zero_shot", "offset_only", "no_model", "affine"]
    # ligand first (equal weight per ligand, matching the macro convention), then k
    per_ligand = detail.groupby(["model", "k", "extractant"])[
        [m for m in methods if m in detail.columns]].mean().reset_index()
    summary = per_ligand.groupby(["model", "k"])[
        [m for m in methods if m in per_ligand.columns]].mean().reset_index()
    summary["n_ligands"] = per_ligand.groupby(["model", "k"]).size().to_numpy()
    summary.to_csv(args.out / "kshot_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    (args.out / "summary.json").write_text(json.dumps({
        "source_oof": str(args.oof), "draws": args.draws,
        "k_values": list(K_VALUES), "n_models": int(oof["model"].nunique()),
    }, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
