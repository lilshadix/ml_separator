"""Score a finished ladder: per-extractant table, leaderboard, paired contrasts.

    .venv/Scripts/python.exe generations/gen13_separation/scripts/g13_analysis.py --label B_primary

The registered contrast set (P1, S1-S4; PRE_REGISTRATION.md §5) is the default; anything
passed through --extra-contrasts is labelled exploratory.  Outputs under metrics/<label>/
and bootstrap/<label>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen13sep import paths  # noqa: E402
from gen13sep.inference import leave_one_chemotype_out, paired_contrasts  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402
from gen13sep.metrics import (effective_sample_size, heavier_always_sign_accuracy,  # noqa: E402
                              per_extractant, summarise)
from gen13sep.runner import load_pair_table  # noqa: E402

REGISTERED = {
    "P1_M_SELECTED_vs_C_DIRECT_ROW": ("C_DIRECT_ROW", "M_SELECTED"),
    "S1_M_SELECTED_vs_B1_MEAN_CURVE": ("B1_MEAN_CURVE", "M_SELECTED"),
    "S2_M_SELECTED_vs_B3_NN_TANIMOTO": ("B3_NN_TANIMOTO", "M_SELECTED"),
    "S3_M_SELECTED_vs_B2_PAIRMEAN": ("B2_PAIRMEAN", "M_SELECTED"),
    "S4_M_SELECTED_vs_B4_HEAVIER_ALWAYS": ("B4_HEAVIER_ALWAYS", "M_SELECTED"),
}
PRIMARY_VALUE = "mae_all"
SECONDARY_VALUES = ("mae_far", "mae_adjacent", "sign_acc_strong", "pair_spearman", "curve_spearman")


def curve_level_spearman(label: str, cohort_frame: pd.DataFrame, arms: list[str]) -> pd.DataFrame | None:
    """Per (seed, cell, arm) Spearman between observed and predicted centred curves (>= 4 metals)."""
    cdir = paths.PREDICTION_DIR / label / "curves"
    if not cdir.exists():
        return None
    from scipy.stats import spearmanr
    ycols = [f"logD__{m}" for m in LANTHANIDES]
    Y = cohort_frame.set_index("cell_id")[ycols]
    rows = []
    for arm in arms:
        p = cdir / f"{arm}.parquet"
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        C = t[[f"c__{m}" for m in LANTHANIDES]].to_numpy(float)
        obs = Y.reindex(t["cell_id"]).to_numpy(float)
        for i in range(len(t)):
            m = ~np.isnan(obs[i])
            if m.sum() >= 4 and np.ptp(obs[i, m]) > 0 and np.ptp(C[i, m]) > 0:
                r = float(spearmanr(obs[i, m] - obs[i, m].mean(), C[i, m]).statistic)
            else:
                r = np.nan
            rows.append({"split_seed": int(t["split_seed"].iat[i]), "cell_id": t["cell_id"].iat[i], "arm": arm, "curve_spearman": r})
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="B_primary")
    ap.add_argument("--extra-contrasts", default="", help="exploratory candidate:reference pairs, comma separated")
    ap.add_argument("--replicates", type=int, default=10_000)
    args = ap.parse_args()

    pairs, arms = load_pair_table(args.label)
    sim = pd.read_parquet(paths.PREDICTION_DIR / args.label / "similarity.parquet")
    pairs = pairs.merge(sim[["split_seed", "fold", "cell_id", "max_train_tanimoto", "band"]],
                        on=["split_seed", "fold", "cell_id"], how="left", validate="many_to_one")
    spec = json.load(open(paths.PREDICTION_DIR / args.label / "run_spec.json", encoding="utf-8"))
    cohort_frame = pd.read_parquet(paths.MANIFEST_DIR / f"cohort_{spec.get('key_mode', 'exact')}.parquet")
    metric_dir = paths.METRIC_DIR / args.label
    boot_dir = paths.BOOTSTRAP_DIR / args.label
    metric_dir.mkdir(parents=True, exist_ok=True); boot_dir.mkdir(parents=True, exist_ok=True)

    cs = curve_level_spearman(args.label, cohort_frame, arms)
    per_ext = per_extractant(pairs, arms, curve_spearman=cs)
    per_ext.to_csv(metric_dir / "per_extractant.csv", index=False)
    board = summarise(per_ext, pairs, arms)
    board.to_csv(metric_dir / "leaderboard.csv", index=False)

    counts = per_ext[per_ext["arm"] == arms[0]].groupby("chemotype")["extractant"].nunique()
    context = {
        "label": args.label, "arms": arms,
        "n_pairs_per_seed": int(len(pairs) / pairs["split_seed"].nunique()),
        "n_extractants": int(pairs["extractant"].nunique()), "n_chemotypes": int(pairs["chemotype"].nunique()),
        "kish_n_eff_chemotypes": effective_sample_size(counts.values),
        "heavier_always_sign_accuracy_strong_macro": heavier_always_sign_accuracy(pairs, macro=True),
        "heavier_always_sign_accuracy_strong_pooled": heavier_always_sign_accuracy(pairs, macro=False),
        "adjacent_pairs_per_seed": int(per_ext[per_ext["arm"] == arms[0]]["n_adjacent"].sum() / pairs["split_seed"].nunique()),
        "far_pairs_per_seed": int((pairs["dZ"] >= 5).sum() / pairs["split_seed"].nunique()),
        "curve_spearman_available": cs is not None,
    }
    with open(metric_dir / "context.json", "w", encoding="utf-8") as fh:
        json.dump(context, fh, indent=2)

    # registered contrasts (only those whose arms exist in this run) + exploratory extras
    comps = {k: v for k, v in REGISTERED.items() if v[0] in arms and v[1] in arms}
    extra = {}
    for item in [x for x in args.extra_contrasts.split(",") if x]:
        cand, ref = item.split(":")
        extra[f"X_{cand}_vs_{ref}"] = (ref, cand)
    tables = []
    for name, cset in (("registered", comps), ("exploratory", extra)):
        if not cset:
            continue
        for value in (PRIMARY_VALUE,) + SECONDARY_VALUES:
            if value not in per_ext.columns:
                continue
            sub = per_ext.dropna(subset=[value])
            if sub["extractant"].nunique() < 3:
                continue
            t = paired_contrasts(sub, cset, value=value, replicates=args.replicates)
            t.insert(1, "status", name)
            tables.append(t)
    if tables:
        boot = pd.concat(tables, ignore_index=True)
        # the registered P1 rule applies to mae_all only; other values are reported with intervals
        boot["passes_P1"] = boot["passes_P1"].astype(object)
        boot.loc[boot["value"] != PRIMARY_VALUE, "passes_P1"] = None
        boot.to_csv(boot_dir / "paired_contrasts.csv", index=False)
        loco = []
        for name, (ref, cand) in {**comps, **extra}.items():
            t = leave_one_chemotype_out(per_ext, ref, cand); t.insert(0, "comparison", name); loco.append(t)
        pd.concat(loco, ignore_index=True).to_csv(boot_dir / "leave_one_chemotype_out.csv", index=False)
        # per-band contrasts (descriptive: bands are per held-out cell and not a partition of extractants)
        band_rows = []
        for band, block in pairs.groupby("band"):
            pe = per_extractant(block, arms)
            t = paired_contrasts(pe.dropna(subset=[PRIMARY_VALUE]), {**comps, **extra}, value=PRIMARY_VALUE,
                                 replicates=max(2000, args.replicates // 5))
            t.insert(0, "band", band); band_rows.append(t)
        if band_rows:
            pd.concat(band_rows, ignore_index=True).to_csv(boot_dir / "band_contrasts.csv", index=False)
        show = boot[boot["value"] == PRIMARY_VALUE]
        print(show[["comparison", "point", "ci95_low", "ci95_high", "bca_low", "bca_high", "p_two_sided", "mde_80",
                    "seeds_positive", "n_seeds", "units_improved", "n_units", "loco_sign_stable", "passes_P1"]].round(4).to_string(index=False))
    cols = ["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd", "macro_mae_chemotype", "macro_mae_adjacent",
            "macro_mae_far", "macro_sign_acc_strong", "macro_pair_spearman"] + (["macro_curve_spearman"] if "macro_curve_spearman" in board.columns else []) + ["pooled_mae"]
    print(board[cols].round(4).to_string(index=False))
    print(json.dumps(context, indent=1))


if __name__ == "__main__":
    main()
