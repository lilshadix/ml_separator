"""Compose the headline tables from the metrics/ and bootstrap/ directories.

    .venv/Scripts/python.exe gen13_separation/scripts/g13_headline_tables.py --primary B_primary \
        --ablations B_abl_cond_only,B_abl_ecfp_only,B_abl_no_coord,B_abl_coord_donors

Writes headline_tables/t1_leaderboard.{csv,md}, t2_contrasts, t3_bands, t4_ablations, t5_selection,
t6_fewshot (if present), t7_loco.  Every table carries its regime in its caption.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen13sep import paths  # noqa: E402


def md(df: pd.DataFrame, caption: str, floats: int = 4) -> str:
    body = df.round(floats).to_markdown(index=False)
    return f"*{caption}*\n\n{body}\n"


def write(name: str, df: pd.DataFrame, caption: str) -> None:
    df.to_csv(paths.HEADLINE_DIR / f"{name}.csv", index=False)
    (paths.HEADLINE_DIR / f"{name}.md").write_text(md(df, caption), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="B_primary")
    ap.add_argument("--ablations", default="")
    args = ap.parse_args()
    mdir = paths.METRIC_DIR / args.primary
    bdir = paths.BOOTSTRAP_DIR / args.primary
    ctx = json.load(open(mdir / "context.json", encoding="utf-8"))
    regime = (f"design B chemotype hold-out, 5 split seeds, extractant-macro over all held-out pairs; "
              f"{ctx['n_extractants']} extractants / {ctx['n_chemotypes']} chemotypes, Kish n_eff "
              f"{ctx['kish_n_eff_chemotypes']:.1f}; 'heavier always' sign accuracy on |log SF| >= 0.3 = "
              f"{ctx.get('heavier_always_sign_accuracy_strong_macro', float('nan')):.3f} macro / "
              f"{ctx.get('heavier_always_sign_accuracy_strong_pooled', float('nan')):.3f} pooled")

    board = pd.read_csv(mdir / "leaderboard.csv")
    cols = [c for c in ["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd", "macro_mae_chemotype",
                        "macro_mae_adjacent", "macro_mae_far", "macro_sign_acc_strong", "macro_pair_spearman",
                        "macro_curve_spearman", "pooled_mae"] if c in board.columns]
    write("t1_leaderboard", board[cols], f"T1 zero-shot leaderboard — {regime}")

    if (bdir / "paired_contrasts.csv").exists():
        boot = pd.read_csv(bdir / "paired_contrasts.csv")
        cols = [c for c in ["comparison", "status", "value", "point", "ci95_low", "ci95_high", "bca_low", "bca_high",
                            "p_two_sided", "mde_80", "seeds_positive", "n_seeds", "units_improved", "n_units",
                            "loco_min", "loco_max", "loco_sign_stable", "passes_intervals", "passes_P1"] if c in boot.columns]
        write("t2_contrasts", boot[cols], "T2 paired contrasts (reference − candidate per extractant, averaged over seeds; "
              "positive favours the candidate; higher-is-better metrics negated first); chemotype-blocked bootstrap "
              "10,000 shared draws; passes_P1 = margin 0.02 + both intervals + p < 0.05 + >= 4/5 seeds + LOCO sign-stable, "
              "evaluated on mae_all only")
    if (bdir / "leave_one_chemotype_out.csv").exists():
        loco = pd.read_csv(bdir / "leave_one_chemotype_out.csv")
        summary = (loco[loco["dropped_chemotype"] != "(none)"].groupby("comparison")["delta"]
                   .agg(["min", "max"]).reset_index())
        full = loco[loco["dropped_chemotype"] == "(none)"][["comparison", "delta"]].rename(columns={"delta": "full"})
        write("t7_loco", full.merge(summary, on="comparison"), "T7 leave-one-chemotype-out range of the primary deltas")

    if (bdir / "band_contrasts.csv").exists():
        bc = pd.read_csv(bdir / "band_contrasts.csv")
        cols = [c for c in ["band", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided", "mde_80", "n_units"] if c in bc.columns]
        write("t3_bands", bc[cols], "T3 per-band paired contrasts on mae_all (band = max training Tanimoto of the held-out "
              "cell: far ≤ 0.40, mid 0.40–0.60, near > 0.60; descriptive — bands are not a partition of extractants)")

    sel_path = paths.PREDICTION_DIR / args.primary / "selections.csv"
    if sel_path.exists():
        sel = pd.read_csv(sel_path)
        counts = sel["selected"].value_counts().rename_axis("selected").reset_index(name="folds")
        write("t5_selection", counts, "T5 inner-validation selection of M_SELECTED over the 25 (seed, fold) pairs")

    if args.ablations:
        rows = []
        for label in [args.primary] + args.ablations.split(","):
            p = paths.METRIC_DIR / label / "leaderboard.csv"
            if not p.exists():
                continue
            b = pd.read_csv(p)
            spec = json.load(open(paths.PREDICTION_DIR / label / "run_spec.json", encoding="utf-8"))
            for _, r in b.iterrows():
                if r["arm"] in ("M_SELECTED", "C_DIRECT_ROW", "B1_MEAN_CURVE") or r["arm"].startswith("X_") or r["arm"].startswith("M_PHYSICS"):
                    rows.append({"label": label, "blocks": "+".join(spec["blocks"]), "arm": r["arm"],
                                 "macro_mae_extractant": r["macro_mae_extractant"], "macro_mae_far": r["macro_mae_far"],
                                 "macro_sign_acc_strong": r["macro_sign_acc_strong"],
                                 "macro_pair_spearman": r.get("macro_pair_spearman", float("nan"))})
        write("t4_ablations", pd.DataFrame(rows), "T4 feature-block ablations (same folds, same learners)")

    fs = mdir / "fewshot_leaderboard.csv"
    if fs.exists():
        t = pd.read_csv(fs)
        write("t6_fewshot", t, "T6 one-pair calibration: macro MAE on the remaining pairs of cells with >= 3 metals, "
              "5 support draws per cell")
    print("headline tables written to", paths.HEADLINE_DIR)


if __name__ == "__main__":
    main()
