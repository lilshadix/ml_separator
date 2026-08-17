"""k-shot per-extractant calibration study on frozen out-of-fold predictions.

Reads a wide OOF file (as written by ``run_gen4_candidates.py``) and asks: if
``k`` pairs of a *new* extractant are measured, how much does recalibrating the
frozen model on those ``k`` pairs reduce the error on the extractant's other
pairs?  No model is retrained.

Three things this script insists on, because the naive version of the question
gives an answer roughly twice too good:

* **Model-free null.**  A ``ZERO`` pseudo-arm (prediction ≡ 0) is added by
  default; with the ``trend`` form it is ``y ≈ b·ΔZ``, a two-parameter fit on the
  same k measurements that uses no model at all.  ``PAIRMEAN_baseline`` is a
  second null.  The model has to beat those, not merely beat ``k = 0``.
* **Free query rows.**  Inside one (extractant, condition) cell the target is
  exactly additive, so query pairs spanned by the support are arithmetic
  consequences of it.  ``--metric mae_free`` (the default) scores only the rows
  that are not.
* **Ligand transfer vs same experiment.**  The ``cross_condition`` policy holds
  out a whole condition and draws support from the extractant's other conditions.

Example::

    .venv/bin/python scripts/run_kshot_calibration.py \
        runs/gen4_candidates_20260817T003449Z/oof_predictions.csv \
        --arms A2_refit_TP A2_refit PAIRMEAN_baseline --n-draws 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen3_metrics import equal_group_macro_mae  # noqa: E402
from lanthanide_separation.kshot_calibration import (  # noqa: E402
    CALIBRATOR_FORMS,
    SUPPORT_POLICIES,
    TRANSITIVE_FORMS,
    KShotConfig,
    run_kshot_study,
    select_and_confirm,
    summarise_draw_deltas,
    summarise_grid,
    summarise_per_extractant,
)

DEFAULT_OOF = REPO_ROOT / "runs" / "gen4_candidates_20260817T003449Z" / "oof_predictions.csv"
ZERO_ARM = "ZERO"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("oof", nargs="?", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--arms", nargs="+", default=["A2_refit_TP", "A2_refit", "PAIRMEAN_baseline"])
    parser.add_argument("--no-zero-arm", action="store_true", help="drop the model-free y ~ b*dZ null")
    parser.add_argument("--split-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=104729)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 10])
    parser.add_argument("--forms", nargs="+", default=list(CALIBRATOR_FORMS), choices=list(CALIBRATOR_FORMS))
    parser.add_argument("--lams", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--policies", nargs="+", default=list(SUPPORT_POLICIES), choices=list(SUPPORT_POLICIES))
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--min-query", type=int, default=10)
    parser.add_argument("--min-pairs", type=int, default=20,
                        help="eligibility threshold, independent of k_max so runs with different --ks are comparable")
    parser.add_argument("--metric", default="mae_free", choices=["mae", "mae_free"],
                        help="mae_free scores only query rows not determined by the support (default)")
    parser.add_argument("--rng-seed", type=int, default=20260817)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def _fmt(df: pd.DataFrame, cols: list[str], nd: int = 4) -> str:
    present = [c for c in cols if c in df.columns]
    return df[present].round(nd).to_string(index=False)


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / "runs" / f"kshot_calibration_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    oof = pd.read_csv(args.oof)
    seeds = args.split_seeds or sorted(int(s) for s in oof["split_seed"].unique())
    arms = list(args.arms)
    policies = list(args.policies)
    n_draws = args.n_draws
    if args.quick:
        arms = arms[:1]
        seeds = seeds[:3]
        policies = [p for p in policies if p in ("random", "foreign", "cross_condition")]
        n_draws = 3
    if not args.no_zero_arm:
        oof[f"prediction_{ZERO_ARM}"] = 0.0
        arms = arms + [ZERO_ARM]
    if args.selection_seed not in seeds:
        raise SystemExit(f"selection seed {args.selection_seed} not among split seeds {seeds}")
    confirmation = [s for s in seeds if s != args.selection_seed]
    if not confirmation:
        raise SystemExit("need at least two split seeds (one selection + one confirmation)")
    cfg = KShotConfig(
        ks=tuple(sorted(set(int(k) for k in args.ks))),
        forms=tuple(args.forms), lams=tuple(float(x) for x in args.lams), policies=tuple(policies),
        n_draws=int(n_draws), min_query=int(args.min_query), min_pairs=int(args.min_pairs),
        rng_seed=int(args.rng_seed), include_oracle=not args.no_oracle,
    )
    log(f"oof {args.oof} rows {len(oof)}; arms {arms}; seeds {seeds}; selection {args.selection_seed}; "
        f"confirmation {confirmation}; metric {args.metric}")
    log(f"ks {cfg.ks}; forms {cfg.forms}; lams {cfg.lams}; policies {cfg.policies}; draws {cfg.n_draws}; "
        f"eligibility threshold {cfg.eligibility_threshold()} pairs")

    t0 = time.time()
    result = run_kshot_study(oof, arms=arms, seeds=seeds, cfg=cfg, log=log)
    log(f"study loop done in {time.time() - t0:.1f}s; {len(result.per_extractant_draw)} stat rows")

    per_ext = summarise_per_extractant(result)
    grid = summarise_grid(result)
    draw_deltas = summarise_draw_deltas(result)
    per_ext.to_csv(output_dir / "per_extractant_metrics.csv", index=False)
    grid.to_csv(output_dir / "grid_metrics.csv", index=False)
    draw_deltas.to_csv(output_dir / "draw_delta_distribution.csv", index=False)
    result.excluded_extractants.to_csv(output_dir / "excluded_extractants.csv", index=False)

    selection, decision = select_and_confirm(
        per_ext, grid, selection_seed=args.selection_seed, confirmation_seeds=confirmation,
        replicates=args.replicates, metric=args.metric,
    )
    selection.to_csv(output_dir / "selection.csv", index=False)
    decision.to_csv(output_dir / "decision_table.csv", index=False)

    macro_col = "macro_mae" if args.metric == "mae" else "macro_mae_free"
    grid_mean = (
        grid.groupby(["arm", "policy", "form", "lam", "k"], sort=True, observed=True)
        .agg(macro_mae=("macro_mae", "mean"), macro_mae_free=("macro_mae_free", "mean"),
             pooled_mae=("pooled_mae", "mean"), pooled_r2=("pooled_r2", "mean"),
             macro_r2_median=("macro_r2_median", "mean"), dispersion=("dispersion_ratio", "mean"),
             sign_acc=("sign_acc", "mean"), reject_frac=("reject_frac", "mean"),
             n_rows=("n_rows", "mean"), n_rows_free=("n_rows_free", "mean"), n_ext=("n_ext", "first"))
        .reset_index()
    )
    k0 = grid_mean[grid_mean.k == 0][["arm", "policy", "macro_mae", "macro_mae_free", "pooled_mae"]]
    k0 = k0.rename(columns={"macro_mae": "k0_macro_mae", "macro_mae_free": "k0_macro_mae_free",
                            "pooled_mae": "k0_pooled_mae"})
    grid_mean = grid_mean.merge(k0, on=["arm", "policy"], how="left")
    grid_mean["delta"] = grid_mean[f"k0_{macro_col}"] - grid_mean[macro_col]
    grid_mean.to_csv(output_dir / "grid_metrics_seedmean.csv", index=False)

    lines: list[str] = []
    A = lines.append
    A(f"k-shot calibration study — {stamp}")
    A(f"oof: {args.oof}")
    A(f"arms {arms}; {len(seeds)} split seeds {seeds}; selection {args.selection_seed}; confirmation {confirmation}")
    A(f"ks {cfg.ks}; lams {cfg.lams}; forms {cfg.forms}; policies {cfg.policies}; draws {cfg.n_draws}; "
      f"eligibility {cfg.eligibility_threshold()} pairs; decision metric {args.metric}")
    A("")
    A("READING THIS REPORT")
    A("  * macro MAE is the equal-extractant mean; k=0 is the same model on the same query rows.")
    A("  * macro_mae_free excludes query pairs that are exact arithmetic consequences of the support")
    A("    (within a condition, log_SF is exactly additive), so `free` is the only stratum where a")
    A("    prediction is actually needed.  It is the decision metric by default.")
    A("  * Query rows differ BETWEEN policies (each policy removes different pairs into its support),")
    A("    so compare deltas within a policy, never macro levels across policies.")
    A("  * `foreign` (support from a different extractant) is a control, never gated.")
    A("  * ZERO / PAIRMEAN_baseline are model-free nulls: what the k measurements buy on their own.")
    A("  * The 5 split seeds re-partition the SAME pairs, so per-seed agreement is weak evidence;")
    A("    seed_delta_spread shows how little they differ.")
    A("")

    # Baseline decomposition: published -> eligible subset -> policy query rows.
    A("BASELINE DECOMPOSITION (why k=0 here is not the published leaderboard macro)")
    elig = set(per_ext.extractant.unique())
    rows = []
    for arm in arms:
        col = f"prediction_{arm}"
        if col not in oof.columns:
            continue
        full = np.mean([equal_group_macro_mae(g[oof.columns[oof.columns.get_loc("log_SF_A_over_B")]],
                                              g[col], g["extractant"])
                        for _, g in oof[oof.split_seed.isin(seeds)].groupby("split_seed")])
        sub = oof[oof.extractant.isin(elig) & oof.split_seed.isin(seeds)]
        eligible = np.mean([equal_group_macro_mae(g["log_SF_A_over_B"], g[col], g["extractant"])
                            for _, g in sub.groupby("split_seed")])
        rr = {"arm": arm, "published_all_extractants": full, "eligible_extractants_all_rows": eligible}
        for pol in cfg.policies:
            cell = grid_mean[(grid_mean.arm == arm) & (grid_mean.policy == pol) & (grid_mean.k == 0)]
            if not cell.empty:
                rr[f"k0_{pol}"] = float(cell.macro_mae.iloc[0])
                rr[f"k0free_{pol}"] = float(cell.macro_mae_free.iloc[0])
        rows.append(rr)
    baseline_table = pd.DataFrame(rows)
    baseline_table.to_csv(output_dir / "baseline_decomposition.csv", index=False)
    A(baseline_table.round(4).to_string(index=False))
    if not result.excluded_extractants.empty:
        ex = result.excluded_extractants.drop_duplicates(["extractant"])[["extractant", "n_pairs"]]
        A(f"excluded {len(ex)} extractant(s) with < {cfg.eligibility_threshold()} pairs: "
          + ", ".join(f"{e[:34]}…({n})" for e, n in ex.itertuples(index=False)))
    A("")

    # Head-to-head against the model-free nulls, at the selected recipe per (policy, k).
    A("HEAD-TO-HEAD vs MODEL-FREE NULLS (decision metric, mean over confirmation seeds)")
    nulls = [a for a in (ZERO_ARM, "PAIRMEAN_baseline") if a in arms]
    if nulls:
        h2h = []
        for pol in cfg.policies:
            for k in (0,) + cfg.ks:
                row = {"policy": pol, "k": k}
                for arm in arms:
                    cell = grid_mean[(grid_mean.arm == arm) & (grid_mean.policy == pol) & (grid_mean.k == k)]
                    if cell.empty:
                        continue
                    row[arm] = float(cell.sort_values(macro_col)[macro_col].iloc[0])
                h2h.append(row)
        h2h = pd.DataFrame(h2h)
        for arm in arms:
            for null in nulls:
                if arm not in (ZERO_ARM, "PAIRMEAN_baseline") and arm in h2h and null in h2h:
                    h2h[f"{arm}_vs_{null}"] = h2h[null] - h2h[arm]
        h2h.to_csv(output_dir / "head_to_head_vs_nulls.csv", index=False)
        A(h2h.round(4).to_string(index=False))
        A("  (best form/lam per cell — an optimistic view of every arm alike; the gated numbers are below)")
    A("")

    for arm in arms:
        A("")
        A(f"=== base arm {arm} ===")
        gm = grid_mean[grid_mean.arm == arm]
        A("k = 0 (raw predictions), mean over seeds, per policy:")
        A(_fmt(gm[gm.k == 0], ["policy", "macro_mae", "macro_mae_free", "n_rows", "n_rows_free",
                               "pooled_mae", "pooled_r2", "dispersion", "sign_acc"]))
        orc = gm[gm.policy == "oracle"]
        if not orc.empty:
            A("in-sample least-squares reference (fitted on ALL rows of the extractant — approximate ceiling,")
            A("not a strict MAE upper bound; scored on the random policy's query rows):")
            A(_fmt(orc, ["form", "macro_mae", "macro_mae_free", "pooled_mae", "pooled_r2",
                         "macro_r2_median", "dispersion", "sign_acc"]))
        best = (gm[(gm.k > 0) & (gm.policy != "oracle")]
                .sort_values(["policy", "k", macro_col], kind="stable")
                .groupby(["policy", "k"], sort=True, observed=True).head(1))
        A(f"best (form, lam) per policy and k by {len(seeds)}-seed-mean {macro_col} — exploratory, "
          "each row carries its own k=0 baseline:")
        A(_fmt(best, ["policy", "k", "form", "lam", f"k0_{macro_col}", macro_col, "delta",
                      "pooled_mae", "pooled_r2", "sign_acc", "reject_frac"]))
        for form in ("scale", "trend"):
            if form not in cfg.forms:
                continue
            sub = gm[(gm.form == form) & np.isclose(gm.lam, 1.0) & (gm.policy == "random")]
            if not sub.empty:
                A(f"fixed recipe {form} lam=1.0 (random policy, {macro_col}): "
                  + "  ".join(f"k={int(r.k)}:{getattr(r, macro_col):.4f}" for r in sub.itertuples()))
        dd = draw_deltas[draw_deltas.arm == arm]
        if not dd.empty:
            merged = best.merge(dd, on=["policy", "k", "form", "lam"], how="left")
            A("single-draw risk for those recipes (one chemist gets ONE set of k pairs, not the average):")
            A(_fmt(merged, ["policy", "k", "form", "lam", "delta_mean", "delta_sd", "delta_q10",
                            "frac_harmful", "delta_free_mean", "frac_harmful_free"]))
        dsel = selection[selection.arm == arm]
        ddec = decision[decision.arm == arm]
        if not dsel.empty:
            A("")
            A(f"selection on seed {args.selection_seed}:")
            A(_fmt(dsel, ["policy", "k", "form", "lam", "n_configs_screened", "macro_k0_selection",
                          "macro_selection", "delta_selection"]))
        if not ddec.empty:
            A("")
            A(f"confirmation on {confirmation} (delta = {macro_col}(k=0) − {macro_col}(calibrated), "
              "same query rows; positive = better):")
            A(_fmt(ddec, ["policy", "k", "form", "lam", "is_control", "macro_k0_confirm", "macro_confirm",
                          "delta_mean_confirm", "positive_confirm_seeds", "seed_delta_spread",
                          "pooled_mae_k0_confirm", "pooled_mae_confirm",
                          "boot_delta_confirm", "ci95_low", "ci95_high", "p_worse_one_sided",
                          "extractants_improved", "n_extractants", "holm_p_one_sided", "passes_rule"]))
            nontrans = ddec[(~ddec.form.isin(TRANSITIVE_FORMS)) & ddec.passes_rule]
            if arm.endswith("_TP") and not nontrans.empty:
                A("  WARNING: the selected form breaks within-cell transitivity on a _TP arm "
                  "(offset/affine); the calibrated field no longer chains La-Ce + Ce-Pr = La-Pr:")
                A(_fmt(nontrans, ["policy", "k", "form"]))

    report = "\n".join(lines)
    (output_dir / "decision_report.txt").write_text(report + "\n")
    print(report)
    summary = {
        "oof": str(args.oof), "arms": arms, "seeds": seeds, "selection_seed": args.selection_seed,
        "confirmation_seeds": confirmation, "decision_metric": args.metric,
        "config": {"ks": list(cfg.ks), "forms": list(cfg.forms), "lams": list(cfg.lams),
                   "policies": list(cfg.policies), "n_draws": cfg.n_draws, "min_query": cfg.min_query,
                   "min_pairs": cfg.min_pairs, "eligibility_threshold": cfg.eligibility_threshold(),
                   "rng_seed": cfg.rng_seed, "include_oracle": cfg.include_oracle,
                   "coef_bound": cfg.coef_bound, "k_max": cfg.k_max},
        "replicates": args.replicates,
        "n_configs_per_cell": len(cfg.forms) * len(cfg.lams),
        "holm_family_size": int(decision.holm_family_size.iloc[0]) if not decision.empty else 0,
        "excluded_extractants": result.excluded_extractants.drop_duplicates(["extractant"])[
            ["extractant", "n_pairs"]].to_dict("records"),
        "baseline_decomposition": baseline_table.to_dict("records"),
        "decision": decision.to_dict("records"),
        "selection": selection.to_dict("records"),
        "note": "post-hoc k-shot per-extractant calibration on frozen OOF predictions; exploratory, not a protocol run",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
