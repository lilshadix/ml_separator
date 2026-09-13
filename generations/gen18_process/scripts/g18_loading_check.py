"""Script 6 of DESIGN.md section 11: the loading endpoint E2 and decision rule R2.

On the publication-aware loading series (E2 cohort of PRE_REGISTRATION.md section 2; the rows
of ``systems/corpus_records.csv`` carrying a ``loading_series_id``), compare C0 (the tracer log D
everywhere) with C1 (ideal ligand depletion: ``log K`` anchored so that ``equilibrium.solve_stage``
reproduces the tracer point; every other point predicted by the same stage solve at O/A 1 with
``n`` = the system's in-sample M1 slope from ``results/dmodels/<id>.json``, else the prior
``n0 = 3``), through ``gen18proc.evalproto.loading_series_evaluate``.  Applies R2 (section 7):
C1 is supported iff it wins on >= 6 of 10 series and the paired series bootstrap of
mean(E2(C0) - E2(C1)) excludes zero; every series counts as it falls.  Secondary: the
loading-active subset, O/A in ``--oa``, the descriptive Spearman; exploratory: X1 (effective
capacity ``phi``, leave-one-point-out) and X5 (publication-blind series from
``systems/series.csv``).

Writes ``results/loading/e2_series.csv``, ``e2_points.csv``, ``e2_summary.csv``,
``phi_exploratory.csv``, ``e2_series_publication_blind.csv``, ``decision.json``, ``DECISION.md``
(the R2 verdict verbatim; the sourced TODGA LOC of the pre-fit addendum as interpretation, never
as an exclusion), ``manifest.json``, and appends the E2 contrasts to
``results/eval/comparisons.csv``.  Refuses unless ``g18_seal_prereg.py --check`` passes, EXCEPT
under ``--unsealed-dry-run`` (series of the two-system subset, outputs under
``results/_dryrun/loading/``).

Usage (from the repository root, one process):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_loading_check.py \\
        [--oa 0.5 1 2] [--seed 18] [--n-boot 2000] [--no-phi] [--unsealed-dry-run]

No wall-clock value is written.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g18_fit_dmodels import DRYRUN_DIR, DRYRUN_SYSTEMS, primary_ligand, require_sealed  # noqa: E402
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc.evalproto import (  # noqa: E402
    DEFAULT_BAND,
    ComparisonCounter,
    fit_mass_action,
    loading_series_evaluate,
    paired_system_bootstrap,
)
from gen18proc.report import read_table, write_manifest, write_table  # noqa: E402
from gen18proc.systems import load_registry  # noqa: E402

TODGA_SYSTEM = "sys_5cb78e5000d40860"
SASAKI_SERIES = "ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1"
TODGA_LOC_M = 0.008
"""mol/L Nd in 0.1 M TODGA / n-dodecane / 3 M HNO3 (doi 10.1081/SEI-120016073, abstract)."""
X1_MARGIN = 0.10

R2_RULE_VERBATIM = (
    "**E2** = per-series MAE of predicted log D over the non-tracer points, for C0 and C1, macro "
    "over the 10 series (each series weight 1). **R2.** C1 is *supported* if and only if (i) "
    "E2(C1) < E2(C0) on more than half of the series (>= 6 of 10) and (ii) the paired "
    "series-level bootstrap (2000 resamples, seed 18) 95 % interval of mean(E2(C0) - E2(C1)) "
    "excludes zero. **Every series counts as it falls**: the three flat pub_d3c970567f series "
    "and the rising TODGA/Ce series are expected to be wins for C0 and are not excluded, "
    "down-weighted or explained away; if they make R2 fail, the verdict is \"the direction of "
    "the loading effect is not consistent across corpus publications and the ideal correction "
    "is not supported corpus-wide\", and the process chain keeps the depletion term *as a "
    "mechanism with a flag* (`OA_ASSUMED`, `HIGH_LOADING`), not as a validated magnitude.")
"""PRE_REGISTRATION.md section 7, quoted verbatim."""

VERDICT_SUPPORTED = "C1 is supported"
VERDICT_NULL = ("the direction of the loading effect is not consistent across corpus "
                "publications and the ideal correction is not supported corpus-wide")


def load_fits(systems: list[str], records: pd.DataFrame, registry: pd.DataFrame,
              dmodels_dir: Path, band: str, seed: int) -> tuple[dict[str, Any], dict[str, str]]:
    """``n`` per E2 system: the fitted block's in-sample slope when the fit script wrote it,
    else an in-sample fit here (the prior applies inside ``loading_series_evaluate`` when the
    slope was fixed or no fit exists)."""
    fits: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for sid in systems:
        path = dmodels_dir / f"{sid}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                f = json.load(fh)
            fits[sid] = {"n": f.get("n"), "slope_status": f.get("slope_status") or {},
                         "ligand": f.get("ligand")}
            sources[sid] = f"results/dmodels/{sid}.json"
            continue
        sub = records.loc[records["system_id"] == sid]
        ligand = primary_ligand(registry, sid)
        fit = fit_mass_action(sub, ligand=ligand, band=band, seed=seed, reliability=False)
        if fit.status == "fitted":
            fits[sid] = fit
            sources[sid] = "in-sample fit computed here (no results/dmodels file)"
        else:
            sources[sid] = "no fittable data: prior n0"
    return fits, sources


def r2_summary(table: pd.DataFrame, n_boot: int, seed: int, label: str, oa: float,
               ) -> dict[str, Any]:
    t = table.loc[table["status"] == "scored"]
    n = int(len(t))
    if n == 0:
        return {"subset": label, "oa": oa, "n_series": 0}
    c0 = t["mae_constant"].to_numpy(dtype=float)
    c1 = t["mae_ideal"].to_numpy(dtype=float)
    wins = int((c1 < c0).sum())
    mean_d, lo, hi = paired_system_bootstrap(c0, c1, n_boot=n_boot, seed=seed)
    need = n // 2 + 1
    cond_i = wins >= need
    cond_ii = bool(np.isfinite(lo) and lo > 0.0)
    return {"subset": label, "oa": oa, "n_series": n, "e2_c0_macro": float(np.mean(c0)),
            "e2_c1_macro": float(np.mean(c1)), "wins_c1": wins, "wins_needed": need,
            "bootstrap_mean_c0_minus_c1": mean_d, "bootstrap_lo": lo, "bootstrap_hi": hi,
            "condition_i_majority": bool(cond_i), "condition_ii_ci_excludes_zero": cond_ii,
            "c1_supported": bool(cond_i and cond_ii),
            "spearman_negative_count": int((t["spearman_logD_vs_logmM"] < 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oa", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--band", default=DEFAULT_BAND)
    ap.add_argument("--no-phi", action="store_true", help="skip the exploratory phi arm")
    ap.add_argument("--unsealed-dry-run", action="store_true")
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    args = ap.parse_args()
    seal_msg = require_sealed(args.unsealed_dry_run)
    if 1.0 not in args.oa:
        args.oa = [1.0] + list(args.oa)
    systems_dir = Path(args.systems_dir)
    if args.unsealed_dry_run:
        out_dir = DRYRUN_DIR / "loading"
        dmodels_dir = DRYRUN_DIR / "dmodels"
        eval_dir = DRYRUN_DIR / "eval"
    else:
        out_dir = paths.RESULTS_LOADING_DIR
        dmodels_dir = paths.RESULTS_DMODELS_DIR
        eval_dir = paths.RESULTS_EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = systems_dir / "corpus_records.csv"
    series_path = systems_dir / "series.csv"
    records = pd.read_csv(records_path, low_memory=False)
    series = pd.read_csv(series_path)
    registry = load_registry(systems_dir)
    has_series = records["loading_series_id"].astype("string").fillna("") != ""
    e2 = records.loc[has_series].copy()
    if args.unsealed_dry_run:
        e2 = e2.loc[e2["system_id"].isin(DRYRUN_SYSTEMS)].copy()
    systems = sorted(set(e2["system_id"]))
    fits, n_sources = load_fits(systems, records, registry, dmodels_dir, args.band, args.seed)
    cohort_tag = "E2 cohort: publication-aware loading series%s" % (
        " (DRY-RUN subset)" if args.unsealed_dry_run else "")
    outputs: list[Path] = []

    # ---- primary and O/A sensitivity ----------------------------------------------------------
    tables, points = [], []
    for oa in args.oa:
        t = loading_series_evaluate(e2, fits, oa=float(oa), phi_arm=(not args.no_phi and oa == 1.0))
        t.insert(0, "definition", "publication_aware")
        p = pd.DataFrame(t.attrs.pop("points", []))
        tables.append(t)
        if len(p):
            points.append(p)
    e2_series = pd.concat(tables, ignore_index=True)
    e2_series["n_source_file"] = e2_series["system_id"].map(n_sources).fillna("prior n0")
    outputs.append(write_table(e2_series, out_dir / "e2_series.csv", regime={
        "cohort": cohort_tag, "holdout": "tracer point anchors log K; scored on the other points "
                                        "of the same series (oa = 1 primary; other oa secondary)",
        "averaging_unit": "series", "status_of_parameters": "n from the in-sample M1 fit or "
                                                            "prior n0; log K anchored per series"},
        regime_table=True))
    e2_points = pd.concat(points, ignore_index=True) if points else pd.DataFrame()
    outputs.append(write_table(e2_points, out_dir / "e2_points.csv", regime={
        "cohort": cohort_tag, "holdout": "as e2_series.csv", "averaging_unit": "point"}))

    # ---- summaries and R2 ----------------------------------------------------------------------
    summaries = []
    for oa in args.oa:
        t = e2_series.loc[e2_series["oa"] == float(oa)]
        summaries.append(r2_summary(t, args.n_boot, args.seed, "all", float(oa)))
        summaries.append(r2_summary(t.loc[t["loading_active"].astype(bool)], args.n_boot,
                                    args.seed, "loading_active", float(oa)))
    primary = next(s for s in summaries if s["subset"] == "all" and s["oa"] == 1.0)
    supported = bool(primary.get("c1_supported", False))

    # ---- X1 phi arm ----------------------------------------------------------------------------
    t1 = e2_series.loc[(e2_series["oa"] == 1.0) & (e2_series["phi_loo_n"] > 0)].copy()
    x1_supported = None
    if len(t1):
        t1["x1_beats_c1_by_margin"] = t1["mae_phi_loo"] < t1["mae_ideal"] - X1_MARGIN
        active = t1.loc[t1["loading_active"].astype(bool)]
        n_active_wins = int(active["x1_beats_c1_by_margin"].sum())
        x1_supported = bool(n_active_wins >= 4)
        phi_table = t1[["loading_series_id", "system_id", "metal", "n_non_tracer", "mae_constant",
                        "mae_ideal", "mae_phi_loo", "phi_loo_n", "phi_median", "phi_min",
                        "phi_max", "phi_insample", "mae_phi_insample", "loading_active",
                        "x1_beats_c1_by_margin"]]
    else:
        n_active_wins = 0
        phi_table = pd.DataFrame(columns=["loading_series_id"])
    outputs.append(write_table(phi_table, out_dir / "phi_exploratory.csv", regime={
        "cohort": cohort_tag + "; series with >= 4 non-tracer points",
        "holdout": "leave-one-point-out (phi fitted on the rest, log K re-anchored per phi)",
        "averaging_unit": "series", "status_of_parameters": "exploratory X1; never used elsewhere"},
        regime_table=True))

    # ---- X5 publication-blind series ---------------------------------------------------------
    blind = series.loc[series["definition"] == "publication_blind"]
    x5_rows = []
    for r in blind.itertuples():
        ids = str(r.record_ids).split(";")
        sub = records.loc[records["record_id"].isin(ids)].copy()
        if args.unsealed_dry_run and str(r.system_id) not in DRYRUN_SYSTEMS:
            continue
        sub["loading_series_id"] = str(r.loading_series_id)
        sub["is_tracer"] = sub["record_id"] == str(r.tracer_record_id)
        x5_rows.append(sub)
    x5_summary: dict[str, Any] = {"subset": "publication_blind", "oa": 1.0, "n_series": 0}
    if x5_rows:
        x5 = loading_series_evaluate(pd.concat(x5_rows, ignore_index=True), fits, oa=1.0,
                                     phi_arm=False)
        x5.insert(0, "definition", "publication_blind")
        x5_summary = r2_summary(x5, args.n_boot, args.seed, "publication_blind", 1.0)
    else:
        x5 = pd.DataFrame(columns=["loading_series_id"])
    outputs.append(write_table(x5, out_dir / "e2_series_publication_blind.csv", regime={
        "cohort": "publication-blind loading series (exploratory X5)",
        "holdout": "as e2_series.csv", "averaging_unit": "series",
        "status_of_parameters": "as e2_series.csv"}, regime_table=True))
    summaries.append(x5_summary)
    summary = pd.DataFrame(summaries)
    outputs.append(write_table(summary, out_dir / "e2_summary.csv", regime={
        "cohort": cohort_tag, "holdout": "as e2_series.csv",
        "averaging_unit": "series (macro, weight 1 each); paired series bootstrap "
                          f"({args.n_boot} resamples, seed {args.seed})"}))

    # ---- LOC interpretation of the Sasaki series ---------------------------------------------
    loc_lines: list[str] = []
    sas = e2_series.loc[(e2_series["oa"] == 1.0)
                        & (e2_series["loading_series_id"] == SASAKI_SERIES)]
    if len(sas):
        srow = sas.iloc[0]
        pts = e2_points.loc[(e2_points["oa"] == 1.0)
                            & (e2_points["loading_series_id"] == SASAKI_SERIES)]
        n_used = float(srow["n_used"])
        lam = pts["loading_fraction_pred"].to_numpy(dtype=float)
        y_pred = lam * float(srow["ligand_M"]) / n_used
        above = int((y_pred > TODGA_LOC_M).sum())
        loc_lines = [
            "## Interpretation of the Sasaki 2015 TODGA/Nd series with the sourced LOC "
            "(addenda/ORCHESTRATOR_prefit_20260913.md, U6; interpretation only, no exclusion)",
            "",
            f"Series `{SASAKI_SERIES}` (0.1 M TODGA, n-dodecane, 3 M HNO3, 4.9-12 mM Nd, D 23.5 -> "
            "0.60) counted as it fell: mae_constant = %.3f, mae_ideal = %.3f, C1 wins = %s. The "
            "sourced third-phase limit for this system is 0.008 M Nd in the organic "
            "(Tachimori, Sasaki, Suzuki 2002, doi 10.1081/SEI-120016073, abstract level). The C1 "
            "prediction puts the organic Nd above that limit at %d of %d non-tracer points "
            "(predicted organic Nd up to %.4f M); the collapse of D along this series is therefore "
            "read as a third-phase boundary, not as ligand depletion, and the cascade raises "
            "`THIRD_PHASE_RISK` above the sourced value for this system. This reading does not "
            "change R2: the series keeps its weight." % (
                float(srow["mae_constant"]), float(srow["mae_ideal"]), bool(srow["c1_wins"]),
                above, len(pts), float(np.nanmax(y_pred)) if len(y_pred) else math.nan), ""]

    # ---- decision ---------------------------------------------------------------------------
    decision = {
        "schema": "gen18.decision.1", "rule": "R2", "c1_supported": supported,
        "dry_run": bool(args.unsealed_dry_run), "primary": primary,
        "secondary": [s for s in summaries if not (s["subset"] == "all" and s["oa"] == 1.0)],
        "x1_supported": x1_supported, "x1_active_wins_by_margin": n_active_wins,
        "verdict": VERDICT_SUPPORTED if supported else VERDICT_NULL,
        "n_sources": n_sources,
    }
    dec_path = out_dir / "decision.json"
    dec_path.write_text(json.dumps(decision, indent=2, sort_keys=True, default=float) + "\n",
                        encoding="utf-8", newline="\n")
    outputs.append(dec_path)
    lines = ["# R2 decision (written by scripts/g18_loading_check.py)", ""]
    if args.unsealed_dry_run:
        lines += ["**UNSEALED DRY RUN on the series of %s: not a result; the rule is applied "
                  "to %d series for testing only.**" % (", ".join(DRYRUN_SYSTEMS),
                                                         primary.get("n_series", 0)), ""]
    lines += ["Regime: cohort = %s; hold-out = the tracer point anchors log K, the other points "
              "of the same series are scored; averaging unit = series (weight 1); parameters: "
              "n from the in-sample M1 fit (results/dmodels) or the prior n0 = 3, log K anchored "
              "per series, O/A = 1 assumed (OA_ASSUMED), K_H unknown." % cohort_tag, "",
              "## Rule (PRE_REGISTRATION.md section 7, verbatim)", "", R2_RULE_VERBATIM, "",
              "## Numbers (primary: all series, O/A 1)", "",
              "| quantity | value |", "|---|---|",
              f"| n_series | {primary.get('n_series')} |",
              f"| E2(C0) macro | {primary.get('e2_c0_macro', math.nan):.4f} |",
              f"| E2(C1) macro | {primary.get('e2_c1_macro', math.nan):.4f} |",
              f"| condition (i): C1 wins | {primary.get('wins_c1')} of {primary.get('n_series')} "
              f"(needed {primary.get('wins_needed')}): {primary.get('condition_i_majority')} |",
              f"| condition (ii): 95 % bootstrap interval of mean(E2(C0) - E2(C1)) | "
              f"[{primary.get('bootstrap_lo', math.nan):.4f}, "
              f"{primary.get('bootstrap_hi', math.nan):.4f}] (mean "
              f"{primary.get('bootstrap_mean_c0_minus_c1', math.nan):.4f}); excludes zero: "
              f"{primary.get('condition_ii_ci_excludes_zero')} |", "",
              "Per-series n source: "
              + "; ".join(f"{k}: {v}" for k, v in sorted(n_sources.items())),
              "", "## Verdict", "",
              ("**" + VERDICT_SUPPORTED + "** (both conditions hold)." if supported else
               "**" + VERDICT_NULL + "**; the process chain keeps the depletion term *as a "
               "mechanism with a flag* (`OA_ASSUMED`, `HIGH_LOADING`), not as a validated "
               "magnitude."), "",
              "## Secondary and exploratory (labelled)", "",
              "| subset | O/A | n | E2(C0) | E2(C1) | wins | CI of C0 - C1 | supported |",
              "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        if s.get("n_series", 0) == 0:
            lines.append(f"| {s['subset']} | {s['oa']} | 0 | | | | | |")
            continue
        lines.append("| %s | %s | %d | %.4f | %.4f | %d | [%.4f, %.4f] | %s |" % (
            s["subset"], s["oa"], s["n_series"], s["e2_c0_macro"], s["e2_c1_macro"], s["wins_c1"],
            s["bootstrap_lo"], s["bootstrap_hi"], s["c1_supported"]))
    lines += ["", "X1 (effective capacity phi, leave-one-point-out, exploratory): supported = "
              f"{x1_supported} ({n_active_wins} loading-active series where LOO-MAE(X1) < "
              f"E2(C1) - {X1_MARGIN}; rule needs >= 4). When not supported phi is a reported "
              "fudge and is used nowhere.", ""]
    lines += loc_lines
    dec_md = out_dir / "DECISION.md"
    dec_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    outputs.append(dec_md)

    # ---- comparisons.csv (append the E2 contrasts) --------------------------------------------
    cc = ComparisonCounter()
    comp_path = eval_dir / "comparisons.csv"
    mine = {"E2: C1 vs C0", "E2 secondary: loading-active subset", "E2 secondary: O/A 0.5",
            "E2 secondary: O/A 2", "X1 effective capacity phi",
            "X5 publication-blind loading series"}
    regime = {"cohort": cohort_tag, "holdout": "as named per contrast",
              "averaging_unit": "system (E1) / series (E2)"}
    if comp_path.exists():
        old, regime_old = read_table(comp_path)
        regime = {**regime, **{k: v for k, v in regime_old.items() if k in regime}}
        for r in old.itertuples():
            if r.name in mine:
                continue
            cc.record(str(r.name), str(r.family),
                      None if pd.isna(r.p_value) else float(r.p_value),
                      "" if pd.isna(r.note) else str(r.note))

    def p_of(s: dict[str, Any]) -> float | None:
        if s.get("n_series", 0) < 2:
            return None
        t = e2_series.loc[(e2_series["oa"] == s["oa"]) & (e2_series["status"] == "scored")]
        if s["subset"] == "loading_active":
            t = t.loc[t["loading_active"].astype(bool)]
        elif s["subset"] == "publication_blind":
            t = x5.loc[x5["status"] == "scored"] if len(x5) else t.iloc[0:0]
        d = t["mae_constant"].to_numpy(dtype=float) - t["mae_ideal"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if d.size < 2:
            return None
        rng = np.random.default_rng(args.seed)
        boots = d[rng.integers(0, d.size, size=(args.n_boot, d.size))].mean(axis=1)
        return min(1.0, 2.0 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0))))

    by = {(s["subset"], s["oa"]): s for s in summaries}
    cc.record("E2: C1 vs C0", "primary", p_of(by[("all", 1.0)]),
              "two-sided paired series bootstrap p of mean(E2(C0) - E2(C1)) = 0; R2 uses the CI")
    cc.record("E2 secondary: loading-active subset", "secondary", p_of(by[("loading_active", 1.0)]),
              "same contrast on the loading-active series")
    for oa_val, name in ((0.5, "E2 secondary: O/A 0.5"), (2.0, "E2 secondary: O/A 2")):
        s = by.get(("all", oa_val))
        cc.record(name, "secondary", p_of(s) if s else None,
                  "same contrast at O/A %s" % oa_val if s else "O/A not run")
    cc.record("X1 effective capacity phi", "exploratory", None,
              f"LOO-MAE(X1) < E2(C1) - {X1_MARGIN} on {n_active_wins} loading-active series; "
              f"supported = {x1_supported}")
    cc.record("X5 publication-blind loading series", "exploratory", p_of(x5_summary),
              "E2 on the publication-blind definition (11 series)")
    outputs.append(cc.write(comp_path, regime=regime))
    write_manifest(out_dir / "manifest.json", outputs,
                   [records_path, series_path, paths.PRE_REGISTRATION_MD,
                    paths.RESULTS_EVAL_DIR / "prereg_sha256.txt"],
                   args.seed, arguments=vars(args),
                   extra={"seal_check": seal_msg, "dry_run": bool(args.unsealed_dry_run),
                          "n_sources": n_sources})
    print(f"E2 (O/A 1): C0 {primary.get('e2_c0_macro', math.nan):.4f}  C1 "
          f"{primary.get('e2_c1_macro', math.nan):.4f}; wins {primary.get('wins_c1')}/"
          f"{primary.get('n_series')}; CI [{primary.get('bootstrap_lo', math.nan):.4f}, "
          f"{primary.get('bootstrap_hi', math.nan):.4f}]; supported = {supported}; "
          f"X1 supported = {x1_supported}")
    print(f"wrote {len(outputs)} files under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
