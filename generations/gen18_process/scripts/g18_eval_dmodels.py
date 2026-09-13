"""Script 5 of DESIGN.md section 11: the pre-registered evaluation of M1 (E1, E3, R1).

Per system of the E1 cohort: leave-one-publication-out scoring of M1 against B0 and B1
(``gen18proc.evalproto.lopo_evaluate``), the labelled in-sample rows, the exploratory arms X2
(metal-specific slopes), X4 (without ``TIED_D`` rows) and X6 (diluent-identity split inside
``sys_5cb78e5000d40860``); the E1 averaging (points -> (publication, metal) -> system -> macro),
the paired system bootstrap, the wins count and decision rule R1 of PRE_REGISTRATION.md
section 5; the E3 reliability table of section 6 (from ``results/dmodels/<id>.json`` when the
fit script has run, else recomputed).  Writes ``results/eval/lopo.csv``, ``e1_macro.csv``,
``e1_per_system.csv``, ``e3_reliability.csv``, ``comparisons.csv`` (``ComparisonCounter``),
``decision.json`` (``m1_adopted`` for ``build_system_model``), ``DECISION.md`` (the R1 verdict
verbatim) and ``manifest.json``.

Refuses to run unless ``g18_seal_prereg.py --check`` passes, EXCEPT under
``--unsealed-dry-run`` (two-system subset, outputs under ``results/_dryrun/eval/``; the R1 rule
is then applied to 2 systems and labelled DRY RUN -- never a result).

Usage (from the repository root, one process):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_eval_dmodels.py \\
        [--seed 18] [--n-boot 2000] [--band 20-30C] [--unsealed-dry-run]

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
from g18_fit_dmodels import DRYRUN_DIR, DRYRUN_SYSTEMS, e1_systems, require_sealed  # noqa: E402
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc.evalproto import (  # noqa: E402
    DEFAULT_BAND,
    MAE_COLUMNS,
    ComparisonCounter,
    fit_mass_action,
    in_sample_evaluate,
    lopo_evaluate,
    macro_over_systems,
    paired_system_bootstrap,
)
from gen18proc.report import write_manifest, write_table  # noqa: E402

TODGA_SYSTEM = "sys_5cb78e5000d40860"
TODGA_N_RANGE = (2.36, 2.88)
R1_MARGIN = 0.05

R1_RULE_VERBATIM = (
    "**R1.** M1 is *adopted as the default D source for corpus systems* if and only if all of:\n"
    "(i) macro E1(M1) < E1(B1) - 0.05 and E1(M1) < E1(B0) (margin 0.05 log D: below that the "
    "process consequence is inside the replicate floor 0.225 and the cheaper B1 is preferred);\n"
    "(ii) the paired system-level bootstrap (2000 resamples of systems with replacement, seed "
    "18, percentile 95 % interval of mean(E1(B1) - E1(M1))) excludes zero;\n"
    "(iii) M1 beats B1 in at least 9 of the 14 systems (ceil(0.6 x number of systems); the "
    "audit's count sets the number).\n"
    "If any of (i)-(iii) fails the result is a **null**: the process chain uses **B1 "
    "(`NearestConditionD`, with the ideal depletion correction and the prior `n0`) as its "
    "default D source for corpus systems**, M1 becomes an exploratory option with its LOPO "
    "table shown, and the report says so verbatim. A null is a result of this generation, not "
    "a failure of it.")
"""PRE_REGISTRATION.md section 5, decision rule R1, quoted verbatim."""

VERDICT_ADOPTED = "M1 is adopted as the default D source for corpus systems"
VERDICT_NULL = ("the result is a null: the process chain uses B1 (`NearestConditionD`, with the "
                "ideal depletion correction and the prior `n0`) as its default D source for "
                "corpus systems, M1 becomes an exploratory option with its LOPO table shown, and "
                "the report says so verbatim. A null is a result of this generation, not a "
                "failure of it.")


def bootstrap_p(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> float:
    """Two-sided bootstrap p-value of ``mean(a - b) = 0`` (same resampling as
    ``paired_system_bootstrap``: units with replacement, ``default_rng(seed)``)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    boots = d[idx].mean(axis=1)
    p_le = float(np.mean(boots <= 0.0))
    p_ge = float(np.mean(boots >= 0.0))
    return min(1.0, 2.0 * min(p_le, p_ge))


def evaluate_arm(records: pd.DataFrame, systems: list[tuple[str, str]], band: str, seed: int,
                 arm: str, **kw: Any) -> pd.DataFrame:
    frames = []
    for sid, ligand in systems:
        sub = records.loc[records["system_id"] == sid]
        if sub.empty:
            continue
        t = lopo_evaluate(sub, ligand=ligand, band=band, seed=seed, **kw)
        t.insert(0, "arm", arm)
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reliability_rows(records: pd.DataFrame, systems: list[tuple[str, str]], band: str,
                     seed: int, dmodels_dir: Path) -> pd.DataFrame:
    rows = []
    for sid, ligand in systems:
        path = dmodels_dir / f"{sid}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                f = json.load(fh)
            jk = f.get("jackknife_se") or {}
            sh = f.get("split_half") or {}
            row = {"system_id": sid, "ligand": f.get("ligand", ligand), "source": path.name,
                   "n": f.get("n"), "p_eff": f.get("p_eff"),
                   "jackknife_se_n": (jk.get("n") or [None, None])[1],
                   "jackknife_se_p_eff": (jk.get("p_eff") or [None, None])[1],
                   "split_half_defined": bool(sh),
                   "split_half_n_agree": sh.get("n_sign_agreement"),
                   "split_half_p_agree": sh.get("p_eff_sign_agreement"),
                   "split_half_repeats": sh.get("repeats"),
                   "split_half_intercept_r_median": sh.get("intercept_r_median"),
                   "slope_status_n": (f.get("slope_status") or {}).get("n"),
                   "slope_status_p_eff": (f.get("slope_status") or {}).get("p_eff"),
                   "interpretable_n": (f.get("interpretable") or {}).get("n"),
                   "interpretable_p_eff": (f.get("interpretable") or {}).get("p_eff"),
                   "n_points": f.get("n_points"), "n_publications": f.get("n_publications")}
        else:
            sub = records.loc[records["system_id"] == sid]
            fit = fit_mass_action(sub, ligand=ligand, band=band, seed=seed, reliability=True)
            jk = fit.jackknife_se or {}
            sh = fit.split_half or {}
            row = {"system_id": sid, "ligand": ligand, "source": "recomputed",
                   "n": fit.n, "p_eff": fit.p_eff,
                   "jackknife_se_n": (jk.get("n") or (None, None))[1],
                   "jackknife_se_p_eff": (jk.get("p_eff") or (None, None))[1],
                   "split_half_defined": bool(sh),
                   "split_half_n_agree": sh.get("n_sign_agreement"),
                   "split_half_p_agree": sh.get("p_eff_sign_agreement"),
                   "split_half_repeats": sh.get("repeats"),
                   "split_half_intercept_r_median": sh.get("intercept_r_median"),
                   "slope_status_n": fit.slope_status.get("n"),
                   "slope_status_p_eff": fit.slope_status.get("p_eff"),
                   "interpretable_n": fit.interpretable.get("n"),
                   "interpretable_p_eff": fit.interpretable.get("p_eff"),
                   "n_points": fit.n_points, "n_publications": fit.n_publications}
        if sid == TODGA_SYSTEM and row["n"] is not None:
            row["todga_n_range"] = f"[{TODGA_N_RANGE[0]}, {TODGA_N_RANGE[1]}]"
            row["todga_n_in_range"] = bool(TODGA_N_RANGE[0] <= float(row["n"]) <= TODGA_N_RANGE[1])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--band", default=DEFAULT_BAND)
    ap.add_argument("--unsealed-dry-run", action="store_true")
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    args = ap.parse_args()
    seal_msg = require_sealed(args.unsealed_dry_run)
    systems_dir = Path(args.systems_dir)
    e1 = e1_systems()
    if args.unsealed_dry_run:
        out_dir = DRYRUN_DIR / "eval"
        dmodels_dir = DRYRUN_DIR / "dmodels"
        e1 = e1.loc[e1["system_id"].isin(DRYRUN_SYSTEMS)]
    else:
        out_dir = paths.RESULTS_EVAL_DIR
        dmodels_dir = paths.RESULTS_DMODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = [(str(r.system_id), str(r.ligand)) for r in e1.itertuples()]
    n_systems = len(systems)
    need_wins = math.ceil(0.6 * n_systems)
    records_path = systems_dir / "corpus_records.csv"
    records = pd.read_csv(records_path, low_memory=False)
    cohort_tag = "E1 cohort (%d systems%s)" % (n_systems, ", DRY-RUN subset"
                                              if args.unsealed_dry_run else "")
    outputs: list[Path] = []

    # ---- primary LOPO, in-sample, exploratory arms -------------------------------------------
    lopo = evaluate_arm(records, systems, args.band, args.seed, "primary")
    ins_frames = []
    for sid, ligand in systems:
        sub = records.loc[records["system_id"] == sid]
        t = in_sample_evaluate(sub, ligand=ligand, band=args.band, seed=args.seed)
        t.insert(0, "arm", "primary")
        ins_frames.append(t)
    ins = pd.concat(ins_frames, ignore_index=True) if ins_frames else pd.DataFrame()
    x2 = evaluate_arm(records, systems, args.band, args.seed, "X2_metal_specific_slopes",
                      pooled_slopes=False)
    no_tied = records.loc[records["duplicate_flag"].astype("string").fillna("") != "TIED_D"]
    x4 = evaluate_arm(no_tied, systems, args.band, args.seed, "X4_without_tied_d")
    x6_frames = []
    if any(sid == TODGA_SYSTEM for sid, _ in systems):
        lig = dict(systems)[TODGA_SYSTEM]
        todga = records.loc[records["system_id"] == TODGA_SYSTEM]
        dil = todga["diluent_name"].astype("string").fillna("")
        for label, mask in (("X6_diluent_n_dodecane", dil == "n_dodecane"),
                            ("X6_diluent_other_aliphatic", dil != "n_dodecane")):
            sub = todga.loc[mask]
            if sub["publication_id"].nunique() >= 2:
                t = lopo_evaluate(sub, ligand=lig, band=args.band, seed=args.seed)
                t.insert(0, "arm", label)
                x6_frames.append(t)
    x6 = pd.concat(x6_frames, ignore_index=True) if x6_frames else pd.DataFrame()
    table = pd.concat([f for f in (lopo, ins, x2, x4, x6) if len(f)], ignore_index=True)
    outputs.append(write_table(table, out_dir / "lopo.csv", regime={
        "cohort": cohort_tag, "holdout": "leave-one-publication-out within system (rows "
                                        "holdout=in_sample are the labelled in-sample scores)",
        "averaging_unit": "(system, held-out publication, metal) row; errors on aggregated "
                          "points"}))

    # ---- E1 macro -----------------------------------------------------------------------------
    macro_rows, per_rows = [], []
    macros: dict[tuple[str, str], Any] = {}
    for arm_name, frame in (("primary", lopo), ("primary", ins), ("X2_metal_specific_slopes", x2),
                            ("X4_without_tied_d", x4)):
        if not len(frame):
            continue
        holdout = str(frame["holdout"].iloc[0])
        mr = macro_over_systems(frame)
        macros[(arm_name, holdout)] = mr
        for col in MAE_COLUMNS:
            macro_rows.append({"arm": arm_name, "holdout": holdout, "model": col[4:],
                               "e1_macro": mr.macro.get(col, np.nan),
                               "e1_point_weighted": mr.point_weighted.get(col, np.nan),
                               "n_systems": mr.n_systems})
        per = mr.per_system.reset_index()
        per.insert(0, "holdout", holdout)
        per.insert(0, "arm", arm_name)
        per_rows.append(per)
    for label in sorted(set(x6["arm"])) if len(x6) else []:
        mr = macro_over_systems(x6.loc[x6["arm"] == label])
        for col in MAE_COLUMNS:
            macro_rows.append({"arm": label, "holdout": "lopo", "model": col[4:],
                               "e1_macro": mr.macro.get(col, np.nan),
                               "e1_point_weighted": mr.point_weighted.get(col, np.nan),
                               "n_systems": mr.n_systems})
    e1_macro = pd.DataFrame(macro_rows)
    outputs.append(write_table(e1_macro, out_dir / "e1_macro.csv", regime={
        "cohort": cohort_tag, "holdout": "LOPO by publication (rows labelled in_sample are "
                                        "in-sample)",
        "averaging_unit": "system (macro); point-weighted column labelled"}))
    per_system = pd.concat(per_rows, ignore_index=True) if per_rows else pd.DataFrame()
    outputs.append(write_table(per_system, out_dir / "e1_per_system.csv", regime={
        "cohort": cohort_tag, "holdout": "LOPO by publication", "averaging_unit": "system"}))

    # ---- R1 ----------------------------------------------------------------------------------
    primary = macros.get(("primary", "lopo"))
    if primary is None or primary.n_systems == 0:
        raise SystemExit("no scored LOPO row: cannot apply R1")
    ps = primary.per_system
    m1, b0, b1 = (float(primary.macro["mae_M1"]), float(primary.macro["mae_B0"]),
                  float(primary.macro["mae_B1"]))
    mean_d, lo, hi = paired_system_bootstrap(ps["mae_B1"].to_numpy(), ps["mae_M1"].to_numpy(),
                                             n_boot=args.n_boot, seed=args.seed)
    wins = int((ps["mae_M1"] < ps["mae_B1"]).sum())
    cond = {"i_margin_vs_B1": bool(m1 < b1 - R1_MARGIN), "i_vs_B0": bool(m1 < b0),
            "ii_bootstrap_excludes_zero": bool(np.isfinite(lo) and lo > 0.0),
            "iii_wins": bool(wins >= need_wins)}
    adopted = all(cond.values())
    p_m1_b1 = bootstrap_p(ps["mae_B1"].to_numpy(), ps["mae_M1"].to_numpy(), args.n_boot,
                          args.seed)
    p_m1_b0 = bootstrap_p(ps["mae_B0"].to_numpy(), ps["mae_M1"].to_numpy(), args.n_boot,
                          args.seed)
    decision = {
        "schema": "gen18.decision.1", "rule": "R1", "m1_adopted": bool(adopted),
        "dry_run": bool(args.unsealed_dry_run), "n_systems": int(primary.n_systems),
        "wins_needed": int(need_wins), "wins_m1_over_b1": wins,
        "e1_macro": {"M1": m1, "B0": b0, "B1": b1,
                     "M1_offset": float(primary.macro.get("mae_M1_offset", np.nan)),
                     "B1_crossmetal": float(primary.macro.get("mae_B1_crossmetal", np.nan))},
        "bootstrap_B1_minus_M1": {"mean": mean_d, "lo": lo, "hi": hi, "n_boot": args.n_boot,
                                  "seed": args.seed},
        "conditions": cond, "margin": R1_MARGIN,
        "verdict": VERDICT_ADOPTED if adopted else VERDICT_NULL,
        "default_d_source_for_corpus_systems": "fitted (M1)" if adopted else "nearest (B1)",
    }
    dec_path = out_dir / "decision.json"
    dec_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                        newline="\n")
    outputs.append(dec_path)

    # ---- E3 reliability ----------------------------------------------------------------------
    e3 = reliability_rows(records, systems, args.band, args.seed, dmodels_dir)
    outputs.append(write_table(e3, out_dir / "e3_reliability.csv", regime={
        "cohort": cohort_tag, "holdout": "jackknife by publication; split-half by publication "
                                        "(20 repeats, seed 18) where >= 4 publications",
        "averaging_unit": "system", "status_of_parameters": "fitted_from_corpus"},
        regime_table=True))

    # ---- comparisons ---------------------------------------------------------------------------
    cc = ComparisonCounter()
    cc.record("E1: M1 vs B0", "primary", p_m1_b0,
              "two-sided paired system bootstrap p of mean(E1(B0) - E1(M1)) = 0")
    cc.record("E1: M1 vs B1", "primary", p_m1_b1,
              "two-sided paired system bootstrap p of mean(E1(B1) - E1(M1)) = 0; R1 uses the CI")
    cc.record("E2: C1 vs C0", "primary", None, "written by g18_loading_check.py")
    cc.record("E2 secondary: loading-active subset", "secondary", None,
              "written by g18_loading_check.py")
    cc.record("E2 secondary: O/A 0.5", "secondary", None, "written by g18_loading_check.py")
    cc.record("E2 secondary: O/A 2", "secondary", None, "written by g18_loading_check.py")
    todga_row = e3.loc[e3["system_id"] == TODGA_SYSTEM]
    todga_note = ("TODGA n = %.3f vs [2.36, 2.88]: %s" % (
        float(todga_row["n"].iloc[0]),
        "inside" if bool(todga_row["todga_n_in_range"].iloc[0]) else "OUTSIDE (reported defect)")
        if len(todga_row) and todga_row["n"].notna().any() else "TODGA not in this run")
    cc.record("E3 secondary: TODGA slope check", "secondary", None, todga_note)
    cc.record("X1 effective capacity phi", "exploratory", None,
              "written by g18_loading_check.py")
    if len(x2):
        mx2 = macro_over_systems(x2).per_system
        p = bootstrap_p(ps.loc[mx2.index, "mae_M1"].to_numpy(), mx2["mae_M1"].to_numpy(),
                        args.n_boot, args.seed) if len(mx2) else math.nan
        cc.record("X2 metal-specific slopes vs pooled M1", "exploratory", p,
                  "paired system bootstrap of mean(E1(M1 pooled) - E1(X2))")
    else:
        cc.record("X2 metal-specific slopes vs pooled M1", "exploratory", None, "not run")
    p_x3 = bootstrap_p(ps["mae_B1"].to_numpy(), ps["mae_B1_crossmetal"].to_numpy(), args.n_boot,
                       args.seed)
    cc.record("X3 cross-metal 1-NN vs B1", "exploratory", p_x3,
              "paired system bootstrap of mean(E1(B1) - E1(B1_crossmetal))")
    if len(x4):
        mx4 = macro_over_systems(x4).per_system
        p = bootstrap_p(ps.loc[mx4.index, "mae_M1"].to_numpy(), mx4["mae_M1"].to_numpy(),
                        args.n_boot, args.seed) if len(mx4) else math.nan
        cc.record("X4 tied-D sensitivity", "exploratory", p,
                  "paired system bootstrap of mean(E1(M1 all rows) - E1(M1 without TIED_D))")
    else:
        cc.record("X4 tied-D sensitivity", "exploratory", None, "not run")
    cc.record("X5 publication-blind loading series", "exploratory", None,
              "written by g18_loading_check.py")
    cc.record("X6 diluent-identity split (TODGA n-dodecane vs other)", "exploratory", None,
              "descriptive: E1 per subset in e1_macro.csv" if len(x6) else "not run")
    p_x7 = bootstrap_p(ps["mae_M1"].to_numpy(), ps["mae_M1_offset"].to_numpy(), args.n_boot,
                       args.seed)
    cc.record("X7 offset-calibrated one-measurement mode vs M1", "exploratory", p_x7,
              "paired system bootstrap of mean(E1(M1) - E1(M1_offset)); one held-out point "
              "sets the publication effect")
    cc.record("X8 Huber-loss M1", "exploratory", None,
              "declared in PRE_REGISTRATION section 8; not implemented in evalproto "
              "(no Huber option); counted, not run")
    outputs.append(cc.write(out_dir / "comparisons.csv", regime={
        "cohort": cohort_tag, "holdout": "as named per contrast",
        "averaging_unit": "system (E1) / series (E2)"}))

    # ---- DECISION.md -------------------------------------------------------------------------
    lines = ["# R1 decision (written by scripts/g18_eval_dmodels.py)", ""]
    if args.unsealed_dry_run:
        lines += ["**UNSEALED DRY RUN on %d systems (%s): not a result. The pre-registration "
                  "was not sealed when this was written; the rule is applied for testing "
                  "only.**" % (n_systems, ", ".join(s for s, _ in systems)), ""]
    lines += ["Regime: cohort = %s; hold-out = leave-one-publication-out within system; "
              "averaging unit = system (points -> (publication, metal) -> system -> macro); "
              "parameters fitted_from_corpus, band %s." % (cohort_tag, args.band), "",
              "## Rule (PRE_REGISTRATION.md section 5, verbatim)", "", R1_RULE_VERBATIM, "",
              "## Numbers", "",
              "| quantity | value |", "|---|---|",
              f"| E1(M1) macro | {m1:.4f} |", f"| E1(B0) macro | {b0:.4f} |",
              f"| E1(B1) macro | {b1:.4f} |",
              f"| E1(M1_offset) macro (one_measurement_mode, labelled) | "
              f"{decision['e1_macro']['M1_offset']:.4f} |",
              f"| condition (i): E1(M1) < E1(B1) - {R1_MARGIN} | {cond['i_margin_vs_B1']} |",
              f"| condition (i): E1(M1) < E1(B0) | {cond['i_vs_B0']} |",
              f"| condition (ii): 95 % bootstrap interval of mean(E1(B1) - E1(M1)) | "
              f"[{lo:.4f}, {hi:.4f}] (mean {mean_d:.4f}); excludes zero: "
              f"{cond['ii_bootstrap_excludes_zero']} |",
              f"| condition (iii): wins of M1 over B1 | {wins} of {n_systems} "
              f"(needed {need_wins}): {cond['iii_wins']} |", "",
              "## Verdict", "",
              ("**" + VERDICT_ADOPTED + ".**") if adopted else ("**" + VERDICT_NULL + "**"), "",
              "Consequence for the chain: the default D source for corpus systems is **%s** "
              "(`results/eval/decision.json`, key `m1_adopted = %s`, read by "
              "`build_system_model(params_source=\"auto\")`)." % (
                  decision["default_d_source_for_corpus_systems"], adopted), ""]
    dec_md = out_dir / "DECISION.md"
    dec_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    outputs.append(dec_md)
    write_manifest(out_dir / "manifest.json", outputs,
                   [records_path, paths.RESULTS_AUDIT_DIR / "e1_cohort.csv",
                    paths.PRE_REGISTRATION_MD, paths.RESULTS_EVAL_DIR / "prereg_sha256.txt"],
                   args.seed, arguments=vars(args),
                   extra={"seal_check": seal_msg, "dry_run": bool(args.unsealed_dry_run)})
    print(f"E1 macro: M1 {m1:.4f}  B0 {b0:.4f}  B1 {b1:.4f}; wins {wins}/{n_systems}; "
          f"bootstrap B1-M1 [{lo:.4f}, {hi:.4f}]; adopted = {adopted}")
    print(f"wrote {len(outputs)} files under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
