"""Script 12 of DESIGN.md section 11: assemble ``GEN18_REPORT.md`` from ``results/``.

Follows the outline of DESIGN.md section 14 (twelve sections).  Every table is copied with the
regime line its CSV carries (cohort, hold-out, averaging unit, parameter status); a table whose
script has not run is reported as "not available".  The pre-registration seal is checked with
``g18_seal_prereg.py --check``: when ``results/eval/prereg_sha256.txt`` differs from the sealed
file (mismatch) the script refuses; when the file is not yet sealed the report is written with a
visible UNSEALED banner and says so.  ``--results-dir`` points the assembly at another results
tree (``results/_dryrun`` for the unsealed dry run; the report then goes to
``<results-dir>/GEN18_REPORT_dryrun.md`` unless ``--out`` is given, so the real report file is
never built from dry-run tables).

Usage (from the repository root):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_report.py \\
        [--results-dir results] [--out GEN18_REPORT.md]

No wall-clock value is written (the bench timings are quoted as the only such numbers).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc.dmodel import ASSUMPTIONS  # noqa: E402
from gen18proc.metrics import Prices  # noqa: E402
from gen18proc.report import markdown_table, read_table, write_manifest  # noqa: E402
from gen18proc.systems import load_registry  # noqa: E402

MAX_ROWS = 40


def seal_state() -> tuple[str, str]:
    """``("sealed" | "unsealed" | "mismatch", message)`` from ``g18_seal_prereg.py --check``."""
    proc = subprocess.run([sys.executable, str(paths.SCRIPTS_DIR / "g18_seal_prereg.py"),
                           "--check"], capture_output=True, text=True, cwd=str(paths.REPO_ROOT),
                          check=False)
    msg = (proc.stdout or proc.stderr).strip()
    if proc.returncode == 0:
        return "sealed", msg
    if msg.startswith("unsealed"):
        return "unsealed", msg
    return "mismatch", msg


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def table(path: Path, cols: list[str] | None = None, max_rows: int = MAX_ROWS,
          query: str | None = None) -> str:
    """A CSV written by ``write_table`` as a markdown table with its regime caption."""
    if not path.exists():
        return f"*not available: {path.relative_to(paths.G18_ROOT).as_posix()} (script not run)*\n"
    df, regime = read_table(path)
    if query:
        try:
            df = df.query(query)
        except Exception:  # noqa: BLE001
            pass
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    note = ""
    if len(df) > max_rows:
        note = f"\n*(first {max_rows} of {len(df)} rows; full table in the CSV)*\n"
        df = df.head(max_rows)
    caption = "*regime: " + "; ".join(f"{k}={v}" for k, v in regime.items()) + "*" \
        if regime else None
    return markdown_table(df, caption=caption) + note


def text_of(path: Path, start: str | None = None, stop: str | None = None) -> str:
    if not path.exists():
        return f"*not available: {path.relative_to(paths.G18_ROOT).as_posix()}*\n"
    txt = path.read_text(encoding="utf-8")
    if start and start in txt:
        txt = txt[txt.index(start):]
    if stop and stop in txt:
        txt = txt[: txt.index(stop)]
    return txt.strip() + "\n"


def demote(md: str, levels: int = 1) -> str:
    """Push markdown headings down ``levels`` so an embedded file nests under its section."""
    out = []
    for line in md.splitlines():
        if line.startswith("#"):
            out.append("#" * levels + line)
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=str(paths.RESULTS_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    args = ap.parse_args()
    res = Path(args.results_dir)
    if not res.is_absolute():
        res = paths.G18_ROOT / res
    dry = res.resolve() != paths.RESULTS_DIR.resolve()
    out = Path(args.out) if args.out else (
        paths.GEN18_REPORT_MD if not dry else res / "GEN18_REPORT_dryrun.md")
    if not out.is_absolute():
        out = paths.G18_ROOT / out
    state, seal_msg = seal_state()
    if state == "mismatch":
        raise SystemExit(f"refused: {seal_msg} -- the pre-registration digest on disk differs "
                         "from the sealed file; restore the file or re-run g18_seal_prereg.py")
    sha_file = paths.RESULTS_EVAL_DIR / "prereg_sha256.txt"
    prereg_sha = sha_file.read_text(encoding="utf-8").split()[0] if sha_file.exists() else None
    systems_dir = Path(args.systems_dir)
    eval_dir, load_dir, case_dir = res / "eval", res / "loading", res / "case_prnd"
    dm_dir, bench_dir, rec_dir = res / "dmodels", paths.RESULTS_BENCH_DIR, res / "recipes"
    L: list[str] = ["# Gen18 report -- the process-design chain", ""]
    if state == "unsealed":
        L += ["> **UNSEALED: `PRE_REGISTRATION.md` has not been sealed (%s). Every number below "
              "that depends on a corpus fit is provisional and must not be quoted as a result "
              "of this generation until the pre-registration is sealed and the fit / eval / "
              "loading scripts are re-run.**" % seal_msg, ""]
    if dry:
        L += ["> **DRY RUN: assembled from `%s` (the two-system subset written under "
              "`--unsealed-dry-run`), not from `results/`.**" % res.relative_to(paths.G18_ROOT)
              .as_posix(), ""]
    L += ["Every number carries its regime (cohort, hold-out, averaging unit, parameter status) "
          "in the caption of its table. Assembled by `scripts/g18_report.py` from `results/`; "
          "nothing here was typed in by hand except the section text.", ""]

    # 0. the synthesis, if one has been written (integration 2026-09-13) ------------------------
    # SUMMARY.md is the one narrative part of this report a person writes; it is inserted here so
    # that re-running the assembler never drops it.  Absent -> the report starts at section 1.
    narrative = paths.G18_ROOT / "SUMMARY.md"
    if narrative.exists():
        L += [demote(narrative.read_text(encoding="utf-8").strip(), 0), "", "---", ""]

    # 1. what was built ------------------------------------------------------------------------
    L += ["## 1. What was built and what the database contains", "",
          "The chain *extraction-system composition -> D of each metal at the given loading -> "
          "countercurrent cascade (extraction + scrub + strip + organic recycle) -> purity, "
          "recovery, throughput, reagent consumption -> local regime optimisation* "
          "(DESIGN.md section 0.1). Modules: `gen18proc/systems.py` (database, validator), "
          "`ingest.py`, `literature.py`, `dmodel.py` (mass-action D models, composition, flags), "
          "`domain.py`, `equilibrium.py` (coupled stage), `cascade.py` (Newton + successive "
          "substitution, Kremser init, origin pass, ledgers), `metrics.py`, `evalproto.py` "
          "(pre-registered fit, LOPO, reliability, loading), `optimize.py` (design space, LHS + "
          "Pareto, gated GP-BO), `screen.py`, `report.py`.", ""]
    try:
        reg = load_registry(systems_dir)
        scaffolds = sorted({s for v in reg["scaffold_ids"].astype(str) for s in v.split(";") if s})
        L += [f"Database: {len(reg)} systems ({reg['origin'].value_counts().to_dict()}); "
              f"{int(reg['n_records'].sum())} distribution records; families "
              f"{reg['family'].value_counts().to_dict()}; two-ligand systems "
              f"{int(reg['ligands'].astype(str).str.contains(';').sum())}; scaffold ids "
              f"{scaffolds}.", ""]
    except Exception as exc:  # noqa: BLE001
        L += [f"*registry not readable: {exc}*", ""]
    excl = systems_dir / "exclusions.csv"
    if excl.exists():
        ex = pd.read_csv(excl)
        L += [f"Exclusions (gen13 quarantine): {len(ex)} rows "
              f"({ex['reason'].value_counts().to_dict()}).", ""]

    # 2. data audit --------------------------------------------------------------------------
    L += ["## 2. Data audit (from `DATA_AUDIT.md`, the frozen source of the cohort counts)", "",
          demote(text_of(paths.DATA_AUDIT_MD, "## 1.", "## 4."), 1),
          demote(text_of(paths.DATA_AUDIT_MD, "## 5.", "## 6."), 1)]

    # 3. E1 / E3 -----------------------------------------------------------------------------
    dec1 = load_json(eval_dir / "decision.json")
    L += ["## 3. Pre-registered D-model result (E1, R1, E3)", "",
          "Regimes: cohort E1, hold-out leave-one-publication-out within system, unit = system "
          "(points -> (publication, metal) -> system -> macro); in-sample and offset-calibrated "
          "variants are labelled in the tables.", "",
          "### E1 macro (M1, B0, B1, labelled variants)", "", table(eval_dir / "e1_macro.csv"),
          "", "### E1 per system (LOPO, primary arm)", "",
          table(eval_dir / "e1_per_system.csv", query="arm == 'primary' and holdout == 'lopo'"),
          "", "### R1 verdict (verbatim from `results/eval/DECISION.md`)", ""]
    if dec1:
        L += [f"**{dec1['verdict']}**", "",
              f"Conditions: {dec1['conditions']}; E1 macro {dec1['e1_macro']}; bootstrap of "
              f"E1(B1) - E1(M1): {dec1['bootstrap_B1_minus_M1']}; wins {dec1['wins_m1_over_b1']}"
              f"/{dec1['n_systems']} (needed {dec1['wins_needed']}). The chain's default D "
              "source for corpus systems is therefore "
              f"**{dec1['default_d_source_for_corpus_systems']}**"
              + (" (DRY RUN, not a result)" if dec1.get("dry_run") else "") + ".", ""]
    else:
        L += ["*not available: results/eval/decision.json (g18_eval_dmodels.py not run)*", ""]
    L += ["### E3 reliability (jackknife by publication; split-half by publication where >= 4 "
          "publications; `interpretable` = jackknife SE < 0.5 and sign agreement >= 18/20)", "",
          table(eval_dir / "e3_reliability.csv"), ""]
    e3 = eval_dir / "e3_reliability.csv"
    if e3.exists():
        df3, _ = read_table(e3)
        t = df3.loc[df3["system_id"] == "sys_5cb78e5000d40860"]
        if len(t):
            L += ["TODGA (`sys_5cb78e5000d40860`, band 20-30C) in-sample pooled n = %.4f "
                  "(jackknife SE %s) against the corpus-validated interval [2.36, 2.88]: **%s**. "
                  "The `validation`-marked test `tests/test_todga_slope.py` asserts the interval; "
                  "its outcome is recorded by the orchestrator's validation run." % (
                      float(t["n"].iloc[0]), t["jackknife_se_n"].iloc[0],
                      "inside" if bool(t.get("todga_n_in_range", pd.Series([False])).iloc[0])
                      else "OUTSIDE (reported defect, not fixed by changing the fit)"), ""]

    # 4. E2 ----------------------------------------------------------------------------------
    dec2 = load_json(load_dir / "decision.json")
    L += ["## 4. Loading (E2, R2)", "",
          "Regime: cohort = publication-aware loading series; hold-out = the tracer point "
          "anchors log K, the other points of the same series are scored; unit = series "
          "(weight 1); O/A = 1 assumed (`OA_ASSUMED`), K_H unknown.", "",
          "### Per series (O/A 1)", "",
          table(load_dir / "e2_series.csv",
                cols=["loading_series_id", "metal", "n_points", "log_d_tracer", "log_d_range",
                      "loading_active", "n_used", "n_source", "anchor_status", "mae_constant",
                      "mae_ideal", "c1_wins", "spearman_logD_vs_logmM", "flags"],
                query="oa == 1.0"), "",
          "### Summary (all series; loading-active subset; O/A sensitivity; publication-blind)",
          "", table(load_dir / "e2_summary.csv"), "",
          "### R2 verdict (verbatim from `results/loading/DECISION.md`)", ""]
    if dec2:
        L += [f"**{dec2['verdict']}**" + (" (DRY RUN, not a result)" if dec2.get("dry_run")
                                          else ""), "",
              demote(text_of(load_dir / "DECISION.md", "## Interpretation", None), 1), ""]
    else:
        L += ["*not available: results/loading/decision.json (g18_loading_check.py not run)*", ""]
    L += ["### Exploratory X1: effective capacity phi (leave-one-point-out)", "",
          table(load_dir / "phi_exploratory.csv"), ""]

    # 5. comparisons -------------------------------------------------------------------------
    L += ["## 5. Exploratory analyses, counted and BH-adjusted", "",
          table(eval_dir / "comparisons.csv"), ""]

    # 6. cascade verification ----------------------------------------------------------------
    bench = load_json(bench_dir / "timing.json")
    L += ["## 6. Cascade verification", "",
          "Invariant tests of DESIGN.md section 12.2 (`tests/test_cascade.py`, "
          "`test_equilibrium.py`, `test_metrics.py`): Kremser oracle in the constant-D limit "
          "(abs 1e-10), analytic Jacobian versus finite differences (rel 1e-6), Newton versus "
          "successive substitution on every stream quantity (rel 1e-8), ledgers recomputed from "
          "the stream table (rel 1e-8), multi-start stage agreement (rel 1e-9), origin-pass "
          "identity (rel 1e-10), failure as a status. Their outcome is the fast suite's "
          "(`pytest -m \"not slow and not validation\"`).", ""]
    if bench:
        ce = bench.get("results", {}).get("cation_exchange", {})
        L += ["Timing (`results/bench/timing.json`, **the only wall-clock numbers of this "
              "generation**; reference cascade %s, %d unknowns): stage solve %.2f ms, Newton "
              "cascade %.1f ms (%s iterations), successive substitution %.0f ms (%s sweeps), one "
              "LHS evaluation %.1f ms. The naive projection from that cost, %.0f min for "
              "64 x 2 x 1000 cascades, is **not** the case study's cost: the reference cascade is "
              "6/3/3, while the case searches 1-40 stages per section and measured **252 ms per "
              "cascade**, i.e. about 9 h for the specified run. It was run at 250 LHS per draw "
              "instead of 1000 (`addenda/INTEGRATION.md` section 3)."
              % (bench.get("reference_cascade"), ce.get("n_unknowns", 0),
                            ce.get("stage_solve_cold", {}).get("median_ms", float("nan")),
                            ce.get("cascade_newton", {}).get("median_ms", float("nan")),
                            ce.get("cascade_newton_iterations"),
                            ce.get("cascade_ss", {}).get("median_ms", float("nan")),
                            ce.get("cascade_ss_sweeps"),
                            ce.get("lhs_evaluation", {}).get("median_ms", float("nan")),
                            bench.get("projection", {}).get(
                                "case_study_64_draws_x_2_systems_x_1000_lhs_min", float("nan"))),
              ""]
    else:
        L += ["*results/bench/timing.json not available (g18_bench.py not run)*", ""]

    # 7. Pr/Nd case --------------------------------------------------------------------------
    case_summary = load_json(case_dir / "summary.json")
    L += ["## 7. Pr/Nd case: PC88A versus Cyanex 272", "",
          "Regime: feed `assumed` (U1), parameters `ASSUMED_PLACEHOLDER` with ranges (U4), "
          f"draws {case_summary.get('draws') if case_summary else '?'}, seed 18, stage bounds "
          f"1-{case_summary.get('max_stages') if case_summary else '?'}; consistency checks "
          "are not validation.", "",
          "### Parameter status", "",
          text_of(case_dir / "parameters.md") if (case_dir / "parameters.md").exists() else
          "*not available*", "",
          "### Interval tables per spec cell (min / median / max over draws)", "",
          text_of(case_dir / "intervals.md"), "",
          "### Sensitivities (one factor at a time at the median draw, reference regime)", "",
          table(case_dir / "sensitivity.csv",
                cols=["system_id", "factor", "value", "status", "purity_mol",
                      "recovery_from_feed", "consumption_index", "regime_status"]), "",
          "### Displacement scrub (all four origin quantities)", "",
          table(case_dir / "displacement_scrub.csv",
                cols=["system_id", "value", "purity_mol", "recovery_from_feed", "recovery_total",
                      "scrub_target_return", "net_product_mol_h", "regime_status"]), "",
          "### Labelled sanity limit (constant D per section, no acid balance)", "",
          table(case_dir / "sanity_limit.csv",
                cols=["system_id", "factor", "status", "purity_mol", "recovery_from_feed",
                      "consumption_index", "regime_status"]), "",
          demote(text_of(case_dir / "cyanex_vs_pc88a.md"), 2), "",
          demote(text_of(case_dir / "consistency_checks.md"), 2), ""]

    # 8. TODGA recipe and DGA exploration --------------------------------------------------
    L += ["## 8. TODGA Pr/Nd nitrate recipe and the DGA + aqueous ligand exploration "
          "(labelled exploratory)", ""]
    todga_rec = rec_dir / "todga_prnd_feed"
    if (todga_rec / "SUMMARY.md").exists():
        L += [demote(text_of(todga_rec / "SUMMARY.md"), 2), ""]
    else:
        L += ["*recipes for the nitrate feed not available (g18_recipes.py not run)*", ""]
    L += [demote(text_of(case_dir / "todga_exploration" / "todga_exploration.md"), 2), ""]

    # 9. consumption and cost ----------------------------------------------------------------
    L += ["## 9. Consumption and cost", "",
          "Consumption per kg of target oxide (acid, base, water, extractant make-up, "
          "complexant) is the primary economic number; the cost proxy uses the placeholder "
          "price table below (open item U5) and is NaN with `COST_INCOMPLETE` wherever a price "
          "or a consumption is missing.", ""]
    try:
        pt = pd.DataFrame(Prices.load().table())
        L += [markdown_table(pt, caption="*regime: config/prices.json; every value assumed "
                                         "with a range; no source*"), ""]
    except Exception as exc:  # noqa: BLE001
        L += [f"*price table not readable: {exc}*", ""]

    # 10. limits -----------------------------------------------------------------------------
    L += ["## 10. Limits, nulls, defects", "",
          "Model idealisations carried by every result (`SystemModel.assumptions`): "
          + "; ".join(ASSUMPTIONS) + ".", ""]
    cb = case_dir / "cell_best.csv"
    if cb.exists():
        dfc, _ = read_table(cb)
        reach = dfc.loc[dfc["reachable"].astype(bool)] if len(dfc) else dfc
        flags = reach["flags"].astype(str).str.split("|").explode() if len(reach) else pd.Series(
            dtype=str)
        counts = flags.value_counts()
        L += ["Flag prevalence over the reachable spec cells of the case (draw x cell rows): "
              + ", ".join(f"{k} {v}" for k, v in counts.items() if k) + ".", ""]
    L += ["Assumptions on corpus rows: `cond__metal_concentration_mM` is the initial aqueous "
          "concentration, O/A = 1 (`OA_ASSUMED`), nominal acid = equilibrium acidity "
          "(`EQUILIBRIUM_ACID_ASSUMED_NOMINAL`), nitrate = nominal HNO3 (DATA_AUDIT.md section "
          "8). HNO3 uptake by diglycolamides is unmodelled (`ACID_UPTAKE_UNMODELLED` above 1 M); "
          "the only sourced third-phase limit is the TODGA LOC of 0.008 M Nd (U6); every other "
          "phase field is null (`PHASE_BEHAVIOUR_UNKNOWN` above 0.3 loading).", ""]
    # defects found after the owners' own tests passed (integration 2026-09-13); the full account
    # is addenda/INTEGRATION.md, which is part of this generation's record.
    L += ["**Defects found in the integration pass** (`addenda/INTEGRATION.md`):", "",
          "1. *Fixed.* The deployed 1-NN D source applied the ideal depletion term to records "
          "that had themselves been measured under load -- 28.7 % of corpus records with a metal "
          "concentration sit above loading fraction 0.1 (TBDGA 81 %, median 0.30). Each record is "
          "now lifted to its own tracer limit first (`NearestConditionD(tracer_correction=True)`, "
          "the law R2 validated). R1 and R2 are unaffected (neither applies the term that way); "
          "`results/recipes/` was re-run, and 276 of 298 candidates of the TODGA system changed "
          "(median purity change 0.088) although the Pareto leaders barely moved.",
          "2. *Reported, not fixed.* Even corrected, the 1-NN is not smooth along the acid axis "
          "(D(Nd) at 0.1 M TODGA: 1.89, 37.9, 7.52, 1.13 at 0.5, 1, 3, 5 M HNO3): it reproduces "
          "genuine between-publication scatter, because a nearest-neighbour cannot average two "
          "publications that disagree. An optimiser searching over acidity can land on a single "
          "flattering record; the OOD flags are the only guard, and this is a property of the D "
          "source the R1 null selected.",
          "3. *Closed.* The hypothesis that M1 helps where its exponent is reliable is **not** "
          "supported (Spearman -0.117, p = 0.69; `results/eval/RELIABILITY_PROBE.md`), so a "
          "reliability gate is not a route around the null.", ""]
    sens = case_dir / "sensitivity.csv"
    if sens.exists():
        ds, _ = read_table(sens)
        rows = []
        for (sid, factor), g in ds.groupby(["system_id", "factor"]):
            if factor == "reference" or len(g) < 2:
                continue
            rows.append({"system_id": sid, "factor": factor,
                         "purity_range": float(g["purity_mol"].max() - g["purity_mol"].min()),
                         "recovery_range": float(g["recovery_from_feed"].max()
                                                 - g["recovery_from_feed"].min())})
        if rows:
            dsr = pd.DataFrame(rows).sort_values(["system_id", "purity_range"], ascending=[True,
                                                                                            False])
            L += ["What a single laboratory measurement would change most (the factor whose "
                  "sweep moves purity and recovery most at the reference regime; D-optimal "
                  "choice in the sense of gen15 section 7 -- measure the parameter with the "
                  "largest response first):", "", markdown_table(dsr), ""]

    # 11. gen15 -----------------------------------------------------------------------------
    L += ["## 11. gen15 pre-screen usage and its limits", "",
          "The gen15 direction model (`deploy_g15.joblib`, gitignored) is used only by "
          "`scripts/g18_screen.py` as a pre-screen prior on the sign of log D(A) - log D(B); "
          "validator V6 refuses any D sourced from it, and no cascade uses it.", ""]
    # summarise rather than dump: the per-molecule table is results/screen/priors.csv, and its
    # one informative fact is how few distinct magnitudes the model produces (integration).
    priors = res / "screen" / "priors.csv"
    if priors.exists():
        pf = pd.read_csv(priors, skiprows=1)
        signs = pf["sign"].value_counts(dropna=False).to_dict()
        mags = (pf["note"].str.extract(r"= ([+-][\d.]+) \(sd")[0].astype(float).abs()
                .round(6).dropna())
        distinct = sorted(mags.unique().tolist())
        L += [f"Over the {len(pf)} candidates of `cases/screen_candidates.txt` (every corpus "
              f"extractant with both Pr and Nd measured): "
              f"{signs.get(1, 0)} called Nd-selective, {signs.get(-1, 0)} Pr-selective. The "
              f"predicted magnitude takes only **{len(distinct)} distinct values** "
              f"({', '.join(f'{m:g}' for m in distinct)} log units), so the prior is a **sign "
              f"call with an essentially constant magnitude** -- the gen16 conclusion, reproduced "
              f"here. Per-molecule rows: `results/screen/priors.csv`.", ""]
    else:
        L += [table(priors), ""]

    # 12. reproduction -------------------------------------------------------------------------
    L += ["## 12. Reproduction", "",
          "From the repository root with `.venv/Scripts/python.exe`, seed 18, one process:", "",
          "```", "generations/gen18_process/scripts/g18_build_db.py",
          "generations/gen18_process/scripts/g18_audit.py",
          "generations/gen18_process/scripts/g18_seal_prereg.py",
          "generations/gen18_process/scripts/g18_fit_dmodels.py --all",
          "generations/gen18_process/scripts/g18_eval_dmodels.py",
          "generations/gen18_process/scripts/g18_loading_check.py",
          "generations/gen18_process/scripts/g18_bench.py",
          "generations/gen18_process/scripts/g18_case_prnd.py --feed cases/prnd_feed.json "
          "--spec cases/prnd_spec.json --allow-placeholders "
          "--todga-feed cases/todga_prnd_feed.json",
          "generations/gen18_process/scripts/g18_recipes.py --feed cases/todga_prnd_feed.json "
          "--spec cases/prnd_spec.json",
          "generations/gen18_process/scripts/g18_screen.py --smiles-file <txt> --pair Nd Pr",
          "generations/gen18_process/scripts/g18_report.py", "```", "",
          f"Pre-registration seal: {seal_msg}" + (f" (`prereg_sha256 = {prereg_sha}`)"
                                                  if prereg_sha else ""), ""]
    manifests = sorted(res.rglob("manifest.json"))
    if manifests:
        rows = []
        for m in manifests:
            obj = load_json(m) or {}
            rows.append({"manifest": m.relative_to(paths.G18_ROOT).as_posix(),
                         "git_head": obj.get("git_head"), "seed": obj.get("seed"),
                         "n_outputs": len(obj.get("outputs", {}))})
        L += ["Manifests (input / output SHA-256 in each file):", "",
              markdown_table(pd.DataFrame(rows)), ""]
    comp = eval_dir / "comparisons.csv"
    if comp.exists():
        dfcmp, _ = read_table(comp)
        L += ["Comparison count: " + ", ".join(
            f"{k} {v}" for k, v in dfcmp["family"].value_counts().items())
            + f" (total {len(dfcmp)}).", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8", newline="\n")
    manifest_name = "manifest_report.json" if not dry else "manifest_report_dryrun.json"
    write_manifest(out.parent / manifest_name,
                   [out], manifests + [paths.PRE_REGISTRATION_MD], args.seed,
                   arguments=vars(args), extra={"seal_state": state, "dry_run": dry})
    print(f"wrote {out} (seal: {state}; results from {res})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
