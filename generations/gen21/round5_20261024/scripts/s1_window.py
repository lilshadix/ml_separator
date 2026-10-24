#!/usr/bin/env python
"""Stage-1 item 1.4 (CLAUDE_CODE_PROMPT.md 6.4; 6.0, 6.10, 8, 9): M09, the B1 level window as a tail-risk reporting
axis.  Descriptive (window axis; never E-A, never next to 0.452); EXPLORATORY (seen) throughout.  Frozen rule list,
report rules and seeds: results/WINDOW/ITEM_SPEC.md (--write-spec).

Order (brief 1.2): Stage-0 seal -> item code seal -> --repro (TECHNICAL SMOKE) -> the arms.  Without --smoke the script
refuses to run unless r5.sealcheck.require_seal() verifies.  --smoke runs the full construction path on real data for
a few folds (construction / leakage counters and identities on real data), then replaces y and every prediction column
by seeded noise before any table runs, so no real-label metric is computed; it prints shapes, counts, NaN patterns,
timings and the counters only, and writes nothing.  A sealed run writes START / END lines in RUN_RECORD.md and ends
with the post-run seal check; --no-lopo is allowed in --smoke only.

Writes (after the seal) under results/WINDOW/: repro_<design>.json; rows_<design>.parquet (row-level levels, s, reads,
fold thresholds, nested choices and every rule's membership), ops_<design>.csv (per-fold operating points and choices,
training side), checks_<design>.json, one CSV per table (<table>_<design>.csv, every row labelled, a seal column),
summary_<design>.json, and ITEM_TABLE.md (the section-9.3 per-item table, with a seal line) when both designs ran; under
results/leakage/: poison_WINDOW_ops_<design>.json (fit poisoning of the operating points) and WINDOW_<design>.json (the
leakage / support counters of checks_<design>.json, PROTOCOL 12).

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_window.py --smoke --design BPG
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_window.py --repro --design both
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_window.py --design both
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402

import pandas as pd  # noqa: E402

from r5 import common as CM  # noqa: E402
from r5 import window as WIN  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = WIN.OUT
SMOKE_REPS = 50
METRICS_NOTE = ("Note: relMSE_in is B1's inside relative MSE, 1 - MSE_in / MSE_all of B1's own residual (inside vs all "
                "rows; per seed, then the seed mean); it is not a relative SSE vs B1 and not E-A currency (M35). "
                "excess_relMSE_vs_CELL is relMSE_in minus its closed-form expectation under cell-matched random "
                "abstention (summary_<design>.json carries the 2,000-draw Monte Carlo estimate as "
                "excess_relMSE_vs_CELL_MC).  ctrl_* summarise the CELL control's draws; where ctrl_reproduces_rule is "
                "True (a rule constant within (seed, cell), such as READMIN-2, WHOLECELL2 or NOC5, or coverage 1) the "
                f"draws equal the rule and P_ctrl_relMSE_ge_rule reads '{WIN.NA_REPRO}'.  The row flag compares "
                "NEST_FAM's relMSE_in with ctrl_relMSE_q975 (Report rules below).")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _shape_report(name: str, df) -> str:
    """Shapes and NaN pattern only (smoke): no value is printed."""
    if not isinstance(df, pd.DataFrame):
        return f"    {name}: {type(df).__name__}"
    num = df.select_dtypes("number")
    nan_cols = sorted(num.columns[num.isna().any()]) if len(num.columns) else []
    return (f"    {name}: {df.shape[0]} rows x {df.shape[1]} cols; numeric cols with NaN {len(nan_cols)}/{len(num.columns)}"
            + (f" {nan_cols}" if nan_cols else ""))


def _write_csv(df: pd.DataFrame, p: Path, written: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    written.append(p)


def _write_json(obj, p: Path, written: list) -> None:
    paths.write_text_lf(p, json.dumps(obj, indent=1, default=str) + "\n")
    written.append(p)


def _counters(checks: dict) -> str:
    """Leakage / support counters (construction checks, no score)."""
    keys = [k for k in WIN.CHECK_KEYS if k in checks] + ["B1_identity_max_abs", "HIST_D7_identity_max_abs", "nan_levels"]
    return ", ".join(f"{k} {checks[k]}" for k in keys if k in checks)


def item_table(S: dict, T: dict, checks: dict, compute_s: float) -> str:
    """The section-9.3 per-item table over both designs (written only after both ran)."""
    cur = pd.concat([T[d]["curve"] for d in S], ignore_index=True)
    ctl = pd.concat([T[d]["controls"] for d in S], ignore_index=True)
    show = list(WIN.MAIN_RULES) + ["NOC5", "READMIN-2", "WHOLECELL2", "LO-4"]
    met = cur[cur.rule.isin(show) & (cur.variant == "all")][
        ["design", "rule", "coverage", "R2_in", "R2_out", "MSE_in", "MSE_out", "relMSE_in", "excess_relMSE_vs_CELL", "label"]]
    cc = ctl[ctl.rule.isin(show) & (ctl.variant == "all") & (ctl.control == "CELL")][
        ["design", "rule", "ctrl_R2_mean", "ctrl_R2_q975", "ctrl_relMSE_q975", "P_ctrl_relMSE_ge_rule",
         "ctrl_reproduces_rule"]]
    met = met.merge(cc, on=["design", "rule"], how="left")
    met = met[[c for c in met.columns if c != "label"] + ["label"]]
    ut = pd.concat([T[d]["unit_tests"] for d in S], ignore_index=True)
    lp = pd.concat([T[d]["lopo"] for d in S], ignore_index=True)
    if len(lp):
        lp = lp[lp.rule.isin(["OP75", "NEST_FAM"])][["design", "rule", "variant", "stat", "point", "lo", "hi", "min",
                                                      "min_by", "max", "max_by", "label"]]
    c5 = pd.concat([T[d]["c5"] for d in S], ignore_index=True)
    c5 = c5[c5.rule.isin(["OP75", "H2", "NEST_FAM"])][["design", "rule", "variant", "rows_per_seed", "cells", "coverage",
                                                         "R2_in", "excess_relMSE_cell", "tail_ratio_0.5", "label"]]
    dup = pd.concat([T[d]["duplicates"] for d in S], ignore_index=True)
    if len(dup):
        dup = dup[dup.rule.isin(["OP75", "H2", "NEST_FAM"])][["design", "rule", "variant", "rows_per_seed", "cells",
                                                                "coverage", "R2_in", "excess_relMSE_cell",
                                                                "tail_ratio_0.5", "label"]]
    rr = pd.concat([T[d]["report_rules"] for d in S], ignore_index=True)
    leak = "; ".join(f"{d}: {_counters(c)}" for d, c in checks.items())
    prim = {d: S[d]["headline"] for d in S}
    pr = {}
    for d in S:
        r = next(h for h in prim[d] if h["rule"] == WIN.PRIMARY and h["variant"] == "all")
        pr[d] = r
    rule_out = {d: {x["rule_name"]: x["outcome"] for x in S[d]["report_rules"] if x["primary"]} for d in S}
    per_seed = {}
    for d in S:
        c = T[d]["curve"]
        r = c[(c.rule == WIN.PRIMARY) & (c.variant == "all")].iloc[0]
        per_seed[d] = (f"{WIN.PRIMARY} inside R2 per seed {r.R2_in_per_seed}; coverage per seed {r.coverage_per_seed} "
                       f"({r.label})")
    b = pr.get("BPG", next(iter(pr.values())))
    rf = {}                                                 # the row flag's own scale and NEST_FAM's R2 beside it
    for d in S:
        c, o = T[d]["controls"], T[d]["optimism"]
        rf[d] = (c[(c.rule == "NEST_FAM") & (c.variant == "all") & (c.control == "CELL")].iloc[0],
                 o[(o.rule == "NEST_FAM") & (o.variant == "all")].iloc[0])
    sse = "; ".join(f"{d} {c.relMSE_in:.4f} vs CELL q97.5 {c.ctrl_relMSE_q975:.4f}" for d, (c, _) in rf.items())
    r2 = ", ".join(f"{d} {c.R2_in:.4f} vs {c.ctrl_R2_mean:.4f} ({'below' if c.R2_in < c.ctrl_R2_mean else 'at or above'})"
                   for d, (c, _) in rf.items())
    ncl = " and ".join(f"{int(o.fold_fits_choosing_cell_level_rule)} of {int(o.fold_fits)} ({d}; membership constant "
                       f"within every cell in {int(o.fold_fits_cell_constant_membership)})" for d, (_, o) in rf.items())
    changed = (f"On E-A's rows (BPG), the training-fold OP75 window keeps {b['coverage']:.3f} of rows with B1 inside R2 "
               f"{b['R2_in']:.4f} against the cell-matched control mean {b['CELL_R2_mean']:.4f} (q97.5 {b['CELL_R2_q975']:.4f}) "
               f"and a |res| > 0.5 outside / inside ratio of {b['tail05_ratio']:.2f} "
               f"[{b['tail05_boot'][0]:.2f}, {b['tail05_boot'][1]:.2f}]; report rules (CONSISTENCY CHECKs): "
               + "; ".join(f"{d} {o}" for d, o in rule_out.items())
               + f". The row flag is evaluated at NEST_FAM on inside relative SSE ({sse}), not on inside R2: NEST_FAM's "
                 f"inside R2 vs its cell-matched control mean is {r2}; NEST_FAM chose a cell-level rule in {ncl} "
                 "fold-fits, and wherever its membership is constant within every cell the CELL control reproduces it "
                 "(EXPLORATORY seen; a window-axis reporting value, never an E-A value).")
    item = {"item": "1.4 M09 level window as a tail-risk reporting axis (window axis; descriptive)",
            "population": CM.population(CM.load_rows("BPG")),
            "design": "BPG (primary) and BP (labelled sensitivity); identical test folds",
            "read_budget": "4 reads (E-A rows, code k = 3); the window restricts which rows are reported, never an E-A value",
            "arms": "rules " + ", ".join(WIN.RULES) + "; controls CELL (primary), PUBPT, UNSTRAT at matched counts; "
                    "residual of B1 (B0 and HIST_D7 inside R2 beside it)",
            "metrics": met, "per_seed": per_seed,
            "units": ut, "lopo_top3": lp if len(lp) else "not run (--no-lopo)", "c5": c5,
            "duplicates": dup if len(dup) else "n/a (duplicate sets unavailable)", "leakage": leak,
            "compute": f"{compute_s:.0f} s ({', '.join(f'{d} {S[d]['seconds']:.0f} s' for d in S)})",
            "label": WIN.SEEN + "; reproduction run TECHNICAL SMOKE; report rules CONSISTENCY CHECK; BP, C5-BTBP out, "
                                "duplicates, winsorised and leave-top-k SENSITIVITY; unit tests DIAGNOSTIC",
            "changed": changed,
            "next_step": "Registry item 1.5 (M14 direction calls, E-REC own-prior row, E-RANK table; results/EDEC/)."}
    txt = CM.per_item_table(item, title="Item 1.4 (window axis, M09): per-item table")
    name = dict(CM.ITEM_FIELDS)["metrics"]
    txt = txt.replace(f"## {name}\n\n", f"## {name}\n\n{METRICS_NOTE}\n\n", 1)
    seal = ", ".join(sorted({str(S[d]["seal"]) for d in S}))
    txt = txt.replace("\n\n| field | value |", f"\n\n**Seal: {seal}** (every number below; each table CSV carries it in its "
                                               "seal column)\n\n| field | value |", 1)
    return txt + "\n## Report rules\n\n" + CM.md_table(rr) + "\n"


ITEM = "WINDOW"
SMOKE_Y_COLS = ("y", "B1", "B0", "HIST_D7", "B1_lev", "HIST_D7_lev")


def smoke_surrogate(R: pd.DataFrame, design: str) -> pd.DataFrame:
    """PRE-SEAL ENGINEERING CHECK (s0-mechanics-22): y and every prediction column replaced by seeded noise (NaN pattern kept)
    before any table runs, so no real-label R2, MSE or contrast is computed; the label-free rule memberships (in_*),
    the read levels and s are kept (they come from the reads only)."""
    import numpy as np
    S = R.copy()
    rng = np.random.default_rng(paths.seed_for(f"WINDOW|SMOKE_SURROGATE|{design}"))
    ys = rng.normal(0.0, 0.3, len(S))
    S["y"] = np.where(np.isfinite(S.y.to_numpy(float)), ys, np.nan)
    for c in SMOKE_Y_COLS[1:]:
        if c in S.columns:
            v = S[c].to_numpy(float)
            S[c] = np.where(np.isfinite(v), ys + rng.normal(0.0, 0.2, len(S)), np.nan)
    return S


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", choices=("BPG", "BP", "both"), default="both")
    ap.add_argument("--smoke", action="store_true", help="technical smoke: metrics suppressed, nothing written")
    ap.add_argument("--smoke-folds", type=int, default=2, help="folds per design in --smoke (default 2)")
    ap.add_argument("--smoke-reps", type=int, default=SMOKE_REPS, help="control / bootstrap draws in --smoke (timing)")
    ap.add_argument("--repro", action="store_true", help="only the reproduction targets of brief 6.0 (TECHNICAL SMOKE)")
    ap.add_argument("--no-lopo", action="store_true", help="skip leave-one-publication-out (--smoke only)")
    ap.add_argument("--write-spec", action="store_true", help="write results/WINDOW/ITEM_SPEC.md and exit (no data read)")
    ap.add_argument("--amend", action="store_true", help="--write-spec after the seal (an amendment and a reseal follow)")
    args = ap.parse_args(argv)
    from r5 import sealcheck
    if args.write_spec:
        sealcheck.refuse_if_sealed("--write-spec (results/WINDOW/ITEM_SPEC.md is a sealed input)", args.amend)
        paths.write_text_lf(OUT / "ITEM_SPEC.md", WIN.item_spec_text())
        print(f"wrote {OUT / 'ITEM_SPEC.md'}")
        return 0
    sealcheck.forbid_reduced_coverage(bool(args.smoke), no_lopo=bool(args.no_lopo))
    if args.smoke:
        return body(args, "SMOKE")
    tag = sealcheck.require_seal()
    return sealcheck.run_sealed(ITEM, "repro" if args.repro else "run", tag, lambda: body(args, tag))


def body(args, tag: str) -> int:
    from r5 import sealcheck
    t0 = time.time()
    written: list[Path] = []
    smoke = bool(args.smoke)
    mode = "smoke" if smoke else ("repro" if args.repro else "run")
    designs = ("BPG", "BP") if args.design == "both" else (args.design,)
    print(f"item 1.4 (window axis, M09) {'PRE-SEAL ENGINEERING CHECK (--smoke; metrics suppressed; nothing written)' if smoke else 'seal ' + tag}; "
          f"designs {designs}; repro only {args.repro}", flush=True)
    bench = D.load_bench()
    print(f"  bench {len(bench.frame)} cells ({time.time() - t0:.0f}s)", flush=True)
    S, T, CK = {}, {}, {}
    for design in designs:
        td = time.time()
        res = WIN.assemble(bench, design, folds=args.smoke_folds if smoke else None, ops=not args.repro,
                           intervention=True, poison=not args.repro, log=lambda s: print(s, flush=True))
        R, O, checks = res["rows"], res["ops"], res["checks"]
        checks["seal"] = tag
        CK[design] = checks
        print(f"  {design}: rows {len(R)} x {R.shape[1]} cols; cells {R.cell_id.nunique()}; seeds {R.split_seed.nunique()}; "
              f"folds {len(O) if len(O) else checks['folds']}; NaN in s {int(R.s.isna().sum())}; ({checks['seconds']:.0f}s)",
              flush=True)
        print(f"  {design} checks: {_counters(checks)}", flush=True)
        bad = WIN.checks_failed(checks)
        if bad:
            if smoke:
                raise SystemExit(f"{design}: leakage / support check failed: {bad}")
            sealcheck.stop(ITEM, mode, tag, f"{design}: leakage / support check failed: {bad}")
        rep = WIN.repro_checks(R, design, scores=not smoke)
        if smoke:
            print(f"  {design} reproduction identity (partial folds; PRE-SEAL ENGINEERING CHECK): rows {rep['rows']}; B1 identity ok "
                  f"{rep['identity_ok']} (max |diff| {rep['B1_identity_max_abs']:.3e}); HIST_D7 max |diff| "
                  f"{rep['HIST_D7_identity_max_abs']:.3e}", flush=True)
            if not args.repro:
                print(_shape_report("rows", R), flush=True)
                print(_shape_report("ops", O), flush=True)
                Rs = smoke_surrogate(R, design)              # no table ever sees a real label in the smoke
                Td = WIN.analyse(Rs, O, design, reps_control=args.smoke_reps, reps_boot=args.smoke_reps,
                                 lopo=not args.no_lopo, log=lambda s: print(s, flush=True))
                for k, v in Td.items():
                    print(_shape_report(k, v), flush=True)
                T[design] = Td
                S[design] = WIN.summary(Td, design, checks, rep, round(time.time() - td, 1), tag)   # in memory only
                print(f"    summary_{design} (surrogate): {len(json.dumps(S[design], default=str))} characters (not "
                      "written)", flush=True)
            print(f"  {design} smoke done ({time.time() - td:.0f}s)", flush=True)
            continue
        rep["seal"] = tag
        _write_json(rep, OUT / f"repro_{design}.json", written)
        print(f"  {design} reproduction (TECHNICAL SMOKE): identity {rep['identity_ok']}; target_ok {rep.get('target_ok')}; "
              f"ok {rep.get('ok')}", flush=True)
        if not rep.get("ok"):
            for p in written:
                print(f"wrote {p}")
            sealcheck.stop(ITEM, mode, tag, f"{design}: reproduction mismatch -- the item stops (brief 6.0); see "
                                            f"results/WINDOW/repro_{design}.json")
        if args.repro:
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        R.assign(label=WIN.row_labels(R, design), seal=tag).to_parquet(OUT / f"rows_{design}.parquet", index=False)
        written.append(OUT / f"rows_{design}.parquet")
        _write_csv(O.assign(label="fitted on the fold's training rows only (label-free thresholds; training-side "
                                  "selection scores, EXPLORATORY context)", seal=tag), OUT / f"ops_{design}.csv", written)
        _write_json(checks, OUT / f"checks_{design}.json", written)
        _write_json({"label": "LEAKAGE CHECK (PROTOCOL 12 chain items 1, 2, 4 and 5: isolation, fit poisoning, "
                              f"prediction intervention, support; the counters of results/WINDOW/checks_{design}.json)",
                     "item": ITEM, "design": design, "created_utc": _now(), "seal": tag,
                     "failed": WIN.checks_failed(checks), "counters": checks,
                     "fit_poisoning": f"results/leakage/poison_WINDOW_ops_{design}.json",
                     "cache_poisoning": "n/a: no new cache (PROTOCOL 12)"},
                    WIN.LEAK / f"WINDOW_{design}.json", written)
        _write_json({"label": "leakage check (fit poisoning of M09's operating points and nested choices; brief 8 item 2)",
                     "design": design, "created_utc": _now(), "seal": tag, "folds": int(len(O)),
                     "poison_checked": checks["poison_checked"], "poison_failed": checks["poison_failed"],
                     "all_identical": bool(checks["poison_failed"] == 0 and checks["poison_checked"] == len(O)),
                     "positive_control_runs": checks["poison_control_runs"],
                     "positive_control_caught": checks["poison_control_caught"],
                     "per_fold": O[["split_seed", "fold", "poison_identical"]].to_dict("records")},
                    WIN.LEAK / f"poison_WINDOW_ops_{design}.json", written)
        Td = WIN.analyse(R, O, design, log=lambda s: print(s, flush=True))
        for k, v in Td.items():
            _write_csv(v.assign(seal=tag) if len(v) else v, OUT / f"{k}_{design}.csv", written)
        T[design] = Td
        S[design] = WIN.summary(Td, design, checks, rep, round(time.time() - td, 1), tag)
        _write_json(S[design], OUT / f"summary_{design}.json", written)
    if set(S) == {"BPG", "BP"}:
        txt = item_table(S, T, CK, time.time() - t0)
        if smoke:
            print(f"  per-item table built in memory on the surrogate: {len(txt.splitlines())} lines (not written)",
                  flush=True)
        elif not args.repro:
            paths.write_text_lf(OUT / "ITEM_TABLE.md", txt)
            written.append(OUT / "ITEM_TABLE.md")
    for p in written:
        print(f"wrote {p}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
