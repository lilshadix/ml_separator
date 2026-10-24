#!/usr/bin/env python
"""Stage-1 item 1.7 (CLAUDE_CODE_PROMPT.md 6.7; 6.0, 6.10, 8, 9): E-W, the M13 wider-pair table.  EXPLORATORY (seen)
throughout; E-W is never pooled with the strict pairs and never placed next to E-A's 0.452.  Frozen arm list, report
rules and seeds: results/EW/ITEM_SPEC.md (--write-spec).

Order (brief 1.2): Stage-0 seal -> item code seal -> --repro (TECHNICAL SMOKE) -> the item.  Without --smoke the script
refuses to run unless r5.sealcheck.require_seal() verifies, records START / END lines, writes the seal tag and code
digests next to every output, and ends with the post-run seal check; --from-rows refuses rows or checks of another
seal or code.  --smoke runs the construction path on real data for a few folds (identity and leakage counters on real
data), replaces y and every prediction column by seeded noise before any table runs, prints shapes, counts, NaN
patterns and timings only, and writes nothing; --no-intervention / --no-lopo are smoke-only.

Writes (after the seal) under results/EW/: repro_<design>.json (TECHNICAL SMOKE); rows_<design>.parquet (row-level
STRICT + WIDE predictions of B1 / B0 / LINE / HIST_D7), checks_<design>.json (leakage / support counters), one CSV per
table (<table>_<design>.csv: cuts, classes, strict_reference, substitution_trap, signs, heavy, ndsm_contrast,
ndsm_clusters, units, unit_tests, duplicates, seen; every number labelled), summary_<design>.json, and ITEM_TABLE.md
(the section-9.3 per-item table) when both designs ran; and results/leakage/EW_<design>.json (PROTOCOL 12: the
leakage / support counters of checks_<design>.json with the seal tag and code digests).

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ew.py --smoke --design both
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ew.py --repro --design both
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ew.py --design both
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402

import pandas as pd  # noqa: E402

from r5 import common as CM  # noqa: E402
from r5 import ew as EW  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = EW.OUT
LEAK = paths.RES / "leakage"
ITEM = "EW"
CODE = ("r5/ew.py", "scripts/s1_ew.py", "r5/common.py", "r5/paths.py")


def code_digests() -> dict:
    return {c: paths.sha256_text_lf(paths.R5 / c) for c in CODE}


def smoke_surrogate(R: pd.DataFrame, design: str) -> pd.DataFrame:
    """PRE-SEAL ENGINEERING CHECK (s0-mechanics-22): y and every prediction column replaced by seeded noise (NaN pattern kept)
    before any table runs, so no real-label metric is computed in memory."""
    import numpy as np
    rng = np.random.default_rng(paths.seed_for(f"EW|SMOKE_SURROGATE|{design}"))
    S = R.copy()
    ys = rng.normal(0.0, 0.5, len(S))
    S["y"] = ys
    for a in EW.ARMS:
        v = S[f"p_{a}"].to_numpy(float)
        S[f"p_{a}"] = np.where(np.isfinite(v), ys + rng.normal(0.0, 0.3, len(S)), np.nan)
    return S


def _shape_report(name: str, df) -> str:
    """Shapes and NaN pattern only (smoke): no value is printed."""
    if not isinstance(df, pd.DataFrame):
        return f"    {name}: {type(df).__name__}"
    num = df.select_dtypes("number")
    nan_cols = sorted(num.columns[num.isna().any()]) if len(num.columns) else []
    return (f"    {name}: {df.shape[0]} rows x {df.shape[1]} cols; numeric cols with NaN {len(nan_cols)}/{len(num.columns)}"
            + (f" {nan_cols[:8]}{' ...' if len(nan_cols) > 8 else ''}" if nan_cols else ""))


def _write_csv(df: pd.DataFrame, p: Path, written: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    written.append(p)


def _write_json(obj, p: Path, written: list) -> None:
    paths.write_text_lf(p, json.dumps(obj, indent=1, default=str) + "\n")
    written.append(p)


def leakage_record(checks: dict, design: str, tag: str) -> dict:
    """PROTOCOL 12's per-item leakage report (results/leakage/EW_<design>.json): the brief-8 counters of
    results/EW/checks_<design>.json with the seal tag and code digests (no score)."""
    c = {k: v for k, v in checks.items() if k not in ("seal", "code_lf_sha256")}
    bad = EW.bad_checks(c)
    return {"label": "LEAKAGE CHECK (section 8 items 1, 4, 5)", "item": ITEM, "design": design, "seal": tag,
            "passed": not bad, "failed": bad,
            "isolation": f"g21.evaluate.assert_test_isolation passed on {c.get('isolation_ok')}/{c.get('folds')} folds",
            "intervention": f"every unread label replaced by N(0, 5): {c.get('intervention_failed')} of "
                            f"{c.get('intervention_checked')} checked cell instances ({c.get('cell_instances')} in all) "
                            f"changed a level curve or the read set; block-Y peek positive control caught "
                            f"{c.get('peek_control_caught')}/{c.get('peek_control_runs')} (one per fold)",
            "support": {k: v for k, v in c.items() if k.endswith("_bad") or k == "wide_both_read_excluded"},
            "fit_poisoning": "not applicable: no arm is fitted or selected (stored round-4 fold betas)",
            "cache_poisoning": "not applicable: no new cache (fold records read with g21.cache.load)",
            "counters": c, "details": f"results/EW/checks_{design}.json", "code_lf_sha256": code_digests()}


def _counts(R: pd.DataFrame) -> str:
    """Row counts per family / cut class (counts only)."""
    ns = max(R.split_seed.nunique(), 1)
    fam = R.groupby("family").size().to_dict()
    cc = (R[R.family == "WIDE"].groupby("cut_class").size() / ns).round(1).to_dict()
    nd = int((R.pair == "Nd/Sm").sum())
    return (f"families {fam}; WIDE rows per seed by cut class {cc}; Nd/Sm rows {nd}; pair types "
            f"{R[R.family == 'WIDE'].pair.nunique()}; cells {R.cell_id.nunique()}; NaN predictions "
            f"{int(R[[f'p_{a}' for a in EW.ARMS]].isna().sum().sum())}")


def _dkey(d: str) -> str:
    return d + (" (SENSITIVITY (design BP))" if d == "BP" else "")


def item_table(S: dict, T: dict, pop: dict, compute_s: float) -> str:
    """The section-9.3 per-item table over both designs (written only after both ran); ``pop``: design ->
    EW.population_table of that design's rows (WIDE lines first, then the STRICT E-A lines)."""
    cl = pd.concat([T[d]["classes"] for d in S], ignore_index=True)
    met = cl[(cl.variant == "all") & cl.frame.isin(EW.MAIN_FRAMES + ("dZ2 without Nd/Sm", "heavy (dZ2-4; NON-CLAIM)"))][
        ["design", "frame", "rows_per_seed", "R2_B1", "R2_B0", "R2_LINE", "skill_vs_LINE", "skill_vs_B0", "MSE_B1",
         "MSE_LINE", "MSE_B0", "var_y", "MSE_mult_B1", "var_mult", "MAE_rel_median_B1", "sign_agree_B1", "macro_R2_B1",
         "label"]]
    c5 = cl[cl.frame.isin(EW.MAIN_FRAMES)].pivot_table(index=["design", "frame"], columns="variant",
                                                         values=["R2_B1", "skill_vs_LINE"])
    c5.columns = [f"{a}_{b}" for a, b in c5.columns]
    c5 = c5.reset_index()
    c5["label"] = [EW.SEEN + ("; SENSITIVITY (design BP)" if d == "BP" else "")
                   + "; C5-BTBP out / single cell out = SENSITIVITY" for d in c5.design]
    lt = cl[(cl.variant == "all") & cl.frame.isin(EW.MAIN_FRAMES)][
        ["design", "frame", "lopo_P", "lopo_R2_B1_min", "lopo_R2_B1_max", "lopo_skill_vs_LINE_lo", "lopo_skill_vs_LINE_hi",
         "lopo_skill_vs_LINE_min", "lopo_skill_vs_LINE_min_by", "lopo_skill_vs_LINE_max", "lopo_skill_vs_LINE_max_by",
         "units_WLT_vs_LINE", "top3_unit_share_vs_LINE", "units_WLT_vs_B0", "top3_unit_share_vs_B0", "label"]]
    ut = pd.concat([T[d]["unit_tests"] for d in S], ignore_index=True)
    ut = ut[ut.variant == "all"]
    dup = pd.concat([T[d]["duplicates"] for d in S], ignore_index=True)
    ps = {"quantity": f"R2_B1 per dev seed ({', '.join(map(str, EW.SEEDS))}, in that order), variant all"}
    for d in S:
        c = T[d]["classes"][T[d]["classes"].variant == "all"]
        ps[_dkey(d)] = {fr: r for fr, r in zip(c.frame, c.R2_B1_per_seed) if fr in EW.MAIN_FRAMES}
    pp = pd.concat([pop[d][pop[d].family == f] for f in ("WIDE", "STRICT") for d in S], ignore_index=True)
    chk = {d: S[d]["checks"] for d in S}
    leak = "; ".join(f"{d}: isolation {c['isolation_ok']}/{c['folds']} folds, intervention failed {c['intervention_failed']}"
                     f"/{c['intervention_checked']}, peek caught {c['peek_control_caught']}/{c['peek_control_runs']}, "
                     f"both-ends-read scored {c['both_ends_read_bad']}, support not exact {c['support_not_exact_bad']}, "
                     f"orientation {c['orientation_bad']}" for d, c in chk.items()) + \
        "; no fitted or selected quantity (stored round-4 betas), so fit / cache poisoning do not apply"
    nd = {d: S[d]["ndsm_primary"] for d in S}
    changed = ("Pre-declared Nd/Sm B1 - LINE contrast (clusters with >= 10 rows per seed, giant cluster removed, "
               "leave-one-cluster-out jackknife-t 95 %): " + "; ".join(
                   f"{d} {r['estimate']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}] over G = {r['G']} -> "
                   f"{'RETIRED' if r['retired'] else 'NOT RETIRED'}" for d, r in nd.items())
               + " (BPG decides; EXPLORATORY seen).  Heavy cuts stay explicit non-claims; E-W is never pooled with "
                 "the strict pairs.")
    item = {"item": "1.7 E-W: M13 wider-pair table", "population": pp,
            "design": "BPG (primary) and BP (labelled sensitivity); identical test folds",
            "read_budget": "4 reads (E-A's DOPT_Q read set, code k = 3); the same reads as E-A",
            "arms": "B1 (stored round-4 fold beta via r4.refs.predict_levels), B0 (beta = 0), LINE (slope-only, "
                    "gen20k naive_line), HIST_D7 (fixed -0.072; DIAGNOSTIC)",
            "metrics": met, "per_seed": ps, "units": ut, "lopo_top3": lt, "c5": c5, "duplicates": dup, "leakage": leak,
            "compute": f"{compute_s:.0f} s ({', '.join(f'{d} {S[d]['seconds']:.0f} s' for d in S)})",
            "label": EW.SEEN + "; reproduction run TECHNICAL SMOKE; BP, C5-BTBP out, duplicates SENSITIVITY; heavy cuts "
                               "NON-CLAIM",
            "changed": changed,
            "next_step": "Registry item 1.8 (M07 Nd dip, E-KX)."}
    return CM.per_item_table(item, title="Item 1.7 (E-W): per-item table")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", choices=("BPG", "BP", "both"), default="both")
    ap.add_argument("--smoke", action="store_true", help="technical smoke: metrics suppressed, nothing written")
    ap.add_argument("--smoke-folds", type=int, default=2, help="folds per design in --smoke (default 2)")
    ap.add_argument("--repro", action="store_true", help="only the reproduction target of brief 6.0 (TECHNICAL SMOKE)")
    ap.add_argument("--from-rows", action="store_true", help="re-run the analysis from results/EW/rows_<design>.parquet")
    ap.add_argument("--no-intervention", action="store_true", help="skip the prediction-intervention re-runs (--smoke only)")
    ap.add_argument("--no-lopo", action="store_true", help="skip leave-one-publication-out (--smoke only)")
    ap.add_argument("--write-spec", action="store_true", help="write results/EW/ITEM_SPEC.md and exit (no data read)")
    ap.add_argument("--amend", action="store_true", help="--write-spec after the seal (an amendment and a reseal follow)")
    args = ap.parse_args(argv)
    from r5 import sealcheck
    if args.write_spec:
        sealcheck.refuse_if_sealed("--write-spec (results/EW/ITEM_SPEC.md is a sealed input)", args.amend)
        paths.write_text_lf(OUT / "ITEM_SPEC.md", EW.item_spec_text())
        print(f"wrote {OUT / 'ITEM_SPEC.md'}")
        return 0
    sealcheck.forbid_reduced_coverage(bool(args.smoke), no_intervention=bool(args.no_intervention),
                                      no_lopo=bool(args.no_lopo))
    if args.smoke:
        if args.from_rows:
            raise SystemExit("--from-rows reads written results; not a smoke")
        return body(args, "SMOKE")
    tag = sealcheck.require_seal()
    mode = "repro" if args.repro else ("from-rows" if args.from_rows else "run")
    return sealcheck.run_sealed(ITEM, mode, tag, lambda: body(args, tag))


def body(args, tag: str) -> int:
    from r5 import sealcheck
    t0 = time.time()
    written: list[Path] = []
    smoke = bool(args.smoke)
    mode = "smoke" if smoke else ("repro" if args.repro else ("from-rows" if args.from_rows else "run"))
    designs = ("BPG", "BP") if args.design == "both" else (args.design,)
    print(f"item 1.7 (E-W) {'PRE-SEAL ENGINEERING CHECK (--smoke; metrics suppressed; nothing written)' if smoke else 'seal ' + tag}; "
          f"designs {designs}; mode {mode}", flush=True)
    bench = D.load_bench()
    clusters = EW.cluster_map()
    print(f"  bench {len(bench.frame)} cells; C3 cells {len(clusters)}; clusters {len(set(clusters.values()))}; frozen "
          f"pair types {len(EW.PAIRS)}", flush=True)
    S, T, POP = {}, {}, {}
    for design in designs:
        td = time.time()
        if args.from_rows:
            R = pd.read_parquet(OUT / f"rows_{design}.parquet")
            checks = json.loads((OUT / f"checks_{design}.json").read_bytes())
            seals = set(R.seal.astype(str)) if "seal" in R.columns else set()
            if seals != {tag} or checks.get("seal") != tag or checks.get("code_lf_sha256") != code_digests():
                sealcheck.stop(ITEM, mode, tag, f"{design}: --from-rows refused: the stored rows / checks do not carry the "
                                                f"current seal {tag} and code digests (s0-mechanics-18)")
            bad = EW.bad_checks({k: v for k, v in checks.items() if k not in ("seal", "code_lf_sha256")})
            if bad:
                sealcheck.stop(ITEM, mode, tag, f"{design}: stored leakage / support checks failed: {bad}")
        else:
            res = EW.compute_design(bench, design, folds=args.smoke_folds if smoke else None, cluster_of=clusters,
                                    intervention=not args.no_intervention, log=lambda s: print(s, flush=True))
            R, checks = res["rows"], dict(res["checks"])
            print(f"  {design}: rows {len(R)} x {R.shape[1]} cols; {_counts(R)}; checks {checks}; {res['seconds']:.0f}s",
                  flush=True)
            bad = EW.bad_checks(checks, intervention=not args.no_intervention)
            if bad:
                if smoke:
                    raise SystemExit(f"{design}: leakage / support check failed: {bad} {checks}")
                sealcheck.stop(ITEM, mode, tag, f"{design}: leakage / support check failed: {bad}")
        rep = EW.repro_checks(R, design)
        if smoke:
            print(f"  {design} reproduction identity (PRE-SEAL ENGINEERING CHECK; partial folds {rep['folds']}): rows matched {rep['rows_matched']} "
                  f"of STRICT {rep['rows_STRICT']} / REF {rep['rows_REF']}; max |diff| vs REF "
                  f"{ {k: f'{v:.2e}' for k, v in rep['max_abs_diff'].items()} }; read sets equal share "
                  f"{rep['read_sets_equal_share']:.3f}; identity ok {rep['identity_ok']}", flush=True)
            if not rep["identity_ok"]:
                raise SystemExit(f"{design}: identity check failed in the smoke")
            if not args.repro:
                POP[design] = EW.population_table(R, design)     # counts only (section-9.3 population)
                Rs = smoke_surrogate(R, design)                  # no table ever sees a real label in the smoke
                Td = EW.analyse(Rs, design, lopo_on=not args.no_lopo, log=lambda s: print(s, flush=True))
                for k, v in Td.items():
                    print(_shape_report(k, v), flush=True)
                T[design] = Td
                S[design] = EW.summary(Td, design, checks, rep, round(time.time() - td, 1))   # in memory only
                print(f"    summary_{design} (surrogate): {len(json.dumps(S[design], default=str))} characters (not "
                      "written)", flush=True)
            print(f"  {design} smoke done ({time.time() - td:.0f}s)", flush=True)
            continue
        rep.update({"seal": tag, "code_lf_sha256": code_digests()})
        _write_json(rep, OUT / f"repro_{design}.json", written)
        print(f"  {design} reproduction (TECHNICAL SMOKE): identity {rep['identity_ok']}; rows matched "
              f"{rep['rows_matched']}/{rep['rows_REF']}; max |diff| {rep['max_abs_diff']}; ok {rep['ok']}", flush=True)
        if not rep["ok"]:
            sealcheck.stop(ITEM, mode, tag, f"{design}: reproduction mismatch -- the item stops (brief 6.0); see "
                                            f"results/EW/repro_{design}.json")
        if args.repro:
            continue
        # PROTOCOL 12: the per-item leakage report (the counters checked above), before any scored write
        _write_json(leakage_record(checks, design, tag), LEAK / f"EW_{design}.json", written)
        POP[design] = EW.population_table(R, design)            # counts only (section-9.3 population)
        if not args.from_rows:
            OUT.mkdir(parents=True, exist_ok=True)
            checks.update({"seal": tag, "code_lf_sha256": code_digests()})
            R.assign(label=EW.row_labels(R), seal=tag).to_parquet(OUT / f"rows_{design}.parquet", index=False)
            written.append(OUT / f"rows_{design}.parquet")
            _write_json(checks, OUT / f"checks_{design}.json", written)
        Td = EW.analyse(R, design, lopo_on=not args.no_lopo, log=lambda s: print(s, flush=True))
        for k, v in Td.items():
            _write_csv(v.assign(seal=tag) if len(v) else v, OUT / f"{k}_{design}.csv", written)
        T[design] = Td
        S[design] = EW.summary(Td, design, checks, rep, round(time.time() - td, 1))
        S[design].update({"seal": tag, "code_lf_sha256": code_digests()})
        _write_json(S[design], OUT / f"summary_{design}.json", written)
    if smoke and set(S) == {"BPG", "BP"}:
        txt = item_table(S, T, POP, time.time() - t0)
        print(f"  per-item table built in memory on the surrogate: {len(txt.splitlines())} lines (not written)", flush=True)
    if not smoke and not args.repro and set(S) == {"BPG", "BP"}:
        paths.write_text_lf(OUT / "ITEM_TABLE.md", item_table(S, T, POP, time.time() - t0) + f"\nSeal: {tag}\n")
        written.append(OUT / "ITEM_TABLE.md")
    for p in written:
        print(f"wrote {p}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
