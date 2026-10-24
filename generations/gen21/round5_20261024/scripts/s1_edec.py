#!/usr/bin/env python
"""Stage-1 item 1.5 (CLAUDE_CODE_PROMPT.md 6.5; 6.0, 6.10, 8, 9): M14 selective direction calls (E-DEC) with the E-REC
own-prior row (I2 recipe-3) and the E-RANK table (IR M15).  Descriptive: EXPLORATORY (seen) throughout, no kill, never
an R2 gain, never next to 0.452.  Frozen arm list, report rules and seeds: results/EDEC/ITEM_SPEC.md.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_edec.py --smoke --design BPG
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_edec.py --repro --design both
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_edec.py --design both

Modes (run from the repository root; order per brief 1.2: Stage-0 seal -> item code seal -> --repro -> the arms):
  default   requires the round-5 seal (r5.sealcheck.require_seal).  Builds the E-A rows (r5.common.load_rows) and the
            r5.ect rows (results/ECT/rows_<design>.parquet when it carries the current seal, else r5.ect.Corpus), runs
            every arm, comparator, control, stratum and sensitivity of r5.edec, the fit-poisoning check of every
            calibrated quantity (section 8 item 2; scope: calibration labels only) with its leaky positive control,
            after the reproduction target (a mismatch stops the item) and before any scored write (a failure stops
            the item), and writes under results/EDEC/:
            rows_<design>.parquet (C5-BTBP in) and rows_<design>_C5out.parquet (every call, Platt probability and
            interval per row), erec_rows_<design>.parquet, thresholds_<design>.parquet, one CSV per table
            (<table>_<design>.csv; every number labelled), summary_<design>.json, PER_ITEM_TABLE_<design>.md,
            repro_<design>.json (the in-run reproduction check, labelled as such); and results/leakage/EDEC_<design>.json
            (its support record counted on the rows).  Appends START / END lines to RUN_RECORD.md (post-run seal check).
  --repro   requires the seal; the reproduction targets that apply to this item (brief 6.0 lists no E-DEC value
            target): B1 recomputed from the fold betas (r5.ect B1_r4) equals the REF parquet within 8.9e-16 on every
            row (18,020 over both designs); D7 at 4 reads from the B1 / B0 identity equals REF HIST_D7 (1e-12); y
            identical; the support contract.  TECHNICAL SMOKE; results/EDEC/repro_<design>.json; exit 1 on a mismatch.
  --smoke   no seal needed; PRE-SEAL ENGINEERING CHECK with every metric suppressed: y and every prediction column are replaced
            by seeded noise (NaN pattern kept) before any table or calibration runs, so no real score is computed;
            prints shapes, counts, NaN patterns, identity checks and timings only and writes
            results/EDEC/SMOKE_COUNTS_<design>.json (counts only).  The poisoning check runs on the surrogate (1 seed).
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r5 import common as CM  # noqa: E402
from r5 import edec as ED  # noqa: E402

paths.redirect_inner_cache()
OUT = ED.OUT
LEAK = paths.RES / "leakage"
CODE = ("r5/edec.py", "scripts/s1_edec.py", "r5/common.py", "r5/ect.py", "r5/population.py", "r5/dupes.py", "r5/paths.py")
# Copy rule (brief 1.2): the recipe-3 comparator source with its LF SHA-256, checked by scripts/r5_seal.py.
COPIED_FROM = {"generations/gen21/round5_design_20261023/ideas2/scripts/recipe3_skepB_signcheck.py":
               "c9ae42cc2b54ea3c6cb2ddb676b0eb4b76b3a5e5b5bfa5be631e0c4a3e2aebb6"}
ITEM = "EDEC"
KEY = ["split_seed", "fold", "cell_id", "A", "B"]
EREC_PRED = ("COPY", "SH_r2", "SH_r4", "NB_COPY", "NB_SH_r2", "NB_SH_r4", "B1_r2", "B0_r2", "B1_r4", "B0_r4", "G14Z",
             "CHPT_MAJ", "D7_r2", "D7_r4", "B1", "HIST_D7")
EREC_KEEP = KEY + ["publication", "chemotype", "cluster_cpg", "pubchem", "pair", "y", "n_sib", "cls", "c5", "t_quant",
                   "t_v2drop", "dga", "in_GE1", "in_GE1_WITH_C5", "in_GE1_NOBIT", "chpt_source", "chpt_donor_rows",
                   "beta_fold"] + list(EREC_PRED)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _js(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    return str(o)


def write_json(p: Path, obj) -> None:
    paths.write_text_lf(p, json.dumps(obj, indent=1, default=_js) + "\n")
    print("wrote", p)


def write_csv(p: Path, df: pd.DataFrame) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    print("wrote", p)


def digests() -> dict:
    code = {c: paths.sha256_text_lf(paths.R5 / c) for c in CODE}
    src = {s: {"recorded": h, "found": paths.sha256_text_lf(paths.ROOT / s)} for s, h in COPIED_FROM.items()}
    return {"code_lf_sha256": code, "sources_lf_sha256": src,
            "sources_match": all(v["recorded"] == v["found"] for v in src.values())}


def load_ect(log=print):
    """(r5.ect module, Corpus) or (None, reason): the E-REC row is PENDING-ECT without them."""
    try:
        from r5 import ect as ECT
    except Exception as e:  # noqa: BLE001
        return None, None, f"PENDING-ECT (r5.ect not importable: {type(e).__name__}: {e})"
    try:
        corpus = ECT.Corpus(log=log)
    except Exception as e:  # noqa: BLE001
        return ECT, None, f"PENDING-ECT (r5.ect.Corpus failed: {type(e).__name__}: {e})"
    return ECT, corpus, ""


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return {}


def ect_rows(corpus, design: str, tag: str, smoke: bool) -> tuple[pd.DataFrame, dict]:
    """The r5.ect rows of one design: the sealed results/ECT parquet only when it carries this seal AND item 1.1's
    intervention (results/leakage/ECT_<d>.json) and reproduction (results/ECT/repro_<d>.json) passed under the same
    seal (s0-mechanics-17); else a fresh build checked by its own intervention (expectation arms against seeded noise
    in --smoke; this item never uses them)."""
    p = paths.RES / "ECT" / f"rows_{design}.parquet"
    if not smoke and p.exists():
        lk, rp = _json(paths.RES / "leakage" / f"ECT_{design}.json"), _json(paths.RES / "ECT" / f"repro_{design}.json")
        ok_lk = lk.get("seal") == tag and (lk.get("intervention") or {}).get("passed") is True
        ok_rp = rp.get("seal") == tag and rp.get("passed") is True
        E = pd.read_parquet(p)
        if ok_lk and ok_rp and "seal" in E.columns and (E.seal.astype(str) == tag).all():
            print(f"  E-CT rows from {p} (seal {tag}; item 1.1 intervention and reproduction passed)")
            return E, {"source": str(p)}
        print(f"  E-CT rows at {p} not reused (seal / leakage / reproduction record not current): rebuilding")
    E, checks = corpus.build_rows(design, y_mode="surrogate" if smoke else "labels")
    checks["source"] = "r5.ect.Corpus.build_rows"
    if not smoke:
        iv = corpus.intervention(design, y_mode="labels")
        checks["ect_intervention_passed"] = bool(iv["passed"])
        if not iv["passed"]:
            raise SystemExit(f"r5.ect intervention failed for the rebuilt E-CT rows ({design}): item stopped")
    return E, checks


def repro_targets(d: pd.DataFrame, checks) -> dict:
    """The reproduction targets that apply to this item (brief 6.0 lists no E-DEC value target); the support contract
    is counted on the rows (r5.edec.support_record), not asserted."""
    sup = ED.support_record(d)
    return {"B1_r4 (fold betas) = REF B1 within 8.9e-16": {
                "rows": int(len(d)), "maxabs": checks["b1_r4_vs_ref_maxabs"] if checks else None,
                "ok": bool(checks and checks["b1_r4_vs_ref_maxabs"] <= 8.9e-16 and len(d) == 9010)},
            "D7_r4 (identity) = REF HIST_D7 within 1e-12": {
                "maxabs": checks["d7_r4_vs_ref_hist_d7_maxabs"] if checks else None,
                "ok": bool(checks and checks["d7_r4_vs_ref_hist_d7_maxabs"] <= 1e-12)},
            "y identical (E-CT rows vs E-A rows)": {"maxabs": checks["y_vs_ref_maxabs"] if checks else None,
                                                    "ok": bool(checks and checks["y_vs_ref_maxabs"] == 0.0)},
            "support contract (no pair with both ends read; strict adjacency; hull flags from read sets)": {
                **{k: sup[k] for k in ("rows", "both_ends_read_rows", "non_strict_adjacent_rows",
                                       "hull_recount_mismatch_rows")}, "ok": sup["ok"]}}


def shape_line(name: str, df) -> str:
    if not isinstance(df, pd.DataFrame):
        return f"    {name}: {type(df).__name__}"
    num = df.select_dtypes("number")
    nan_cols = int(num.isna().any().sum()) if len(num.columns) else 0
    return f"    {name}: {df.shape[0]} rows x {df.shape[1]} cols; numeric cols with NaN {nan_cols}/{len(num.columns)}"


def smoke_counts(d, F, T, rows, iv, checks, va, vb, seconds) -> dict:
    """Counts only: populations, hull eligibility, censored-read rows, E-REC strata, table shapes, identities."""
    ns = d.split_seed.nunique()
    pt = d.groupby("pair").agg(rows=("y", "size"), out_of_hull=("hull", lambda h: int((~h).sum())))
    pt = (pt / ns).round(2).to_dict("index")
    cens = int(((np.isfinite(va) & (va <= ED.LIMIT)) | (np.isfinite(vb) & (vb <= ED.LIMIT))).sum())
    pop = T.get("erec_population")
    return {"label": "PRE-SEAL ENGINEERING CHECK (--smoke: every metric suppressed; tables ran on a seeded-noise surrogate)",
            "rows": CM.population(d, "cluster_cpg"), "hull_eligible_rows_per_seed": float(d.hull.sum() / ns),
            "pair_types_per_seed": pt, "rows_with_read_end_at_or_below_limit_per_seed": cens / ns,
            "erec_population": pop.to_dict("records") if isinstance(pop, pd.DataFrame) else None,
            "erec_identity_checks": checks, "erec_rows_shape": list(F.shape) if F is not None else None,
            "table_shapes_on_surrogate": {k: list(v.shape) for k, v in T.items() if isinstance(v, pd.DataFrame)},
            "row_frames_on_surrogate": {k: list(v.shape) for k, v in rows.items()},
            "poisoning_on_surrogate": {k: iv[k] for k in ("clusters", "seeds", "quantities_compared", "clusters_with_diff",
                                                          "pc_caught", "pc_clusters", "passed")},
            "seconds": round(seconds, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", choices=("BPG", "BP", "both"), default="both")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="technical smoke, every metric suppressed (no seal needed)")
    g.add_argument("--repro", action="store_true", help="the applicable section-6.0 reproduction targets (TECHNICAL SMOKE)")
    ap.add_argument("--poison-seeds", type=int, default=None,
                    help="dev seeds the fit-poisoning check covers (--smoke only; the registered run covers all 5)")
    a = ap.parse_args(argv)
    from r5 import sealcheck
    sealcheck.forbid_reduced_coverage(a.smoke, poison_seeds=a.poison_seeds)
    if a.smoke:
        print("PRE-SEAL ENGINEERING CHECK (--smoke): every metric suppressed; no seal required")
        return body(a, "UNSEALED (technical smoke)")
    tag = sealcheck.require_seal()
    return sealcheck.run_sealed(ITEM, "repro" if a.repro else "run", tag, lambda: body(a, tag))


def body(a, tag: str) -> int:
    from r5 import sealcheck
    t00 = time.time()
    mode = "smoke" if a.smoke else ("repro" if a.repro else "run")
    designs = ("BPG", "BP") if a.design == "both" else (a.design,)
    started = _now()
    dg = digests()
    if not dg["sources_match"]:
        print("WARNING: a recorded source digest differs:", dg["sources_lf_sha256"])
    t1 = time.time()
    ECT, corpus, erec_status = load_ect()
    if corpus is not None:
        bench = corpus.bench
    else:
        from gen20k import data as D
        bench = D.load_bench()
        print(" ", erec_status)
    print(f"  setup {time.time() - t1:.0f}s; E-REC {'available' if corpus is not None else erec_status}")
    status = 0
    for design in designs:
        t0 = time.time()
        d = ED.prepare(CM.load_rows(design))
        va, vb = ED.read_values(d, bench)
        F, checks = None, None
        if corpus is not None:
            E, bchk = ect_rows(corpus, design, tag, a.smoke)
            if a.smoke:      # real predictions (identity checks), surrogate labels (CHPT_MAJ, never scored)
                d_y = d.assign(y=ED.surrogate(d, design, ED.ARMS).y.to_numpy())
                E = E.drop(columns=["y"]).merge(d_y[KEY + ["y"]], on=KEY, how="left", validate="one_to_one")
                F, checks = ED.erec_frame(E, d_y, design)
            else:
                F, checks = ED.erec_frame(E, d, design)
            checks.update({k: bchk.get(k) for k in ("b1_maxabs", "y_maxabs", "readset_mismatch", "nested_reads_ok",
                                                     "legal_steps_ok", "both_ends_read_rows", "source",
                                                     "ect_intervention_passed") if k in bchk})
            print(f"  {design}: E-REC rows {F.shape}; identities B1_r4 {checks['b1_r4_vs_ref_maxabs']:.1e}, "
                  f"D7_r4 {checks['d7_r4_vs_ref_hist_d7_maxabs']:.1e}, min |beta| {checks['beta_min_abs']:.3f}")
        if not a.smoke:                       # brief 1.2 / 6.0: the reproduction target first; a mismatch stops the item
            tg = repro_targets(d, checks)
            # the arms run repeats the check in line and rewrites this file: labelled as such, not as the --repro run
            rlab = "TECHNICAL SMOKE (reproduction run, brief 6.0)" if a.repro else "in-run reproduction check (arms run)"
            rep = {"label": rlab, "mode": mode, "design": design, "seal": tag,
                   "started_utc": started, "targets": tg, "erec_status": erec_status or "available",
                   "note": "brief 6.0 lists no E-DEC value target; IR M14 / M15 / I2 recipe-3 values are seen, not targets",
                   "passed": bool(all(v["ok"] for v in tg.values())), **dg}
            if not a.repro:
                rep["record"] = ("the arms run re-checks the brief-6.0 targets before any scored write and rewrites this "
                                 "file; the standalone --repro run (TECHNICAL SMOKE) is recorded in RUN_RECORD.md")
            write_json(OUT / f"repro_{design}.json", rep)
            print(f"{'TECHNICAL SMOKE reproduction' if a.repro else rlab} {design}: "
                  f"{'MATCH' if rep['passed'] else 'MISMATCH (item stops)'}")
            if a.repro:
                status |= 0 if rep["passed"] else 1
                continue
            if not rep["passed"]:
                sealcheck.stop(ITEM, mode, tag, f"reproduction mismatch ({design}; results/EDEC/repro_{design}.json) -- "
                                                "item stops (brief 6.0)")
        n_poison = a.poison_seeds if a.poison_seeds is not None else (1 if a.smoke else len(paths.DEV_SEEDS))
        if a.smoke:
            d_s = ED.surrogate(d, design, ED.ARMS)
            F_s = ED.surrogate(F, design, EREC_PRED, key="EDEC|SMOKE_SURROGATE_EREC") if F is not None else None
            T, rows, extra = ED.design_tables(d_s, design, va=va, vb=vb, F=F_s, erec_status=erec_status)
            iv = ED.poison_check(d_s, design, seeds=paths.DEV_SEEDS[:n_poison])
            print(f"  {design}: tables on the surrogate ({time.time() - t0:.0f}s):")
            for k, v in T.items():
                print(shape_line(k, v))
            for k, v in rows.items():
                print(shape_line("rows " + k, v))
            sc = smoke_counts(d, F, T, rows, iv, checks, va, vb, time.time() - t0)
            print(f"  {design}: E-A rows {d.shape}; hull-eligible rows per seed {sc['hull_eligible_rows_per_seed']:.1f}; "
                  f"read end at or below {ED.LIMIT} per seed {sc['rows_with_read_end_at_or_below_limit_per_seed']:.1f}; "
                  f"poisoning (surrogate) checks ok {iv['checks_ok']} (coverage full {iv['coverage_full']})")
            if sc["erec_population"]:
                for r in sc["erec_population"]:
                    print(f"    E-REC {r['stratum']}: cells {r['cells']}, publications {r['publications']}, units "
                          f"{r['units_pubchem']}, rows per seed {r['rows_per_seed']:.1f}")
            write_json(OUT / f"SMOKE_COUNTS_{design}.json", sc)
            continue
        # brief 8 / s0-mechanics-17: the fit-poisoning check runs before any scored write; a failure stops the item
        iv = ED.poison_check(d, design, seeds=paths.DEV_SEEDS[:n_poison])
        leak = {"label": "LEAKAGE CHECK (section 8 items 2 and 5)", "seal": tag, "design": design, "fit_poisoning": iv,
                "scope": ED.POISON_SCOPE,
                "support": ED.support_record(d, va, vb),
                "erec_identities": checks,
                "chpt_majority": "leave-own-publication-out by construction (donor rows of other publications; C5-BTBP never a donor)",
                "note": "predictions are round-4 OOF rows (round-4 leakage chain) and r5.ect arms (its own intervention check, item 1.1)",
                **dg}
        write_json(LEAK / f"EDEC_{design}.json", leak)
        if not iv["passed"]:
            sealcheck.stop(ITEM, mode, tag, f"fit-poisoning check failed ({design}; results/leakage/EDEC_{design}.json): "
                                            "item stopped")
        T, rows, extra = ED.design_tables(d, design, va=va, vb=vb, F=F, erec_status=erec_status)
        lab = f"{ED.label(design, ED.NOT_R2)}; seal {tag}"
        for k, R in rows.items():
            R = R.assign(label=lab, seal=tag)
            p = OUT / (f"rows_{design}.parquet" if k == "C5_in" else f"rows_{design}_C5out.parquet")
            R.to_parquet(p, index=False)
            print("wrote", p)
        if F is not None:
            p = OUT / f"erec_rows_{design}.parquet"
            F[[c for c in EREC_KEEP if c in F.columns]].assign(label=lab, seal=tag).to_parquet(p, index=False)
            print("wrote", p)
        th = T.pop("thresholds")
        th.assign(label=ED.label(design, "fitted thresholds (leave-one-cluster-out)"), seal=tag).to_parquet(
            OUT / f"thresholds_{design}.parquet", index=False)
        print("wrote", OUT / f"thresholds_{design}.parquet")
        for k, v in T.items():
            write_csv(OUT / f"{k}_{design}.csv", v.assign(seal=tag) if len(v) else v)
        lk = (f"fit poisoning {'passed' if iv['passed'] else 'FAILED'}: {iv['quantities_compared']} fitted values over "
              f"{iv['clusters']} clusters x {len(iv['seeds'])} seeds bit-identical when the held-out cluster's labels are "
              f"replaced; leaky positive control caught {iv['pc_caught']}/{iv['pc_clusters']}; scope: calibration "
              "labels only (the prediction models are not refitted; B0 is training-free)")
        item = ED.per_item(T, extra, design, d, seal_tag=tag, leakage=lk, seconds=time.time() - t0, ect_checks=checks)
        CM.per_item_table(item, OUT / f"PER_ITEM_TABLE_{design}.md", title=f"1.5 E-DEC ({design}): per-item table")
        print("wrote", OUT / f"PER_ITEM_TABLE_{design}.md")
        write_json(OUT / f"summary_{design}.json", {
            "label": ED.label(design, ED.NOT_R2), "seal": tag, "started_utc": started, "design": design,
            "erec_status": extra["erec_status"], "erec_identity_checks": checks, "fit_poisoning_passed": iv["passed"],
            "fit_poisoning_scope": ED.POISON_SCOPE,
            "primary": extra["primary"], "seen_values_recorded": ED.SEEN_VALUES, "tables": sorted(T),
            "seconds": round(time.time() - t0, 1), **dg})
    print(f"done {time.time() - t00:.0f}s (status {status})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
