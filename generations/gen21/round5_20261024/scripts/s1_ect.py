#!/usr/bin/env python
"""Stage-1 item 1.1, the E-CT core (CLAUDE_CODE_PROMPT.md section 6.1; conventions 6.0, 6.10, 8, 9).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ect.py \
        [--design BPG|BP|both] [--smoke [--intervention-seeds N] | --repro]

Modes (run from the repository root):
  default   requires the round-5 seal (r5.sealcheck.require_seal); builds every arm of the 6.1 table on every round-4
            REF row (r5.ect), runs the item's reproduction target first (results/ECT/repro_<design>.json; a mismatch
            stops the item), then the section-8 prediction intervention with its positive controls on every dev seed
            (a failure stops the item before any scored write), then writes the row-level parquet and every table under
            results/ECT/ (labelled per section 9.1) and the leakage report under results/leakage/; START / END lines
            in RUN_RECORD.md and the post-run seal check (r5.sealcheck.post_run).  EXPLORATORY (seen): every in-corpus
            number has been seen; none is placed next to 0.452.
  --repro   requires the seal; the item's section-6.0 reproduction targets only (B1 from the fold betas within
            8.9e-16; E6's V1 population SIB 0.5734 / SH(K = 2) 0.5946 / B1 0.4489; P1's PRIMARY without the bit rule
            B1 0.5309 / SH 0.7082 / SIB 0.6987 / OFF 0.6963), labelled TECHNICAL SMOKE, written to
            results/ECT/repro_<design>.json; exit code 1 on a mismatch (a mismatch stops the item).
  --smoke   no seal needed; PRE-SEAL ENGINEERING CHECK with every metric suppressed: the rows are built on real data with the
            expectation arms scored against seeded noise, the tables run on a noise surrogate of the row frame (NaN
            pattern kept), and only shapes, counts, NaN patterns, identity checks and timings are printed and written
            (results/ECT/SMOKE_COUNTS_<design>.json).  No R2, SSE, MAE or contrast value is computed on real labels.
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

from r5 import ect as ECT  # noqa: E402
from r5 import ect_report as REP  # noqa: E402

paths.redirect_inner_cache()
OUT = paths.RES / "ECT"
LEAK = paths.RES / "leakage"
CODE = ("r5/ect.py", "r5/ect_report.py", "r5/population.py", "r5/dupes.py", "r5/common.py", "scripts/s1_ect.py")


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(p: Path, obj) -> None:
    paths.write_text_lf(p, json.dumps(obj, indent=1, default=_js) + "\n")
    print("wrote", p)


def write_csv(p: Path, df: pd.DataFrame) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    print("wrote", p)


def code_digests() -> dict:
    return {c: paths.sha256_text_lf(paths.R5 / c) for c in CODE}


def smoke_counts(R: pd.DataFrame, checks: dict, T: dict, iv: dict, seconds: float) -> dict:
    """Counts only: shapes, NaN shares, stratum sizes, table shapes, identity checks and timings."""
    num = [c for c in R.columns if R[c].dtype.kind == "f"]
    flags = [c for c in R.columns if c.startswith("in_")]
    return {"label": "PRE-SEAL ENGINEERING CHECK (--smoke: real inputs, surrogate labels, every metric suppressed)", "rows_shape": list(R.shape),
            "builder_checks": checks, "nan_share": {c: round(float(R[c].isna().mean()), 4) for c in num},
            "stratum_rows": {c: int(R[c].sum()) for c in flags},
            "stratum_cells": {c: int(R[R[c]].cell_id.nunique()) for c in flags},
            "geom_same_rows": int(R.geom_same.sum()), "v25_rows_ok": int(R.v25_row_ok.sum()),
            "np_ladder_sources": {f"r{r}": R[f"np_lad_src_r{r}"].value_counts().to_dict() for r in (2, 3, 4, 5)},
            "table_shapes_on_surrogate": {k: list(v.shape) for k, v in T.items()},
            "intervention": iv, "seconds": round(seconds, 1)}


ITEM = "ECT"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=("BPG", "BP", "both"), default="both")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="technical smoke, every metric suppressed (no seal needed)")
    g.add_argument("--repro", action="store_true", help="the section-6.0 reproduction targets only (TECHNICAL SMOKE)")
    ap.add_argument("--intervention-seeds", type=int, default=None,
                    help="dev seeds the intervention check covers (--smoke only; the registered run covers all 5)")
    a = ap.parse_args(argv)
    from r5 import sealcheck
    sealcheck.forbid_reduced_coverage(a.smoke, intervention_seeds=a.intervention_seeds)
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
    corpus = ECT.Corpus()
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if mode == "run":
        tt = corpus.target_table()
        tt["label"] = "METADATA (population construction; no score)"
        write_csv(OUT / "populations.csv", tt)
        L = corpus.links_table()
        L["label"] = "METADATA + label-based SENSITIVITY flags (shape sd, offset / near-id); no score"
        write_csv(OUT / "links.csv", L)
    status = 0
    for design in designs:
        t0 = time.time()
        y_mode = "surrogate" if a.smoke else "labels"
        R, checks = corpus.build_rows(design, y_mode=y_mode)
        if not a.smoke:                       # brief 1.2 / 6.0: the reproduction target first; a mismatch stops the item
            rep = REP.reproduction(R, checks, design)
            rep.update({"seal": tag, "started_utc": started, "code_lf_sha256": code_digests()})
            write_json(OUT / f"repro_{design}.json", rep)
            print(f"TECHNICAL SMOKE reproduction {design}: {'MATCH' if rep['passed'] else 'MISMATCH (item stops)'}")
            for k, v in rep["targets"].items():
                print(f"  {k}: counts {v['counts_found']} ok {v['counts_ok']}; values {v['values_found']}"
                      + (f" ok {v['values_ok']}" if "values_ok" in v else ""))
            if a.repro:
                status |= 0 if rep["passed"] else 1
                continue
            if not rep["passed"]:
                sealcheck.stop(ITEM, mode, tag, f"reproduction mismatch ({design}; results/ECT/repro_{design}.json) -- "
                                                "item stops (brief 6.0)")
        n_iv = a.intervention_seeds if a.intervention_seeds is not None else (1 if a.smoke else len(paths.DEV_SEEDS))
        iv = corpus.intervention(design, seeds=paths.DEV_SEEDS[:n_iv], y_mode=y_mode)
        if a.smoke:
            S = REP.surrogate(R, design)
            T = REP.all_tables(S, design)
            txt = REP.per_item_table(REP.per_item(T, design, seal_tag=tag, leakage="smoke", seconds=0.0), None)
            print(f"  per-item table exercised on the surrogate: {len(txt.splitlines())} lines (not written)")
            counts = smoke_counts(R, checks, T, iv, time.time() - t0)
            write_json(OUT / f"SMOKE_COUNTS_{design}.json", counts)
            print(f"  {design}: rows {R.shape}; strata rows {counts['stratum_rows'].get('in_PRIMARY')} (PRIMARY); "
                  f"tables on the surrogate: {len(T)}; intervention checks ok {iv['checks_ok']} (coverage full "
                  f"{iv['coverage_full']}; {time.time() - t0:.0f}s)")
            for k, v in T.items():
                print(f"    table {k}: {v.shape}")
            continue
        # brief 8 / s0-mechanics-17: the leakage checks run before any scored write; a failure stops the item
        leak = {"label": "LEAKAGE CHECK (section 8 items 4-5)", "seal": tag, "intervention": iv,
                "support": {k: checks[k] for k in ("nested_reads_ok", "legal_steps_ok", "both_ends_read_rows",
                                                   "readset_mismatch", "b1_maxabs", "y_maxabs")},
                "fit_poisoning": "not applicable: E-CT fits nothing (beta = the stored round-4 fold fit; K fixed at 1.78)",
                "cache_poisoning": "not applicable: no new cache (read sequences from the gen21 fold cache)",
                "code_lf_sha256": code_digests()}
        write_json(LEAK / f"ECT_{design}.json", leak)
        if not iv["passed"]:
            sealcheck.stop(ITEM, mode, tag, f"leakage check failed ({design}; results/leakage/ECT_{design}.json): item "
                                            "stopped")
        T = REP.all_tables(R, design)
        R["label"] = f"{REP.SEEN}; {REP.NOT_NEXT_TO}"
        R["seal"] = tag
        R.to_parquet(OUT / f"rows_{design}.parquet", index=False)
        print("wrote", OUT / f"rows_{design}.parquet")
        for k, v in T.items():
            write_csv(OUT / f"{k}_{design}.csv", v)
        lk = (f"intervention {'passed' if iv['passed'] else 'FAILED'} on {iv['targets']} targets x 2 passes; positive "
              f"controls caught {iv['pc_self_guard_caught']}/{iv['pc_copy_guard_caught']}/{iv['pc_self_intervention_caught']}"
              f" of {iv['pc_targets']}; supports exact, nested, legal ({checks['legal_steps_ok']}); B1 max diff "
              f"{checks['b1_maxabs']:.1e}")
        item = REP.per_item(T, design, seal_tag=tag, leakage=lk, seconds=time.time() - t0)
        REP.per_item_table(item, OUT / f"PER_ITEM_TABLE_{design}.md", title=f"1.1 E-CT core ({design}): per-item table")
        print("wrote", OUT / f"PER_ITEM_TABLE_{design}.md")
        write_json(OUT / f"summary_{design}.json", {
            "label": REP.SEEN, "note": REP.NOT_NEXT_TO, "seal": tag, "started_utc": started, "design": design,
            "builder_checks": checks, "intervention_passed": iv["passed"], "tables": sorted(T),
            "code_lf_sha256": code_digests(), "seconds": round(time.time() - t0, 1)})
    print(f"done {time.time() - t00:.0f}s (status {status})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
