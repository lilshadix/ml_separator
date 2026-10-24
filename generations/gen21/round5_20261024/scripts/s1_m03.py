#!/usr/bin/env python
"""Stage-1 item 1.2, the M03 sibling-weighting factorial, an E-CT annex (CLAUDE_CODE_PROMPT.md section 6.2;
conventions 6.0, 6.10, 8, 9).  Code: r5/m03.py (built on r5/ect.py and r5/population.py).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_m03.py \
        [--design BPG|BP|both] [--smoke [--ect-check-seeds N] [--intervention-seeds N] | --repro] [--perms B]

Modes (run from the repository root):
  default   requires the round-5 seal (r5.sealcheck.require_seal); builds every arm on every round-4 REF row of the
            n_sib >= 2 E-CT targets (BIT pool), of P1's CLEAN pool without the bit rule and of the OFS pool (offset /
            near-identity links removed; duplicate sensitivities), runs the reproduction target first (a mismatch
            stops the item) and the intervention (a failure stops the item before any scored write), then runs
            the nested leave-one-publication-out selection in every population, the selection-aware nulls (B
            permutations each, default 500), the strata, the acid-trend arms, the level rule and the report rules, the
            section-8 intervention and fit-poisoning checks, and writes every row-level output and table under
            results/ECT_M03/ (labelled per section 9.1) and the leakage report under results/leakage/.
            EXPLORATORY (seen): no number is placed next to E-A's 0.452; B1 and SH on the same rows are the comparators.
  --repro   requires the seal; the item's section-6.0 reproduction targets only (B1 from the fold betas within
            8.9e-16 of the parquet; every shared arm bit-identical to r5.ect; the kernel's uniform limit = SH / SIB;
            P1's PRIMARY without the bit rule B1 0.5309 / SH 0.7082 / SIB 0.6987 through this item's kernel code),
            labelled TECHNICAL SMOKE, written to results/ECT_M03/repro_<design>.json; exit code 1 on a mismatch.
  --smoke   no seal needed; PRE-SEAL ENGINEERING CHECK with every metric suppressed: the rows are built on real data (expectation
            arms against seeded noise), then y, the templates and every prediction are replaced by seeded noise and
            the whole analysis runs on that surrogate (few permutations); only shapes, counts, NaN patterns, identity
            checks and timings are printed and written (results/ECT_M03/SMOKE_COUNTS_<design>.json).
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
from r5 import m03 as M3  # noqa: E402

paths.redirect_inner_cache()
OUT = paths.RES / M3.OUTDIR
LEAK = paths.RES / "leakage"
CODE = ("r5/m03.py", "r5/ect.py", "r5/ect_report.py", "r5/population.py", "r5/dupes.py", "r5/common.py",
        "scripts/s1_m03.py")
SMOKE_PERMS = 3


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


def code_digests() -> dict:
    return {c: paths.sha256_text_lf(paths.R5 / c) for c in CODE}


def smoke_counts(builds: dict, checks: dict, T: dict, iv: dict, fp: dict, seconds: dict) -> dict:
    """Counts only: shapes, NaN shares, population sizes, table shapes, identity checks and timings."""
    out = {"label": "PRE-SEAL ENGINEERING CHECK (--smoke: every metric suppressed; tables computed on a seeded-noise surrogate only)",
           "checks": checks, "intervention": iv, "fit_poisoning": fp, "seconds": seconds, "pools": {}}
    for pool, (R, KD) in builds.items():
        num = [c for c in R.columns if R[c].dtype.kind == "f"]
        out["pools"][pool] = {
            "rows_shape": list(R.shape), "groups": int(len(KD.S)), "kernel_tensor": list(KD.S.shape),
            "nan_share": {c: round(float(R[c].isna().mean()), 4) for c in num if not c.startswith("KW_")},
            "kw_columns": int(sum(c.startswith("KW_") for c in R.columns)),
            "acid_trend_rows": int(R.in_ACIDTREND.sum()), "acid_position_rows": R.acid_pos_bit.value_counts().to_dict(),
            "strata_rows": {c: int(R[c].sum()) for c in R.columns if c.startswith("st_")}}
    pops = T.get("populations")
    if isinstance(pops, pd.DataFrame) and len(pops):
        keep = [c for c in ("population", "cells", "publications", "units", "rows_per_seed", "cpg_clusters", "c5_cells",
                            "acid_trend_cells", "dense_cells") if c in pops.columns]
        out["populations"] = pops[keep].to_dict(orient="records")
    out["table_shapes_on_surrogate"] = {k: list(v.shape) for k, v in T.items() if isinstance(v, pd.DataFrame)}
    ch = T.get("choices")
    out["nested_fold_fits_on_surrogate"] = int(len(ch)) if isinstance(ch, pd.DataFrame) else 0
    return out


def leakage_line(iv: dict, fp: dict, checks: dict) -> str:
    """The per-item leakage status.  The bit-copy guard positive control does not apply to the NB pool (bit copies are
    allowed there by design; m03.intervention), so it reads n.a."""
    ivs = ", ".join("%s: %d groups, positive controls %d/%s/%d/%d of %d" % (
        p, v["groups"], v["pc_self_guard"], "n.a." if p == "NB" else str(v["pc_copy_guard"]), v["pc_self_kernel"],
        v["pc_peek_read"], v["pc_groups"]) for p, v in iv.items())
    return (f"intervention {'passed' if all(v['passed'] for v in iv.values()) else 'FAILED'} ({ivs}); "
            f"fit poisoning {fp['identical']}/{fp['fold_fits']} selections, {fp['poolpair_identical']}/"
            f"{fp['poolpair_pubs']} POOLPAIR slopes identical, leaky positive control caught {fp['pc_caught']}/"
            f"{fp['pc_runs']}; B1 max diff {checks['bit']['b1_maxabs']:.1e}; "
            f"r5.ect identity on {checks['bit']['ect_identity_groups']} groups")


ITEM = "ECT_M03"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=("BPG", "BP", "both"), default="both")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="technical smoke, every metric suppressed (no seal needed)")
    g.add_argument("--repro", action="store_true", help="the section-6.0 reproduction targets only (TECHNICAL SMOKE)")
    ap.add_argument("--perms", type=int, default=None, help=f"permutations per null (default {M3.B_PERM}; "
                                                              f"{SMOKE_PERMS} in --smoke); the registry needs >= 200")
    ap.add_argument("--ect-check-seeds", type=int, default=None,
                    help="dev seeds on which every shared arm is compared with r5.ect (--smoke only; default all 5)")
    ap.add_argument("--intervention-seeds", type=int, default=None,
                    help="dev seeds the intervention check covers (--smoke only; default all 5)")
    a = ap.parse_args(argv)
    from r5 import sealcheck
    sealcheck.forbid_reduced_coverage(a.smoke, ect_check_seeds=a.ect_check_seeds,
                                      intervention_seeds=a.intervention_seeds)
    B = a.perms if a.perms is not None else (SMOKE_PERMS if a.smoke else M3.B_PERM)
    if not a.smoke and not a.repro and B < 200:
        raise SystemExit(f"--perms {B}: the registered selection-aware nulls need >= 200 permutations")
    if a.smoke:
        print("PRE-SEAL ENGINEERING CHECK (--smoke): every metric suppressed; no seal required")
        return body(a, "UNSEALED (technical smoke)", B)
    tag = sealcheck.require_seal()
    return sealcheck.run_sealed(ITEM, "repro" if a.repro else "run", tag, lambda: body(a, tag, B))


def body(a, tag: str, B: int) -> int:
    from r5 import sealcheck
    t00 = time.time()
    mode = "smoke" if a.smoke else ("repro" if a.repro else "run")
    n_chk = a.ect_check_seeds if a.ect_check_seeds is not None else (1 if a.smoke else len(paths.DEV_SEEDS))
    n_iv = a.intervention_seeds if a.intervention_seeds is not None else (1 if a.smoke else len(paths.DEV_SEEDS))
    designs = ("BPG", "BP") if a.design == "both" else (a.design,)
    corpus = ECT.Corpus()
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    y_mode = "surrogate" if a.smoke else "labels"
    status = 0
    for design in designs:
        t0 = time.time()
        secs = {}
        builds, checks = {}, {}
        for pool in M3.POOLS:
            R, KD, chk = M3.build(corpus, design, pool=pool, y_mode=y_mode, check_seeds=paths.DEV_SEEDS[:n_chk])
            builds[pool], checks[pool] = (R, KD), chk
        secs["build"] = round(time.time() - t0, 1)
        if a.smoke:
            for pool in list(builds):
                R, KD = builds[pool]
                checks[pool]["uniform_dev_real_templates"] = M3.kernel_predict(KD.S, KD.mask, KD.b1, KD.da, KD.dr,
                                                                               KD.sib, KD.sh)[3]
                builds[pool] = M3.surrogate(R, KD, design, pool)
        for pool, (R, KD) in builds.items():
            checks[pool]["uniform_dev"] = M3.add_config_columns(R, KD)
        if not a.smoke:                       # brief 1.2 / 6.0: the reproduction target first; a mismatch stops the item
            rep = M3.reproduction(builds, checks, design)
            rep.update({"seal": tag, "started_utc": started, "code_lf_sha256": code_digests()})
            write_json(OUT / f"repro_{design}.json", rep)
            print(f"TECHNICAL SMOKE reproduction {design}: {'MATCH' if rep['passed'] else 'MISMATCH (item stops)'}")
            for k, v in rep["targets"].items():
                print(f"  {k}: counts {v['counts_found']} ok {v['counts_ok']}; values {v['values_found']}"
                      + (f" ok {v['values_ok']}" if "values_ok" in v else ""))
            for pool, c in checks.items():
                print(f"  {pool}: B1 max diff {c['b1_maxabs']:.1e}; y max diff {c['y_maxabs']}; read-set mismatches "
                      f"{c['readset_mismatch']}; r5.ect identity {c['ect_identity_maxabs']} on {c['ect_identity_groups']} "
                      f"groups; uniform limit deviation {c['uniform_dev']:.1e}")
            if a.repro:
                status |= 0 if rep["passed"] else 1
                continue
            if not rep["passed"]:
                sealcheck.stop(ITEM, mode, tag, f"reproduction mismatch ({design}; results/ECT_M03/repro_{design}.json) "
                                                "-- item stops (brief 6.0)")
        t1 = time.time()
        iv = {pool: M3.intervention(corpus, design, pool=pool, seeds=paths.DEV_SEEDS[:n_iv]) for pool in builds}
        secs["intervention"] = round(time.time() - t1, 1)
        if not a.smoke and not all(v["passed"] for v in iv.values()):
            write_json(LEAK / f"ECT_M03_{design}.json", {"label": "LEAKAGE CHECK (section 8 item 4) FAILED", "seal": tag,
                                                          "intervention": iv, "code_lf_sha256": code_digests()})
            sealcheck.stop(ITEM, mode, tag, f"leakage check failed ({design}: intervention; "
                                            f"results/leakage/ECT_M03_{design}.json): item stopped")
        t1 = time.time()
        T = M3.analysis(builds, corpus, design, B=B)
        secs["analysis"] = round(time.time() - t1, 1)
        secs["nulls"] = {f"{p}|{k}": round(v["seconds"], 1) for (p, k), v in T["_nullobs"].items()}
        secs["per_permutation"] = round(sum(v["seconds"] for v in T["_nullobs"].values()) / max(1, B) /
                                        max(1, len(T["_nullobs"])), 3)
        t1 = time.time()
        fp = M3.fit_poisoning(T["_frames"], design)
        secs["fit_poisoning"] = round(time.time() - t1, 1)
        secs["total"] = round(time.time() - t0, 1)
        leak_ok = all(v["passed"] for v in iv.values()) and fp["passed"]
        if a.smoke:
            dry = emit(design, builds, checks, T, iv, fp, tag=tag, started=started, B=B, secs=secs, t0=t0, dry=True)
            counts = smoke_counts(builds, checks, T, iv, fp, secs)
            counts["projected_null_seconds_at_B500"] = round(secs["per_permutation"] * 500 * max(1, len(T["_nullobs"])), 0)
            counts["dry_run_outputs"] = dry
            write_json(OUT / f"SMOKE_COUNTS_{design}.json", counts)
            for pool, (R, KD) in builds.items():
                c = checks[pool]
                print(f"  {design} {pool}: rows {R.shape}; groups {len(KD.S)}; kernel tensor {KD.S.shape}; B1 max diff "
                      f"{c['b1_maxabs']:.1e}; y max diff {c['y_maxabs']}; read-set mismatches {c['readset_mismatch']}; "
                      f"r5.ect identity max {max(c['ect_identity_maxabs'].values()) if c['ect_identity_maxabs'] else 'n/a'}"
                      f" on {c['ect_identity_groups']} groups; uniform-limit deviation (pre-override) {c['uniform_dev']:.1e}")
            for k, v in T.items():
                if isinstance(v, pd.DataFrame):
                    print(f"    table {k}: {v.shape}")
            print(f"  dry-run outputs (serialised in memory, not written): {len(dry)} files, "
                  f"{sum(dry.values())} bytes; intervention checks ok "
                  f"{ {p: v['checks_ok'] for p, v in iv.items()} }; fit poisoning passed {fp['passed']}; timings {secs}")
            continue
        if not leak_ok:                       # s0-mechanics-17: no scored write after a failed leakage check
            write_json(LEAK / f"ECT_M03_{design}.json", {"label": "LEAKAGE CHECK (section 8 items 2, 4) FAILED",
                                                          "seal": tag, "intervention": iv, "fit_poisoning": fp,
                                                          "code_lf_sha256": code_digests()})
            sealcheck.stop(ITEM, mode, tag, f"leakage check failed ({design}: fit poisoning; "
                                            f"results/leakage/ECT_M03_{design}.json): item stopped")
        emit(design, builds, checks, T, iv, fp, tag=tag, started=started, B=B, secs=secs, t0=t0, dry=False)
    print(f"done {time.time() - t00:.0f}s (status {status})")
    return status


def emit(design, builds, checks, T, iv, fp, *, tag, started, B, secs, t0, dry: bool) -> dict:
    """Every output of the full run.  ``dry`` (technical smoke) serialises each file in memory only and returns
    {name: bytes}, so the writing path is exercised without writing any surrogate table."""
    sizes = {}

    def put(name: str, data: bytes, path: Path) -> None:
        sizes[name] = len(data)
        if not dry:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print("wrote", path)

    def parquet(df: pd.DataFrame, name: str) -> None:
        import io
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        put(name, buf.getvalue(), OUT / name)

    def csv(df: pd.DataFrame, name: str, base: Path = OUT) -> None:
        put(name, df.to_csv(index=False, lineterminator="\n").encode("utf-8"), base / name)

    def js(obj, name: str, base: Path = OUT) -> None:
        put(name, (json.dumps(obj, indent=1, default=_js) + "\n").encode("utf-8"), base / name)

    leak_ok = all(v["passed"] for v in iv.values()) and fp["passed"]
    for pool, (R, KD) in builds.items():
        R = R.copy()
        R["label"] = np.where(R.in_PRIMARY, M3.SEEN, M3.SENS) if pool == "bit" else M3.SENS
        R["note"] = M3.NOT_NEXT_TO
        R["seal"] = tag
        parquet(R, f"rows_{pool}_{design}.parquet")
    NL = M3.nested_long(T["_frames"])
    NL["label"] = [M3.rule_label(M3.SEEN if p == "PRIMARY" else M3.SENS, a) for p, a in zip(NL.population, NL.arm)]
    parquet(NL, f"nested_rows_{design}.parquet")
    for k, v in T.items():
        if k.startswith("_") or not isinstance(v, pd.DataFrame):
            continue
        if k == "null_draws":
            v = v.assign(label=f"{M3.DIAG} (selection-aware null replicate)")
        csv(v, f"{k}_{design}.csv")
    cfg = M3.config_table()
    cfg["label"] = "FROZEN GRID (registry; no score)"
    csv(cfg, "grid_definition.csv")
    leak = {"label": "LEAKAGE CHECK (section 8 items 2, 4, 5)", "seal": tag, "intervention": iv, "fit_poisoning": fp,
            "support": {p: {k: c[k] for k in ("b1_maxabs", "y_maxabs", "readset_mismatch", "both_ends_read_rows",
                                              "ect_identity_maxabs", "ect_identity_groups", "uniform_dev")}
                        for p, c in checks.items()},
            "cache_poisoning": "not applicable: no new cache (read sequences from the gen21 fold cache)",
            "code_lf_sha256": code_digests()}
    js(leak, f"ECT_M03_{design}.json", LEAK)
    lk = leakage_line(iv, fp, checks)
    item = M3.per_item(T, design, seal_tag=tag, leakage=lk, seconds=time.time() - t0)
    txt = M3.per_item_markdown(item, title=f"1.2 M03 sibling-weighting factorial ({design}): per-item table")
    put(f"PER_ITEM_TABLE_{design}.md", txt.encode("utf-8"), OUT / f"PER_ITEM_TABLE_{design}.md")
    js({"label": M3.SEEN, "note": M3.NOT_NEXT_TO, "seal": tag, "started_utc": started, "design": design,
        "permutations": B, "builder_checks": checks, "leakage_passed": leak_ok,
        "tables": sorted(k for k, v in T.items() if not k.startswith("_") and isinstance(v, pd.DataFrame)),
        "rules": T["rules"].to_dict(orient="records"), "code_lf_sha256": code_digests(), "seconds": secs},
       f"summary_{design}.json")
    return sizes


if __name__ == "__main__":
    raise SystemExit(main())
