#!/usr/bin/env python
"""Stage-1 item 1.3, E-LAB: campaign calibration of a new extractant (lab-1; the E-CT annex), CLAUDE_CODE_PROMPT.md
section 6.3 (conventions 6.0, 6.10, 8, 9).  Code: r5/elab.py.  Spec: results/ELAB/ITEM_SPEC.md.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_elab.py \
        [--design BPG|BP|both] [--smoke [--iv-seeds N] | --repro]

Modes (run from the repository root):
  default   requires the round-5 seal (r5.sealcheck.require_seal); runs the reproduction target first (a mismatch stops
            the item) and the leakage checks before any scored write (a failure stops the item); builds every arm,
            comparator, control, stratum and
            sensitivity of section 6.3 on the round-4 REF rows of the POPX targets (r5.elab: MX_LM10, B1, the E_PBC /
            E_PBA / E_PCM / E_PCMC placebos, E_RM, the global-K grid with its oracle and LOPO choices, same- vs
            other-series donors, the low-D band split, the duplicate sensitivities, C5-BTBP in / out), the unit tables
            (publication, laboratory conservative / liberal, source cluster), LOPO, leave-two-publications-out, the
            top-3 share and the report-rule readings; runs the section-8 prediction intervention with its positive
            controls and the fit-poisoning check of the LOPO K choice; writes row-level outputs and tables under
            results/ELAB/ (every number labelled, section 9.1) and the leakage report under results/leakage/.
            EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression.  Never next to 0.452.
  --repro   requires the seal; the item's section-6.0 reproduction targets only (B1 from the fold betas within 8.9e-16
            on every REF row; POPX 92 cells / 16 publications / 787 rows per seed and POPX0 60 / 13 / 534; MX_LM10 of
            the published pool against ideas2/lab_out/lab_rows2.parquet, maximum difference 0.0 over 7,870 rows; on BPG
            B1 0.444 and MX_LM10 0.494), labelled TECHNICAL SMOKE, written to results/ELAB/repro_<design>.json; exit
            code 1 on a mismatch (a mismatch stops the item).
  --smoke   no seal needed; PRE-SEAL ENGINEERING CHECK with every metric suppressed: the rows and every arm are built on real
            data without any expected error (only identity checks of the round-4 parquet, counts, NaN patterns and
            timings), then the whole pipeline (arms, tables, per-item table, intervention, poisoning check) runs on a
            seeded noise surrogate of V1 (NaN pattern kept); only shapes, counts, check flags and timings are printed
            and written (results/ELAB/SMOKE_COUNTS_<design>.json).  No R2, SSE, MAE or contrast value is computed on
            real labels.
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
from r5 import elab as EL  # noqa: E402

paths.redirect_inner_cache()
OUT = paths.RES / "ELAB"
LEAK = paths.RES / "leakage"
CODE = ("r5/elab.py", "r5/common.py", "r5/dupes.py", "r5/population.py", "r5/supply.py", "scripts/s1_elab.py")


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


def smoke_counts(C, R: pd.DataFrame, ck: dict, pools: pd.DataFrame, S: pd.DataFrame, ck_s: dict, T: dict, iv: dict,
                 gp: dict, seconds: dict) -> dict:
    """Counts only: shapes, NaN patterns, pool-size distributions (label-free or duplicate-rule counts), identity
    checks of the round-4 parquet, population counts per stratum, table shapes on the surrogate, check flags, timings."""
    num = [c for c in R.columns if R[c].dtype.kind == "f" and c.split("@")[0] in EL.POINT_ARMS]
    pop = EL.t_populations(R, ck["design"])[["c5", "stratum", "cells", "publications", "labs_cons", "labs_lib", "clusters",
                                             "rows_per_seed", "cells_with_nearcopy_donor", "rows_prim_empty"]]

    def dist(col):
        return pools[col].describe()[["min", "50%", "max"]].round(2).to_dict() if len(pools) else {}
    return {"label": "PRE-SEAL ENGINEERING CHECK (--smoke: every metric suppressed; tables, intervention and poisoning run on a noise surrogate)",
            "design": ck["design"], "rows_shape": list(R.shape), "identity_checks_on_real_data": ck,
            "static": {"bench_cells": C.N, "c3_targets": len(C.c3), "popx_src_cells": len(C.popx_src),
                       "primary_cells": len(C.primary), "bit_links": len(C.bit), "nearid_links": len(C.nearid),
                       "offset_links": len(C.offset), "metal_conc_groups": len(C.mgroups),
                       "lab_key_missing_publications": sorted(C.static[C.static.lab_key_missing & C.static.cell_id.isin(C.popx_src)].publication.unique()),
                       "scope_in_primary_cells": int(C.static[C.static.cell_id.isin(C.primary)].scope_in.sum())},
            "nan_share_point_arms": {c: round(float(R[c].isna().mean()), 4) for c in num},
            "draw_nan_total": int(np.nansum([R[c].sum() for c in R.columns if c.startswith("ndraw_nan_")])),
            "pool_sizes": {c: dist(c) for c in ("n_src", "n_clean", "n_prim", "n_nc", "n_ss", "n_os", "n_xpub", "n_xpub_fam",
                                                "unread", "unread_no_match")},
            "target_seeds_with_nearcopy": int((pools.n_nc > 0).sum()) if len(pools) else 0,
            "cells_with_nearcopy_donor_clean_pool": int(pools[pools.n_nc > 0].cell_id.nunique()) if len(pools) else 0,
            "cells_with_nearcopy_donor_src_pool": int(pools[pools.n_nc_src > 0].cell_id.nunique()) if len(pools) else 0,
            "target_seeds_prim_empty": int((pools.n_prim == 0).sum()) if len(pools) else 0,
            "populations_counts": pop.to_dict(orient="records"),
            "surrogate": {"rows_shape": list(S.shape), "identity_checks": {k: ck_s[k] for k in
                          ("b1_predict_levels_maxabs", "readset_mismatch_groups", "fold_mismatch_groups", "metals_mismatch_groups",
                           "both_ends_read_rows", "adjacent_lighter_first", "finite_on_primary")},
                          "table_shapes": {k: list(v.shape) for k, v in T.items()}, "intervention": iv,
                          "rules_not_available": [r.rule for r in T["rules"].itertuples(index=False)
                                                  if "NOT AVAILABLE" in str(r.in_corpus_value)],
                          "rules_with_nan_on_surrogate (top-3 share undefined when the noise gain <= 0)": [
                              r.rule for r in T["rules"].itertuples(index=False) if "nan" in str(r.in_corpus_value)],
                          "gk_poison_check": gp},
            "seconds": {k: round(v, 1) for k, v in seconds.items()}}


ITEM = "ELAB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=("BPG", "BP", "both"), default="both")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="technical smoke, every metric suppressed (no seal needed)")
    g.add_argument("--repro", action="store_true", help="the section-6.0 reproduction targets only (TECHNICAL SMOKE)")
    ap.add_argument("--iv-seeds", type=int, default=None,
                    help="dev seeds covered by the intervention (--smoke only; the registered run covers all 5)")
    a = ap.parse_args(argv)
    from r5 import sealcheck
    sealcheck.forbid_reduced_coverage(a.smoke, iv_seeds=a.iv_seeds)
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
    nseeds = a.iv_seeds if a.iv_seeds is not None else (1 if a.smoke else len(paths.DEV_SEEDS))
    iv_seeds = tuple(paths.DEV_SEEDS[:max(1, min(nseeds, len(paths.DEV_SEEDS)))])
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    C = EL.Campaign()
    status = 0
    for design in designs:
        t1 = time.time()
        if not a.smoke:                       # brief 1.2 / 6.0: the reproduction target first; a mismatch stops the item
            Rr, ckr, _ = C.build(design, mode="repro")
            rep = EL.reproduction(C, Rr, ckr, design)
            write_json(OUT / f"repro_{design}.json", {**rep, "seal": tag, "started_utc": started,
                                                      "identity_checks": ckr, "code_lf_sha256": code_digests()})
            print(f"TECHNICAL SMOKE reproduction {design}: {'MATCH' if rep['passed'] else 'MISMATCH (item stops)'}; "
                  f"targets {sum(rep['targets'].values())}/{len(rep['targets'])}; B1 identity "
                  f"{ckr['b1_parquet_maxabs']:.1e}; rows compared {rep['checks']['rows_compared']}")
            del Rr
            if a.repro:
                status |= 0 if rep["passed"] else 1
                continue
            if not rep["passed"]:
                sealcheck.stop(ITEM, mode, tag, f"reproduction mismatch ({design}; results/ELAB/repro_{design}.json) -- "
                                                "item stops (brief 6.0)")
        if a.smoke:
            sec = {}
            R, ck, pools = C.build(design, score=False)
            sec["build_real_no_score"] = time.time() - t1
            t2 = time.time()
            Ys = EL.surrogate_Y(C.Y, f"ELAB|SMOKE_SURROGATE|{design}")
            S, ck_s, _ = C.build(design, Yv=Ys)
            sec["build_surrogate"] = time.time() - t2
            t2 = time.time()
            T = EL.all_tables(S, design)
            sec["tables_surrogate"] = time.time() - t2
            txt = CM.per_item_table(EL.per_item(T, design, seal_tag=tag, leakage="smoke", seconds=0.0), None)
            print(f"  per-item table exercised on the surrogate: {len(txt.splitlines())} lines (not written)")
            t2 = time.time()
            iv = C.intervention(design, Yv=Ys, seeds=iv_seeds)
            sec["intervention_surrogate"] = time.time() - t2
            gp = EL.gk_poison_check(S)
            counts = smoke_counts(C, R, ck, pools, S, ck_s, T, iv, gp, sec)
            write_json(OUT / f"SMOKE_COUNTS_{design}.json", counts)
            print(f"  {design}: real rows {R.shape}, B1 identity {ck['b1_parquet_maxabs']:.1e} (passed "
                  f"{ck['b1_identity_passed']}), predict_levels {ck['b1_predict_levels_maxabs']:.1e}; surrogate tables "
                  f"{len(T)}; intervention checks ok {iv['checks_ok']} (coverage full {iv['coverage_full']}); "
                  f"K-poisoning passed {gp['passed']}; {time.time() - t1:.0f}s")
            for k, v in T.items():
                print(f"    table {k}: {v.shape}")
            continue
        R, ck, pools = C.build(design)
        iv = C.intervention(design, seeds=iv_seeds)
        gp = EL.gk_poison_check(R)
        # brief 8 / s0-mechanics-17: the leakage checks run before any scored write; a failure stops the item
        leak = {"label": "LEAKAGE CHECK (section 8 items 1-5)", "seal": tag, "design": design, "intervention": iv,
                "intervention_seeds": list(iv_seeds), "fit_poisoning_lopo_K": gp,
                "isolation": "not applicable by design: E-LAB's donors are the campaign's other series (allowed by its "
                             "information contract; brief 8); the B1 base beta is the fold's round-4 fit (round-4 "
                             "isolation passed; r5.common.load_rows checks the REF intervention / identity records)",
                "support": {k: ck[k] for k in ("both_ends_read_rows", "adjacent_lighter_first", "readset_mismatch_groups",
                                               "fold_mismatch_groups", "metals_mismatch_groups", "b1_parquet_maxabs",
                                               "b1_predict_levels_maxabs", "y_maxabs", "finite_on_primary")},
                "cache_poisoning": "not applicable: no new cache (fold records read with g21.cache.load)",
                "code_lf_sha256": code_digests()}
        write_json(LEAK / f"ELAB_{design}.json", leak)
        if not (iv["passed"] and gp["passed"] and ck["b1_identity_passed"]):
            sealcheck.stop(ITEM, mode, tag, f"leakage / identity check failed ({design}; results/leakage/ELAB_{design}.json):"
                                            " item stopped")
        T = EL.all_tables(R, design)
        R["label"] = EL.rows_label(design)
        R["seal"] = tag
        R.to_parquet(OUT / f"rows_{design}.parquet", index=False)
        print("wrote", OUT / f"rows_{design}.parquet")
        write_csv(OUT / f"pools_{design}.csv", pools.assign(label="counts (no score)"))
        for k, v in T.items():
            write_csv(OUT / f"{k}_{design}.csv", v)
        lk = (f"intervention {'passed' if iv['passed'] else 'FAILED'} on {iv['target_seeds']} target-seeds "
              f"({iv['targets']} targets); positive controls self {iv['pc_self_caught']}/{iv['pc_self_applicable']}, "
              f"bit copy {iv['pc_bit_caught']}/{iv['pc_bit_applicable']}, leak {iv['pc_leak_caught']}/{iv['pc_leak_applicable']}; "
              f"label-based sensitivities changed {iv['label_based_changed']} times (declared); LOPO-K poisoning "
              f"{'passed' if gp['passed'] else 'FAILED'} (leaky control caught {gp['pc_caught']}/{gp['pc_runs']}); B1 "
              f"identity {ck['b1_parquet_maxabs']:.1e}")
        item = EL.per_item(T, design, seal_tag=tag, leakage=lk, seconds=time.time() - t1)
        CM.per_item_table(item, OUT / f"PER_ITEM_TABLE_{design}.md", title=f"1.3 E-LAB ({design}): per-item table")
        print("wrote", OUT / f"PER_ITEM_TABLE_{design}.md")
        write_json(OUT / f"summary_{design}.json", {
            "label": EL.label(design=design, c5=EL.PRIMARY_C5, stratum="PRIMARY"), "note": EL.NOT_NEXT_TO, "seal": tag,
            "started_utc": started,
            "design": design, "primary_design": EL.PRIMARY_DESIGN, "primary_c5": EL.PRIMARY_C5,
            "primary_unit": EL.PRIMARY_UNIT, "identity_checks": ck,
            "restated_targets": EL.restated_targets(T, design), "report_rules": T["rules"].to_dict(orient="records"),
            "seen_values": EL.SEEN_VALUES, "external_rule": EL.EXTERNAL_RULE, "intervention_passed": iv["passed"],
            "gk_poison_passed": gp["passed"], "tables": sorted(T), "code_lf_sha256": code_digests(),
            "seconds": round(time.time() - t1, 1)})
    print(f"done {time.time() - t00:.0f}s (status {status})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
