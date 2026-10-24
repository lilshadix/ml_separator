#!/usr/bin/env python
"""Stage-1 item 1.0, EA_CHECK: the M35 positive-control and six-design check (CLAUDE_CODE_PROMPT.md sections 4.1,
6.0, 6.10 row 1.0, 8, 9; r5/ea_check.py).  Estimand E-A, DIAGNOSTIC, every number EXPLORATORY (seen); no E-A slot.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py --build-cache
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py --smoke
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py --repro
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py [--design BPG|BP]

--build-cache  builds (or verifies) the A, B, BR, BQ fold records (g21.cache.build_fold, 4 designs x 5 dev seeds x 5
               folds) into cache/<design>_<seed>_<fold>.pkl, runs the cache-poisoning and isolation checks, writes
               manifests/ea_check_caches.json and results/leakage/cache_poison_EA_CHECK.json.  Computes no score, so it
               runs before the seal.
--smoke        PRE-SEAL ENGINEERING CHECK on real data with every metric suppressed: shapes, row / cell / unit counts, NaN patterns,
               identity booleans (reproduction on one fold, isolation, intervention, peek control, fixed denominators)
               and timings; every table function is exercised on the real row structure with y replaced by seeded
               random values and only the table shapes are printed.  Writes nothing.  Needs no seal.
--repro        the item's reproduction target of section 6.0 only (B1 from the fold betas, B0 and HIST_D7 recomputed on
               the gen21 caches = REF parquet within 8.9e-16 on all 18,020 rows), labelled TECHNICAL SMOKE; writes
               results/EA_CHECK/repro.json.  Requires the seal.
(default)      requires the seal (r5.sealcheck.require_seal); runs the reproduction target first (a mismatch stops the
               item), then every arm, comparator, stratum, sensitivity and table of the item; --design restricts the E-A
               contrasts to one design (BPG decides; BP is a sensitivity).  Writes results/EA_CHECK/* (with the
               section-9.3 PER_ITEM_TABLE_<design>.md of every scored design, and ITEM_TABLE.md when both ran),
               results/leakage/intervention_EA_CHECK.json; START / END lines in RUN_RECORD.md (the END line carries
               the brief-8 post-run seal check, r5.sealcheck.post_run).  Cache poisoning is read from the sealed
               manifest manifests/ea_check_caches.json (every record poison-identical).
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
from r5 import ea_check as EA  # noqa: E402
from r5 import sealcheck  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

OUT = EA.OUT
SMOKE_SEED_KEY = ("smoke", "fake_y")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def record(line: str) -> None:
    with open(paths.R5 / "RUN_RECORD.md", "ab") as fh:
        fh.write((line.rstrip() + "\n").encode("utf-8"))


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def write_json(path: Path, obj, wrote: list) -> None:
    paths.write_text_lf(path, json.dumps(obj, indent=1, default=_default) + "\n")
    wrote.append(path)


def write_csv(path: Path, df: pd.DataFrame, wrote: list) -> None:
    paths.write_text_lf(path, df.to_csv(index=False, lineterminator="\n"))
    wrote.append(path)


def rel(p: Path) -> str:
    try:
        return p.relative_to(paths.ROOT).as_posix()
    except ValueError:
        return str(p)


# ------------------------------------------------------------------------------------------------------------
# --build-cache (pre-seal; no score)
# ------------------------------------------------------------------------------------------------------------
def build_cache(a) -> int:
    t0 = time.time()
    bench = D.load_bench()
    designs = tuple(a.cache_designs.split(",")) if a.cache_designs else EA.NEW_DESIGNS
    record(f"- {now()} START EA_CHECK cache build ({', '.join(designs)}; g21.cache.build_fold + poisoning; no score)")
    man = EA.build_caches(bench, designs, force=a.force, log=lambda s: print(s, flush=True))
    s = man["summary"]
    ok = s["poison_identical"] == s["n_records"] and s["isolation_passed"] == s["n_records"]
    record(f"- {now()} END EA_CHECK cache build: {s['n_records']} records, poisoning identical {s['poison_identical']}, "
           f"isolation passed {s['isolation_passed']}, build {s['build_seconds_total']}s + poison "
           f"{s['poison_seconds_total']}s, wall {time.time() - t0:.0f}s")
    print(json.dumps(s, indent=1))
    print("wrote", rel(EA.MANIFEST), ",", rel(EA.POISON_REPORT), f"and {s['n_records']} pickles under",
          rel(paths.CACHE), "; appended 2 lines to", rel(paths.R5 / "RUN_RECORD.md"))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------
# --smoke (pre-seal; every metric suppressed)
# ------------------------------------------------------------------------------------------------------------
def _nan_pattern(t: pd.DataFrame, cols) -> str:
    return ", ".join(f"{c} {int(t[c].isna().sum())}" for c in cols if c in t)


def _fake_y(rows: dict) -> dict:
    """y replaced by B0 + N(0, 0.15) keyed on (seed, cell, A, B), identical across designs (smoke only)."""
    key = ["split_seed", "cell_id", "A", "B"]
    base = rows["BPG"][key + ["B0"]].copy()
    rng = np.random.default_rng(EA.rng_seed(*SMOKE_SEED_KEY))
    base["y_fake"] = base.B0.to_numpy(float) + rng.normal(0.0, 0.15, len(base))
    out = {}
    for d, t in rows.items():
        m = t.merge(base[key + ["y_fake"]], on=key, how="left", validate="one_to_one")
        out[d] = m.assign(y=m.y_fake).drop(columns="y_fake")
    return out


def smoke(a) -> int:
    t0 = time.time()
    print("PRE-SEAL ENGINEERING CHECK (--smoke; every metric suppressed; nothing written)")
    bench = D.load_bench()
    prov = EA.provenance_cells()
    t_bpg = EA.ea_rows("BPG")
    cmap = EA.cluster_map(t_bpg)
    rows = {"BPG": t_bpg, "BP": EA.ea_rows("BP", cmap)}
    for d, t in rows.items():
        c5 = CM.c5_split(t)
        bf = EA.add_big_fold(t)
        print(f"  {d}: rows {t.shape}, seeds {t.split_seed.nunique()}, cells {t.cell_id.nunique()}, rows/seed "
              f"{len(t) / t.split_seed.nunique():.0f}, cpg units {t.cluster_cpg.nunique()}, publications "
              f"{t.publication.nunique()}; NaN: {_nan_pattern(t, ['y', 'B0', 'B1', 'HIST_D7'])}; C5-out rows/seed "
              f"{len(c5['C5_out']) / 5:.0f}, C5-cell-out rows/seed {len(c5['C5_cell_out']) / 5:.0f}; big-fold rows/seed "
              f"{bf.big_fold.sum() / 5:.0f}")
    pu = EA.unit_provenance(t_bpg, prov)
    print(f"  provenance: {len(pu)} units, ORNL-only units {int(pu.ornl_only.sum())}, cells with a provenance row "
          f"{int(prov.index.isin(t_bpg.cell_id.unique()).sum())}/{t_bpg.cell_id.nunique()}, classes "
          f"{prov.loc[t_bpg.cell_id.unique()].loc_class.value_counts().to_dict()}")
    t1 = time.time()
    first = {(folds_for(bench.frame, "BPG", seeds=list(paths.DEV_SEEDS))[0].seed, 0)}
    rp = EA.repro(bench, fold_filter=first)
    print(f"  reproduction (one fold per design): " + "; ".join(
        f"{d} rows {v['rows']} keys {v['same_keys']} y {v.get('same_y')} read sets {v.get('same_read_set')} "
        f"B1/B0/HIST_D7 within {EA.REPRO_TOL:g}: {v['passed']}" for d, v in rp["designs"].items())
          + f" ({time.time() - t1:.1f}s)")
    probs = EA.cache_problems(bench)
    print(f"  round-5 caches: {100 - len(probs)}/100 usable" + (f"; first problems {probs[:3]}" if probs else ""))
    have_all = not probs
    if have_all:
        man = EA.load_manifest()
        for d in EA.NEW_DESIGNS:
            t2 = time.time()
            fl = {(f.seed, f.fold) for f in folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS))[:a.smoke_folds]}
            td, iso, iv = EA.design_rows(bench, d, cmap, man, fold_filter=fl)
            print(f"  {d} ({len(fl)} fold(s)): rows {td.shape}, cells {td.cell_id.nunique()}; NaN: "
                  f"{_nan_pattern(td, ['y', 'B0', 'HIST_D7'])}; isolation {all(x['passed'] for x in iso)}; intervention "
                  f"{all(x['passed'] for x in iv)} ({sum(x['n_predictions'] for x in iv)} predictions), peek caught "
                  f"{all(x['peek_caught'] for x in iv)}, support {all(x['support_ok'] for x in iv)} "
                  f"({time.time() - t2:.1f}s)")
        t3 = time.time()
        for d in EA.NEW_DESIGNS:
            rows[d], _, _ = EA.design_rows(bench, d, cmap, man)
        same = EA.assert_same_queries(rows)
        print(f"  all six designs built ({time.time() - t3:.1f}s): fixed denominators " +
              ", ".join(f"{d} {v['rows_per_seed']:.0f}/seed" for d, v in same.items()))
    st = EA.registry_status("M08")
    print(f"  REGISTRY M08 status: {st['status']}" + (f" ({st.get('reason')})" if st.get("reason") else ""))
    t4 = time.time()
    fake = _fake_y(rows)
    for d in ("BPG", "BP"):
        res = EA.score_design(fake[d], prov, d, m08=True)
        print(f"  pipeline on fake y, {d}: " + ", ".join(f"{k} {v.shape}" for k, v in res.items()))
    if have_all:
        six = EA.six_design(fake, prov)
        print("  pipeline on fake y, six-design: " + ", ".join(f"{k} {v.shape}" for k, v in six.items()))
    else:
        print("  six-design pipeline skipped (round-5 caches incomplete)")
    print(f"  pipeline exercise {time.time() - t4:.1f}s; total {time.time() - t0:.1f}s; wrote nothing")
    return 0


# ------------------------------------------------------------------------------------------------------------
# scoring (after the seal)
# ------------------------------------------------------------------------------------------------------------
def run_repro(bench, tag: str, wrote: list) -> dict:
    rp = EA.repro(bench, log=print)
    rp["seal"] = tag
    write_json(OUT / "repro.json", rp, wrote)
    print(f"  TECHNICAL SMOKE reproduction: {rp['rows']} rows, passed {rp['passed']}: " +
          "; ".join(f"{d} max|dB1| {v.get('max_abs_diff_B1')}, max|dB0| {v.get('max_abs_diff_B0')}, "
                    f"max|dHIST_D7| {v.get('max_abs_diff_HIST_D7')}" for d, v in rp["designs"].items()))
    return rp


def item_table(res: dict, designs, leak: dict, compute: str, decision: dict, tag: str) -> dict:
    """The section-9.3 item of ``designs`` (one: PER_ITEM_TABLE_<design>.md; BPG and BP: ITEM_TABLE.md)."""
    return EA.per_item({d: res[d] for d in designs}, pop=CM.population(res[designs[0]]["rows"]), leakage=leak["text"],
                       compute=compute, decision=decision, seal_tag=tag)


def leakage_text(leak: dict) -> str:
    pois = leak["cache_poisoning"]
    return (f"cache poisoning {pois['n_folds']} folds identical: {pois['passed']}; isolation "
            f"{leak['isolation_passed']}/{leak['isolation_folds']} (design-aware); intervention "
            f"{leak['intervention_passed']}/{leak['intervention_folds']} folds, max change "
            f"{leak['intervention_max_abs_change']:g} over {leak['intervention_predictions']} predictions; "
            f"block['Y'] peek caught in {leak['peek_caught']}/{leak['intervention_folds']}; assert_test_isolation on "
            "the 50 BP / BPG folds (repro); fixed denominators across the six designs")


def make_decision(gates: pd.DataFrame) -> dict:
    """The recorded M35 outcomes (summary decision): C1 on BPG C5_in decides; BP is a sensitivity."""
    def pick(d, c):
        g = gates[(gates.design == d) & (gates.stratum == "C5_in") & (gates.contrast == c)]
        return None if g.empty else {k: g.iloc[0][k] for k in ("relSSE", "SF_POOL_p", "dMAE_bal", "dR2", "G", "gate_pass",
                                                                "gate_relSSE_ge_0.02", "gate_SF_POOL_p_le_0.025",
                                                                "guard_dMAE_bal_ge_-0.001", "label")}
    decision = {"rule": "pass = relSSE >= 0.02 AND SF_POOL p <= 0.025 (exact, cluster_cpg) AND dMAE_bal >= -0.001; "
                        "BPG decides; a pass is DIAGNOSTIC (M34's kill fired)",
                "positive_control": pick("BPG", "B1 - B0"), "HIST_D7_minus_B1": pick("BPG", "HIST_D7 - B1"),
                "BP_sensitivity": {"B1 - B0": pick("BP", "B1 - B0"), "HIST_D7 - B1": pick("BP", "HIST_D7 - B1")}}
    if decision["positive_control"] is not None:
        decision["M35_passes_its_positive_control"] = bool(decision["positive_control"]["gate_pass"])
        decision["M35_kill_fires"] = not decision["M35_passes_its_positive_control"]
    return decision


def run_detail(decision: dict) -> str:
    """The positive-control words of the END line: 'not evaluated' when BPG was not scored (never a FAIL)."""
    if "M35_passes_its_positive_control" in decision:
        s = f"positive control {'PASS' if decision['M35_passes_its_positive_control'] else 'FAIL'} on BPG"
    else:
        s = "positive control not evaluated in this run (BPG not scored)"
    bp = decision["BP_sensitivity"]["B1 - B0"]
    if bp is not None:
        s += f"; BP sensitivity B1 - B0 {'PASS' if bp['gate_pass'] else 'FAIL'}"
    return s


def sealed_poisoning(man: dict) -> dict:
    """Cache poisoning read from the sealed manifest (s0-mechanics-10): every one of the 100 records poison-identical."""
    s = man["summary"]
    ok = bool(s["poison_identical"] == s["n_records"] == 100
              and all(r["poison"]["identical"] for r in man["records"].values()))
    return {"n_folds": int(s["n_records"]), "passed": ok, "source": "manifests/ea_check_caches.json (sealed)"}


def run(a, tag: str, bench):
    from r5 import sealcheck
    t0 = time.time()
    wrote = []
    designs = (a.design,) if a.design else paths.DESIGNS
    OUT.mkdir(parents=True, exist_ok=True)
    rp = run_repro(bench, tag, wrote)
    if not rp["passed"]:
        sealcheck.stop("EA_CHECK", "run", tag, "reproduction target mismatch (results/EA_CHECK/repro.json): item 1.0 "
                                               "stopped (section 6.0); fix by amendment and reseal")
    probs = EA.cache_problems(bench)
    if probs:
        sealcheck.stop("EA_CHECK", "run", tag, f"round-5 caches not usable ({len(probs)} problems; {probs[:3]}): run "
                                               "--build-cache")
    man = EA.load_manifest()
    prov = EA.provenance_cells()
    t_bpg = EA.ea_rows("BPG")
    cmap = EA.cluster_map(t_bpg)
    rows = {"BPG": t_bpg, "BP": EA.ea_rows("BP", cmap)}
    iso_all, iv_all = [], []
    t1 = time.time()
    for d in EA.NEW_DESIGNS:
        rows[d], iso, iv = EA.design_rows(bench, d, cmap, man)
        iso_all += iso
        iv_all += iv
        print(f"  {d}: {len(rows[d])} rows", flush=True)
    t_rows = time.time() - t1
    same = EA.assert_same_queries(rows)
    pois = sealed_poisoning(man)
    leak = {"label": "leakage checks (no score)", "seal": tag, "isolation_folds": len(iso_all),
            "isolation_passed": int(sum(x["passed"] for x in iso_all)), "intervention_folds": len(iv_all),
            "intervention_passed": int(sum(x["passed"] for x in iv_all)),
            "intervention_predictions": int(sum(x["n_predictions"] for x in iv_all)),
            "intervention_max_abs_change": float(max(x["max_abs_change"] for x in iv_all)),
            "peek_caught": int(sum(x["peek_caught"] for x in iv_all)),
            "support_ok": int(sum(x["support_ok"] for x in iv_all)), "fixed_denominators": same,
            "cache_poisoning": {"n_folds": pois["n_folds"], "passed": pois["passed"]},
            "repro_isolation_BP_BPG": {d: v["folds_checked_isolation"] for d, v in rp["designs"].items()},
            "isolation": iso_all, "intervention": iv_all}
    leak["passed"] = bool(leak["isolation_passed"] == leak["isolation_folds"] and leak["intervention_passed"]
                          == leak["intervention_folds"] and pois["passed"])
    write_json(EA.INTERVENTION_REPORT, leak, wrote)
    leak["text"] = leakage_text(leak)
    if not leak["passed"]:
        sealcheck.stop("EA_CHECK", "run", tag, "leakage check failed (results/leakage/intervention_EA_CHECK.json): item "
                                               "1.0 stopped")
    st = EA.registry_status("M08")
    m08 = st["status"] == "PLANNED"
    res = {}
    t2 = time.time()
    for d in designs:
        t = rows[d]
        r = EA.score_design(t, prov, d, m08=m08)
        r["rows"] = t
        res[d] = r
        for name, df in r.items():
            if name != "rows":
                write_csv(OUT / f"{name}_{d}.csv", df, wrote)
        rr = EA.add_big_fold(t).join(prov, on="cell_id")
        rr["c5_btbp"] = rr.publication.astype(str) == paths.C5
        rr["label"] = EA.label(d)
        cols = CM.KEY + ["publication", "chemotype", "extractant", "cluster_cpg", "pubchem", "read_set", "a_read", "b_read",
                         "y", "B0", "B1", "HIST_D7", "big_fold", "c5_btbp", "loc_class", "loc_class_raw", "ornl_import", "label"]
        p = OUT / f"rows_EA_{d}.parquet"
        rr[cols].to_parquet(p, index=False, compression="zstd")
        wrote.append(p)
    if not m08:
        write_json(OUT / "m08_status.json", {"status": st["status"], "detail": st, "run": False,
                                             "note": "M08 appendix computed only when REGISTRY.csv marks it PLANNED"}, wrote)
    six = EA.six_design(rows, prov)
    for name, df in six.items():
        write_csv(OUT / f"{name}.csv", df, wrote)
    sx = pd.concat([rows[d].assign(design=d) for d in EA.SIX], ignore_index=True)
    sx["label"] = [EA.label(d, "C5_in", "training-free arms only") for d in sx.design]
    p = OUT / "rows_sixdesign.parquet"
    sx[["design", "split_seed", "fold", "cell_id", "A", "B", "publication", "chemotype", "extractant", "cluster_cpg",
        "read_set", "a_read", "b_read", "y", "B0", "HIST_D7", "label"]].to_parquet(p, index=False, compression="zstd")
    wrote.append(p)
    t_score = time.time() - t2
    gates = pd.concat([r["gate"] for r in res.values()], ignore_index=True)
    cons = EA.consistency(gates, {d: r["rows"] for d, r in res.items()})
    decision = make_decision(gates)
    compute = (f"repro + rows {t_rows:.0f}s for A/B/BR/BQ, tables {t_score:.0f}s, total {time.time() - t0:.0f}s; "
               f"{EA.env_note()}; caches: manifests/ea_check_caches.json")
    summary = {"item": EA.ITEM, "item_dir": "results/EA_CHECK", "estimand": "E-A (DIAGNOSTIC, seen)", "seal": tag,
               "designs_scored": list(designs), "label": "EXPLORATORY (seen); DIAGNOSTIC", "decision": decision,
               "consistency_checks": cons, "m08": {"status": st["status"], "run": m08},
               "six_design": six["sixdesign"][["design", "stratum", "contrast", "relSSE", "SF_POOL_p", "dMAE_bal", "label"]]
               .to_dict("records"),
               "breadth": pd.concat([r["breadth"] for r in res.values()], ignore_index=True)
               [["design", "stratum", "contrast", "variant", "G", "mean_L", "SF_LOG_p", "IM_p", "label"]].to_dict("records"),
               "leakage": {k: v for k, v in leak.items() if k not in ("isolation", "intervention")},
               "reproduction": {k: v for k, v in rp.items()}, "compute": compute}
    sname = "summary.json" if set(designs) == set(paths.DESIGNS) else f"summary_{designs[0]}.json"
    write_json(OUT / sname, summary, wrote)
    for d in designs:                                     # the section-9.3 table of every scored design (METRICS 12)
        CM.per_item_table(item_table(res, (d,), leak, compute, decision, tag), OUT / f"PER_ITEM_TABLE_{d}.md",
                          title=f"Item 1.0 EA_CHECK (E-A, DIAGNOSTIC, seen; {d}): per-item table")
        wrote.append(OUT / f"PER_ITEM_TABLE_{d}.md")
    if "BPG" in designs and "BP" in designs:
        CM.per_item_table(item_table(res, ("BPG", "BP"), leak, compute, decision, tag), OUT / "ITEM_TABLE.md",
                          title="Item 1.0 EA_CHECK (E-A, DIAGNOSTIC, seen): per-item table")
        wrote.append(OUT / "ITEM_TABLE.md")
    pd.set_option("display.width", 220)
    show = gates[gates.stratum != "C5_cell_out"][["design", "stratum", "contrast", "relSSE", "SF_POOL_p", "dMAE_bal", "dR2",
                                                  "gate_pass", "label"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("consistency checks:", json.dumps(cons, default=_default))
    detail = f"{run_detail(decision)}; {len(wrote)} files; {time.time() - t0:.0f}s"
    print("wrote:")
    for p in wrote:
        print("  ", rel(p))
    return 0, detail                  # the END line (with the post-run seal check) is written by sealcheck.run_sealed


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build-cache", action="store_true", help="build / verify the A, B, BR, BQ fold records (no score)")
    ap.add_argument("--cache-designs", default=None, help="comma list for --build-cache (default A,B,BR,BQ)")
    ap.add_argument("--force", action="store_true", help="--build-cache: rebuild every record")
    ap.add_argument("--amend", action="store_true", help="--build-cache after the seal (an amendment and a reseal follow)")
    ap.add_argument("--smoke", action="store_true", help="technical smoke, every metric suppressed, nothing written")
    ap.add_argument("--smoke-folds", type=int, default=1, help="--smoke: folds per new design for the fold checks")
    ap.add_argument("--repro", action="store_true", help="reproduction targets only (TECHNICAL SMOKE; needs the seal)")
    ap.add_argument("--design", choices=paths.DESIGNS, default=None, help="score one E-A design (default both)")
    a = ap.parse_args(argv)
    if a.build_cache:
        sealcheck.refuse_if_sealed("--build-cache (cache/*.pkl and manifests/ea_check_caches.json are sealed inputs)",
                                   a.amend)
        return build_cache(a)
    if a.smoke:
        return smoke(a)
    tag = sealcheck.require_seal()
    bench = D.load_bench()
    if a.repro:
        def repro_only():
            wrote = []
            OUT.mkdir(parents=True, exist_ok=True)
            rp = run_repro(bench, tag, wrote)
            print("wrote", ", ".join(rel(p) for p in wrote))
            return 0 if rp["passed"] else 1
        return sealcheck.run_sealed("EA_CHECK", "repro", tag, repro_only)
    return sealcheck.run_sealed("EA_CHECK", "run", tag, lambda: run(a, tag, bench))


if __name__ == "__main__":
    raise SystemExit(main())
