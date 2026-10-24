#!/usr/bin/env python
"""Stage-1 item 1.8, M07: the sc009 (TODGA-class) Nd dip frozen at -0.11 for the Pr/Nd warning, estimand E-KX
(CLAUDE_CODE_PROMPT.md section 6.8; conventions 6.0, 6.10, 8, 9).  Code: r5/ekx.py.  Spec: results/EKX/ITEM_SPEC.md.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ekx.py \
        [--design BPG|BP|both] [--smoke | --repro] [--perm-draws N]

Modes (run from the repository root):
  default   requires the round-5 seal (r5.sealcheck.require_seal); runs the reproduction target first (a mismatch stops
            the item) and the intervention before any scored write (a failure stops the item); builds every arm,
            comparator, control, stratum and sensitivity of section 6.8 on the round-4 REF rows of each design (B1 base
            = the design's OOF B1; the BP OOF prediction is the primary base, the item is E-KX-LOPO and never called BP;
            the BPG base is a sensitivity), runs the random-chemotype permutation (seeded, --perm-draws >= 1,000),
            the publication sign-flip over sc009 publications, leave-top-3-out (the report rule), the acid strata, the
            donor-pool and C5-BTBP sensitivities, LOPO and the unit table, the section-8 prediction intervention with
            its positive controls, and writes row-level outputs and tables under results/EKX/ (every number labelled,
            section 9.1; the seal tag on every output, a 'seal' column in every CSV, the seal and code LF digests in
            every JSON) and the leakage report under results/leakage/.  EXPLORATORY (seen): never next to 0.452; the
            BPG outputs are labelled SENSITIVITY.  The consistency lines are printed and never stop the item.
  --repro   requires the seal; the item's section-6.0 reproduction target only (B1 from the fold betas within 8.9e-16
            on every row; construction identities and row counts), labelled TECHNICAL SMOKE, written to
            results/EKX/repro_<design>.json; exit code 1 on a mismatch (a mismatch stops the item).
  --smoke   no seal needed; PRE-SEAL ENGINEERING CHECK with every metric suppressed: identity checks on real data (they compare
            reconstructions with the parquet, never a prediction with a label), then every arm, table, permutation and
            the intervention run on a seeded noise surrogate (V1's NaN pattern kept; y, B1 and B0 replaced), and only
            shapes, counts, NaN patterns, check flags and timings are printed and written
            (results/EKX/SMOKE_COUNTS_<design>.json).  No R2, SSE, MAE or contrast value is computed on real labels.
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
from r5 import ekx as EKX  # noqa: E402

paths.redirect_inner_cache()
OUT = paths.RES / "EKX"
LEAK = paths.RES / "leakage"
CODE = ("r5/ekx.py", "r5/common.py", "r5/dupes.py", "r5/population.py", "scripts/s1_ekx.py")
SMOKE_DRAWS = 20


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


def surrogate(D, built: dict, key: str = "EKX|SMOKE_SURROGATE"):
    """Seeded noise in place of every label: Y (NaN pattern kept) and the rows' y, B1, B0."""
    rng = np.random.default_rng(paths.seed_for(key))
    Ys = np.where(np.isfinite(D.Y), rng.normal(0.0, 1.0, D.Y.shape), np.nan)
    out = {}
    for design, (R, ck) in built.items():
        S = R.copy()
        for c in ("y", "B1", "B0"):
            S[c] = rng.normal(0.0, 0.3, len(S))
        out[design] = (S, ck)
    return Ys, out


def smoke_counts(R: pd.DataFrame, ck: dict, TV: dict, T: dict, iv: dict, ext: dict, seconds: float) -> dict:
    """Counts only: shapes, NaN shares, donor-count distributions (metadata), table shapes, check flags, timings."""
    tv = TV[(EKX.PRIMARY_C5, "PRIMARY")]
    num = [c for c in R.columns if R[c].dtype.kind == "f" and c not in ("y", "B1", "B0")]
    e1 = R[R.is_sc009 & R.is_prnd]
    return {"label": "PRE-SEAL ENGINEERING CHECK (--smoke: every metric suppressed; tables run on a noise surrogate)",
            "rows_shape": list(R.shape), "identity_checks_on_real_data": ck,
            "nan_share": {c: round(float(R[c].isna().mean()), 4) for c in num},
            "sc009_prnd_rows_per_seed": e1.groupby("split_seed").size().astype(int).to_dict(),
            "sc009_prnd_cells": int(e1.cell_id.nunique()), "sc009_prnd_publications": int(e1.publication.nunique()),
            "dphi_nd_nonzero_rows": int((R.dphi_Nd != 0).sum()),
            "targets_with_chem_donors": int((tv.n_chem_pub > 0).sum()),
            "sc009_targets_n_chem_pub": tv[tv.is_sc009].n_chem_pub.value_counts().sort_index().astype(int).to_dict(),
            "n_ndu_pub_range": [int(tv.n_ndu_pub.min()), int(tv.n_ndu_pub.max())],
            "acid_strata_targets": tv.acid_stratum.value_counts().astype(int).to_dict(),
            "frozen_external_n_w": [ext["n_donor_publications"], round(ext["w"], 6)],
            "table_shapes_on_surrogate": {k: list(v.shape) for k, v in T.items() if isinstance(v, pd.DataFrame)},
            "intervention_on_surrogate": iv, "seconds": round(seconds, 1)}


ITEM = "EKX"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=("BPG", "BP", "both"), default="both")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="technical smoke, every metric suppressed (no seal needed)")
    g.add_argument("--repro", action="store_true", help="the section-6.0 reproduction target only (TECHNICAL SMOKE)")
    ap.add_argument("--perm-draws", type=int, default=None,
                    help=f"random-chemotype draws (default {EKX.B_PERM}; {SMOKE_DRAWS} in --smoke; >= 1,000 otherwise)")
    a = ap.parse_args(argv)
    B = a.perm_draws if a.perm_draws is not None else (SMOKE_DRAWS if a.smoke else EKX.B_PERM)
    if not a.smoke and not a.repro and B < 1000:
        raise SystemExit("section 6.8 needs >= 1,000 random-chemotype draws")
    if a.smoke:
        print("PRE-SEAL ENGINEERING CHECK (--smoke): every metric suppressed; no seal required")
        return body(a, "UNSEALED (technical smoke)", B)
    from r5 import sealcheck
    tag = sealcheck.require_seal()
    return sealcheck.run_sealed(ITEM, "repro" if a.repro else "run", tag, lambda: body(a, tag, B))


def body(a, tag: str, B: int) -> int:
    from r5 import sealcheck
    t00 = time.time()
    mode = "smoke" if a.smoke else ("repro" if a.repro else "run")
    designs = ("BPG", "BP") if a.design == "both" else (a.design,)
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    D = EKX.Donors()
    built = {d: EKX.build_rows(D, d) for d in designs}
    cons = None                               # the consistency lines (not computed in --smoke)
    if not a.smoke:                           # brief 1.2 / 6.0: the reproduction target first; a mismatch stops the item
        rep = EKX.reproduction(D, built)
        cons = rep["consistency"]
        for d in designs:
            out = {"label": EKX.SMOKE, "seal": tag, "started_utc": started, "design": d, "target": rep["targets"][d],
                   "consistency": {"rows": rep["consistency"][d], "frozen_value_source": rep["consistency"]["frozen_value_source"]},
                   "rows_total_this_run": rep["rows_total"], "passed": rep["targets"][d]["passed"],
                   "code_lf_sha256": code_digests()}
            write_json(OUT / f"repro_{d}.json", out)
            t = rep["targets"][d]
            print(f"TECHNICAL SMOKE reproduction {d}: {'MATCH' if t['passed'] else 'MISMATCH (item stops)'}; B1 "
                  f"{t['b1_predict_levels_maxabs']:.1e} (tol {EKX.B1_TOL:.1e}); rows {t['checks']['rows']}")
        for r in EKX.consistency_table(cons).itertuples(index=False):
            print(f"  {r.label} (does not stop the item): {r.line}: {r.observed} vs {r.reference} -> {r.result}")
        print(f"  rows over the designs run: {rep['rows_total']}; consistency lines (do not stop the item) written")
        if a.repro:
            return 0 if rep["passed"] else 1
        if not rep["passed"]:
            sealcheck.stop(ITEM, mode, tag, "reproduction mismatch (results/EKX/repro_<design>.json) -- item stops "
                                            "(brief 6.0)")
    if a.smoke:
        Yw, built = surrogate(D, built)
    else:
        Yw = None
    targets = D.c3_ids
    t0 = time.time()
    TV = EKX.all_target_values(D, targets, Y=Yw)
    ext = EKX.frozen_external(D, sc=None if cons is None else cons["frozen_value_source"])
    labs, labs_seed = EKX.perm_labels(D, B)
    U = {c5: EKX.perm_u(D, targets, labs, Y=Yw, c5=c5) for c5 in EKX.PERM_C5}
    print(f"  target values ({len(EKX.CONFIGS)} donor configurations) and {B} random-chemotype draws "
          f"({time.time() - t0:.0f}s)")
    status = 0
    for design in designs:
        t1 = time.time()
        R, ck = built[design]
        betas = EKX.fold_betas(design)
        OFF = {c5: EKX.offsets(D, R, betas=betas, Y=Yw, c5=c5) for c5 in EKX.C5_VARIANTS}
        EKX.add_arms(R, TV, OFF, ext["u_ext"])
        iv = EKX.intervention(D, R, design, betas=betas, Y=Yw)
        if not a.smoke:                       # s0-mechanics-17: leakage before any scored computation or write
            leak = {"label": "LEAKAGE CHECK (section 8 items 1, 4, 5)", "seal": tag, "intervention": iv,
                    "isolation": "every donor set passes Donors.guard (no cell of the target's publication or copy group, "
                                 "never the target, no bit-identical copy outside NB); B1 base = round-4 REF run whose "
                                 "intervention and identity checks passed (r5.common.load_rows)",
                    "support": {k: ck[k] for k in ("both_ends_read_rows", "adjacent_lighter_first",
                                                   "readset_or_fold_inconsistent_groups", "b1_predict_levels_maxabs",
                                                   "y_maxabs", "kx_additive_identity_maxabs", "dphi_nd_on_cells_without_nd")},
                    "fit_poisoning": "not applicable: E-KX fits nothing (u is a donor mean with K = 5 fixed; beta = the "
                                     "stored round-4 fold fit)",
                    "cache_poisoning": "not applicable: no new cache", "code_lf_sha256": code_digests()}
            write_json(LEAK / f"EKX_{design}.json", leak)
            if not iv["passed"]:
                sealcheck.stop(ITEM, mode, tag, f"leakage check failed ({design}; results/leakage/EKX_{design}.json): "
                                                "item stopped")
        T = EKX.all_tables(R, design, D, targets, U, labs_seed, Y=Yw)
        if a.smoke:
            txt = EKX.per_item_table(EKX.per_item(T, design, seal_tag=tag, leakage="smoke", seconds=0.0, ext=ext))
            print(f"  per-item table exercised on the surrogate: {len(txt.splitlines())} lines (not written)")
            counts = smoke_counts(R, ck, TV, T, iv, ext, time.time() - t1)
            write_json(OUT / f"SMOKE_COUNTS_{design}.json", counts)
            print(f"  {design}: rows {R.shape}; sc009 Pr/Nd rows per seed {sorted(set(counts['sc009_prnd_rows_per_seed'].values()))}; "
                  f"tables {len(T)}; intervention passed {iv['passed']} ({iv['seconds']}s); {time.time() - t1:.0f}s")
            for k, v in T.items():
                print(f"    table {k}: {v.shape}")
            continue
        R["label"] = f"{EKX.design_label(design)}; {EKX.NOT_NEXT_TO}"
        R["seal"] = tag
        R.to_parquet(OUT / f"rows_{design}.parquet", index=False)
        print("wrote", OUT / f"rows_{design}.parquet")
        for k, v in T.items():
            write_csv(OUT / f"{k}_{design}.csv", v.assign(seal=tag))
        lk = (f"intervention {'passed' if iv['passed'] else 'FAILED'} on {iv['targets']} targets x "
              f"{len(iv['configs'])} donor configurations + offsets; positive controls caught "
              f"{iv['pc_self_guard_caught']}/{iv['targets']} (self), {iv['pc_samepub_guard_caught']}/"
              f"{iv['pc_samepub_guard_applicable']} (same publication), {iv['pc_leaky_caught']}/"
              f"{iv['pc_leaky_applicable']} (leaky pool); supports exact; B1 identity {ck['b1_predict_levels_maxabs']:.1e}")
        item = EKX.per_item(T, design, seal_tag=tag, leakage=lk, seconds=time.time() - t1, ext=ext, consistency=cons)
        EKX.per_item_table(item, OUT / f"PER_ITEM_TABLE_{design}.md", title=f"1.8 M07 E-KX ({design}): per-item table")
        print("wrote", OUT / f"PER_ITEM_TABLE_{design}.md")
        rule = T["top3_rule"].to_dict(orient="records")
        write_json(OUT / f"summary_{design}.json", {
            "label": EKX.design_label(design), "note": EKX.NOT_NEXT_TO, "seal": tag, "started_utc": started,
            "design": design,
            "primary_design": EKX.PRIMARY_DESIGN, "identity_checks": ck, "intervention_passed": iv["passed"],
            "report_rule": rule, "seen_values": EKX.SEEN_VALUES, "frozen_external": ext,
            "consistency_lines": EKX.consistency_table(cons, designs=(design,)).to_dict(orient="records"),
            "perm_draws": B,
            "perm_rng_seed": labs_seed, "tables": sorted(T), "code_lf_sha256": code_digests(),
            "seconds": round(time.time() - t1, 1)})
    if not a.smoke:
        prov = {"seal": tag, "code_lf_sha256": code_digests()}
        write_csv(OUT / "targets.csv", EKX.targets_table(TV).assign(seal=tag))
        u, c, sc = EKX.chord_tables(D)
        write_csv(OUT / "chord_units.csv", u.assign(seal=tag))
        write_csv(OUT / "chord_chemotypes.csv", c.assign(seal=tag))
        write_json(OUT / "chord_sc009.json", {**sc, **prov})
        write_json(OUT / "reliability.json", {**EKX.split_half(D), **prov})
        write_json(OUT / "frozen_external.json", {**ext, **prov})
    print(f"done {time.time() - t00:.0f}s (status {status})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
