#!/usr/bin/env python
"""Stage-0 input, brief section 4.3 item 12: the external DOPT_Q read policy for new cells.

A new cell has no fold, hence no fold covariance.  C_all and s2_all are g21.cache.build_fold's outer covariance and
noise variance (G14 residual covariance from inner cross-fits of the training set, PSD-projected; robust replicate
noise x gen20's inner-selected noise factor) fitted once with the whole corpus training: the outer block of the BPG
all-cell record of s0_beta_frozen.py (every bench cell trains, all 229 C3 cells included; declared inner seed
gen20k.splits.inner_seed(104729, 0)).  A new cell's reads are
    gen20k.design.support_sequence(dz, 'DOPT_Q', min(k_max, 6), cov=C_all, noise_var=2 * s2_all),
4 reads = union of seq[:3], 3 reads = seq[:2] (r5.frozen.external_sequence / read_set).

Label-free summaries only (no prediction, no score): the frozen object's parameters, the external sequences of the
229 corpus C3 cells (stored in the pickle), the Gd-read rate at 3 / 4 reads, and three read-set sensitivities:
design BP, the outer recipe on the 229 C3 cells only, and agreement with the dev folds' cached BPG sequences.

Writes cache/ext/ext_dopt.pkl (a rerun must give identical bytes; a differing pickle is never overwritten) and
results/stage0/ext_dopt.json.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore \
        generations/gen21/round5_20261024/scripts/s0_external_dopt.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402

import numpy as np  # noqa: E402

from r5 import frozen as FR  # noqa: E402
from g21 import cache as GC  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k.data import IDX, cell_view  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

OUT = FR.STAGE0 / "ext_dopt.json"
GD = IDX["Gd"]


sequences = FR.corpus_sequences


def agreement(sa: dict, sb: dict, reads=(3, 4)) -> dict:
    keys = sorted(set(sa) & set(sb))
    out = {"n_cells": len(keys)}
    for r in reads:
        same = [FR.read_set(sa[k], r) == FR.read_set(sb[k], r) for k in keys if len(sa[k]) >= r - 1 and len(sb[k]) >= r - 1]
        out[f"same_read_set_{r}_reads"] = f"{sum(same)}/{len(same)}"
    out["same_full_sequence"] = f"{sum(sa[k] == sb[k] for k in keys)}/{len(keys)}"
    return out


def cov_diff(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return {"max_abs_diff": float(np.abs(a - b).max()), "rel_frobenius": float(np.linalg.norm(a - b) / np.linalg.norm(a))}


def gd_rate(seqs: dict, gd_cells: set, r: int) -> str:
    ks = [k for k in seqs if k in gd_cells and len(seqs[k]) >= r - 1]
    return f"{sum(GD in FR.read_set(seqs[k], r) for k in ks)}/{len(ks)}"


def main() -> int:
    from r5 import sealcheck
    sealcheck.refuse_if_sealed("scripts/s0_external_dopt.py (rewrites sealed Stage-0 inputs)", "--amend" in sys.argv[1:])
    t0 = time.time()
    bench = D.load_bench()
    c3 = FR.c3_cells(bench)
    ids = [str(bench.frame.cell_id.iat[ci]) for ci in c3]
    print(f"Stage-0 input (brief 4.3 item 12): external DOPT_Q; bench {len(bench.frame)} cells, C3 {len(c3)}", flush=True)
    rec, info = FR.load_or_build(bench, "BPG")
    chk = FR.record_checks(bench, rec)
    if not chk["ok"]:
        raise AssertionError(f"BPG all-cell record fails its construction checks: {chk}")
    ext = FR.build_external(bench, rec, record_sha256=info["sha256"])
    seqs = ext["corpus_c3_sequences"]
    if list(seqs) != ids:
        raise AssertionError("corpus sequences are not keyed by the C3 cells in bench order")
    for ci in c3[:25]:                       # the helper path equals the batch path
        dz = DS.make_design(cell_view(bench, ci), "adjacent", "metal")
        assert FR.external_sequence(ext, dz) == seqs[str(bench.frame.cell_id.iat[ci])]
        assert FR.external_sequence(ext, FR.design_for_metals(dz.cell.metals)) == FR.external_sequence(ext, dz)
    b = FR.dump_bytes(ext)
    if FR.EXT_PATH.exists():
        if FR.EXT_PATH.read_bytes() != b:
            raise AssertionError(f"rebuilt external object differs from the stored {FR.EXT_PATH}; not overwritten")
        status = "rebuilt, bit-identical to the stored pickle"
    else:
        FR.EXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FR.EXT_PATH.write_bytes(b)
        status = "built"
    print(f"  ext object {status}: n_train {ext['n_train']}, s2_all {ext['s2_all']:.6f} (robust {ext['noise_s2_robust']:.6f}"
          f" x factor {ext['noise_factor']}), noise_var {ext['noise_var']:.6f}; C_all {ext['C_all'].shape}, "
          f"min eig {np.linalg.eigvalsh(ext['C_all']).min():.3g}", flush=True)
    lens = [len(s) for s in seqs.values()]
    gd_cells = {str(bench.frame.cell_id.iat[ci]) for ci in c3 if np.isfinite(bench.Y[ci, GD])}

    # ---- sensitivities (label-free read-set comparisons) ----
    sens = {}
    rec_bp, info_bp = FR.load_or_build(bench, "BP")
    e_bp = FR.external_dopt(rec_bp, record_sha256=info_bp["sha256"])
    s_bp = sequences(bench, c3, e_bp["C_all"], e_bp["noise_var"])
    sens["design_BP"] = {"s2_all": e_bp["s2_all"], "noise_factor": e_bp["noise_factor"], "cov": cov_diff(ext["C_all"], e_bp["C_all"]),
                         "read_sets_vs_primary": agreement(seqs, s_bp)}
    t1 = time.time()
    outer_c3, _ = FR.outer_block(bench, FR.all_cells_fold(bench, "BPG", train_index=c3))
    s_c3 = sequences(bench, c3, outer_c3["cov"], 2 * outer_c3["s2"])
    sens["outer_recipe_on_C3_only"] = {"n_train": len(outer_c3["train_index"]), "s2_all": float(outer_c3["s2"]),
                                       "noise_factor": float(outer_c3["noise_factor"]),
                                       "noise_s2_robust": float(outer_c3["noise_s2_robust"]),
                                       "cov": cov_diff(ext["C_all"], outer_c3["cov"]),
                                       "read_sets_vs_primary": agreement(seqs, s_c3), "seconds": round(time.time() - t1, 1)}
    dev, dev_nf = {}, {}
    for design in ("BPG", "BP"):
        nf = []
        for f in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS)):
            r = GC.load(f.design, f.seed, f.fold)
            nf.append(float(r["outer"]["noise_factor"]))
            if design == "BPG":
                d = dev.setdefault(int(f.seed), {})
                d.update({c["cell_id"]: c["seqs"]["DOPT_Q"] for c in r["outer"]["cells"] if c["k_max"] >= 3})
            del r
        dev_nf[design] = {str(x): nf.count(x) for x in sorted(set(nf))}
    sens["dev_fold_BPG_sequences"] = {str(s): agreement(seqs, d) for s, d in dev.items()}
    sens["dev_fold_noise_factors (context: 25 cached outer folds per design)"] = dev_nf
    for k, v in sens.items():
        print(f"  sensitivity {k}: {json.dumps(v)}", flush=True)

    out = {
        "label": "Stage-0 input (CLAUDE_CODE_PROMPT.md 4.3 item 12), computed before the Stage-0 seal; label-free read "
                 "policy for new cells (no prediction, no score)",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "pickle": {"path": FR.EXT_PATH.relative_to(paths.R5).as_posix(), "sha256": FR.sha256_bytes(b), "status": status,
                   "bytes": len(b)},
        "usage": "ext = r5.frozen.load_external(); seq = r5.frozen.external_sequence(ext, dz) with "
                 "dz = r5.frozen.design_for_metals(metal indices); reads = r5.frozen.read_set(seq, 4) (= seq[:3]) or 3 "
                 "(= seq[:2]); B1 on new data uses beta_frozen (results/stage0/beta_frozen.json)",
        "recipe": {"source_record": {**info, "checks": chk},
                   "outer_recipe": "r5.frozen.outer_block = g21.cache.build_fold's outer part (fold_covariance of G14 "
                                   "inner OOF residuals, PSD-projected; s2 = training_noise_s2 x select_on_inner noise "
                                   "factor)",
                   "training_set": "every bench cell (all 229 C3 cells included), design BPG, declared inner seed "
                                   "gen20k.splits.inner_seed(104729, 0)",
                   "policy": "support_sequence(dz, 'DOPT_Q', min(k_max, 6), cov=C_all, noise_var=2*s2_all)",
                   "blas_threads": FR.BLAS_THREADS},
        "parameters": {"n_train": ext["n_train"], "s2_all": ext["s2_all"], "noise_var": ext["noise_var"],
                       "noise_factor": ext["noise_factor"], "noise_s2_robust": ext["noise_s2_robust"],
                       "ridge_lambda_gen20": float(rec["outer"]["ridge_lambda"]),
                       "C_all_diag": [float(x) for x in np.diag(ext["C_all"])],
                       "C_all_min_eig": float(np.linalg.eigvalsh(ext["C_all"]).min()),
                       "noise_factor_selection_scores": {k: v for k, v in rec["outer"]["selection_scores"].items()
                                                         if k.startswith("G14@BLUP20@noise")}},
        "corpus_c3": {"n_cells": len(seqs), "sequence_length_counts": {str(n): lens.count(n) for n in sorted(set(lens))},
                      "gd_read_rate_3_reads": gd_rate(seqs, gd_cells, 3), "gd_read_rate_4_reads": gd_rate(seqs, gd_cells, 4),
                      "note": "Gd-read rate = C3 cells with Gd measured whose external read set contains Gd"},
        "sensitivities": sens,
        "code_lf_sha256": {"r5/frozen.py": paths.sha256_text_lf(paths.R5 / "r5" / "frozen.py"),
                           "scripts/s0_external_dopt.py": paths.sha256_text_lf(Path(__file__))},
        "seconds": round(time.time() - t0, 1),
    }
    paths.write_text_lf(OUT, json.dumps(out, indent=1) + "\n")
    print(f"  corpus C3 sequences: lengths {out['corpus_c3']['sequence_length_counts']}; Gd-read rate 3 reads "
          f"{out['corpus_c3']['gd_read_rate_3_reads']}, 4 reads {out['corpus_c3']['gd_read_rate_4_reads']}")
    print(f"wrote {FR.EXT_PATH} (sha256 {out['pickle']['sha256'][:12]}, {status})")
    print(f"wrote {OUT}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
