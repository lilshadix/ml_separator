#!/usr/bin/env python
"""Stage-0 seal of round 4: LF-normalised digests of the protocol documents, every round-4 code file, every
repository module the round-4 runner imports, the round-4 design documents, the inputs (cohort, log D matrix,
dev fold plans, fold-cache contents) and the package versions.  ``manifests/stage0_manifest.json`` +
``manifests/stage0_seal.txt`` (= SHA-256 of the manifest bytes).  A local seal is not public preregistration.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_seal.py --seal --reason "..."
    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_seal.py --verify
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from r4 import paths  # noqa: E402

DOCS = ("PROTOCOL.md", "EXPERIMENT_REGISTRY.csv", "DATA_CONTRACT.md", "METRICS.md", "RESOURCE_PLAN.md", "README.md",
        "PROVENANCE.md", "POWER.md", "DATA_REQUEST.md")
EVIDENCE = ("EVIDENCE_INVENTORY.md", "results/task_identity.json", "results/reproduction.json",
            "results/leakage/cache_poison_nontrain_0of3.json", "results/leakage/cache_poison_nontrain_1of3.json",
            "results/leakage/cache_poison_nontrain_2of3.json",
            # stage 1 (amendment A1): results seen before block A was sealed
            "STAGE0_REPORT.md", "results/REF/TABLE.md", "results/leakage/poison_REF.json",
            "results/provenance/summary.json", "results/provenance/cells.csv", "results/provenance/fingerprint_donor_class.json",
            "results/power/summary.json", "results/power/power_grid.csv", "results/power/run1/summary.json",
            "results/synthetic_a/run1/SUMMARY.md", "results/synthetic_a/NULL.json", "results/synthetic_a/CHEM_D7.json",
            "results/synthetic_a/GDCLASS.json", "results/synthetic_a/ELEC.json", "results/synthetic_a/PROJ.json",
            "results/synthetic_a/UNIVERSAL.json", "results/synthetic_a/SUMMARY.md", "results/power/run2/summary.json", "results/power/run3_badboundary/summary.json",
            "results/source_recovery.json", "results/blockA_review.json", "results/provenance/geometry_coverage.json",
            # amendment A2: block A outcome and block-B pre-seal evidence
            "results/blockA_decision.json", "results/S01/TABLE.md", "results/S02/TABLE.md",
            "results/leakage/poison_S01.json", "results/leakage/poison_S02.json", "results/provenance/fingerprint_blockB.json",
            "results/synthetic_b/run1/SUMMARY.md", "results/synthetic_b/NULL.json", "results/synthetic_b/GEO.json",
            "results/synthetic_b/MISSING.json", "results/synthetic_b/SUMMARY.md", "results/blockB_review.json",
            # amendment A3: block B outcome and the report
            "results/blockB_decision.json", "results/S05/TABLE.md", "results/S06/TABLE.md",
            "results/leakage/poison_S05.json", "results/leakage/poison_S06.json", "REPORT.md",
            # amendment A4: block-C pre-seal evidence (label-free counts, isolation, synthetic runs 2-3, review, tests)
            "results/blockC_counts.json", "results/leakage/isolation_an.csv", "results/blockC_review.json",
            "results/synthetic_c/run2/SUMMARY.md", "results/synthetic_c/run2/NULL_0.json",
            "results/synthetic_c/run2/NULL_2.json", "results/synthetic_c/run2/NULL_4.json",
            "results/synthetic_c/run2/AN_SIGNAL_0.json", "results/synthetic_c/run2/AN_SIGNAL_2.json",
            "results/synthetic_c/run2/AN_SIGNAL_4.json", "results/synthetic_c/design_diagnostics.json",
            *(f"results/synthetic_c/run3/rep_{i}.json" for i in range(6)), "results/synthetic_c/run3/SUMMARY.md",
            *(f"results/synthetic_c/rep_{i}.json" for i in range(6)), "results/synthetic_c/SUMMARY.md",
            "results/synthetic_c/summary.json", "results/blockC_review2.json", "results/logs/tests_a4.log",
            # amendment A5: block C outcome (close-out)
            "results/gate_c.json", "results/blockC_decision.json", "results/S11/TABLE.md",
            "results/leakage/poison_S11.json")
DESIGN = paths.G21 / "round4_design_20261022"
MANIFEST = paths.MAN / "stage0_manifest.json"
SEAL = paths.MAN / "stage0_seal.txt"


def _rel(p: Path) -> str:
    return str(p.resolve().relative_to(paths.ROOT)).replace("\\", "/")


def code_digests() -> dict:
    """Every .py file in the directory of every repository module the runner's closure imports (whole
    directories, so modules imported lazily during fitting are covered whatever the import order)."""
    import importlib
    sys.path.insert(0, str(paths.ROOT / "generations" / "gen19_chem_transfer"))   # block C's An(III) loader (A4)
    for name in ("r4.metrics", "r4.registry", "r4.refs", "r4.slots_a", "r4.slots_b", "r4.slots_c", "r4.an_data",
                 "gen19ct.data.load", "r4.donor", "r4.leakage", "r3.harness", "r3.metrics", "r3.corrector", "r3.configs",
                 "r3.policy", "r3.inner_preds", "r3.chem", "g21.cache", "g21.evaluate", "g21.methods",
                 "gen13sep.amplitude_bench", "gen20k.design", "gen20k.inference", "gen20k.splits", "gen20k.data"):
        importlib.import_module(name)
    from gen20k import data as D
    bench = D.load_bench()
    dirs = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not Path(f).is_absolute():          # e.g. torch.ops ('_ops.py'): never resolve against the cwd
            continue
        p = Path(f).resolve()
        try:
            p.relative_to(paths.ROOT)
        except ValueError:
            continue
        if ".venv" in p.parts or p.suffix != ".py":
            continue
        dirs.add(p.parent)
    files = {p.resolve() for d in dirs for p in d.glob("*.py")}
    for sub in ("r4", "scripts", "tests", "report"):
        files.update(p.resolve() for p in (paths.R4 / sub).glob("*.py"))
    files.add((paths.R3DIR / "scripts" / "r3_manifest.py").resolve())      # the seal's own digest function
    files.add((paths.R3DIR / "scripts" / "r3_cache_poison.py").resolve())  # compare() used by r4_cache_poison
    return {_rel(p): paths.sha256_text_lf(p) for p in sorted(files)}, bench


def input_digests(bench) -> dict:
    sys.path.insert(0, str(paths.R3DIR / "scripts"))
    from r3_manifest import content_digest  # round 3's order-stable digest (read-only import)
    from gen20k.splits import fold_record, folds_for
    out = {"cohort_manifest_sha256": paths.sha256_file(paths.ROOT / "generations/gen13_separation/manifests/cohort_exact.parquet"),
           "logD_matrix_sha256": hashlib.sha256(np.ascontiguousarray(np.nan_to_num(bench.Y, nan=-999.0)).tobytes()).hexdigest(),
           "n_cells": int(bench.Y.shape[0]), "folds": {}, "cache_content": {}}
    for d in ("BP", "BPG"):
        fr = fold_record(bench.frame, folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS)))
        out["folds"][d] = {"seeds": list(paths.DEV_SEEDS), "n_rows": int(len(fr)),
                           "sha256": hashlib.sha256(fr.to_csv(index=False, lineterminator="\n").encode()).hexdigest()}
    for p in sorted(paths.CACHE.glob("*.pkl")):
        with open(p, "rb") as fh:
            out["cache_content"][p.name] = content_digest(pickle.load(fh))
    from gen20k.splits import COPY_GROUPS_CSV, copy_group_map
    pubs = sorted(set(bench.frame["publication_id"].astype(str)))
    cg = copy_group_map()
    out["copy_groups_csv_sha256_lf"] = paths.sha256_text_lf(COPY_GROUPS_CSV)
    out["copy_group_map_cohort_sha256"] = hashlib.sha256(json.dumps({p: cg.get(p) for p in pubs}, sort_keys=True).encode()).hexdigest()
    out["bench_fields"] = content_digest({"basis": bench.basis, "coef": bench.coef, "groups": bench.groups.astype(str),
                                          "n_obs": bench.n_obs})
    from r4 import donor
    out["donor_smarts"] = donor.SMARTS
    out["gen19_extractant_components_sha256_lf"] = paths.sha256_text_lf(
        paths.ROOT / "generations/gen19_chem_transfer/descriptors/extractant_components.csv")
    import importlib
    gm = importlib.import_module("gen13sep.metals")
    out["gen13_physics_basis"] = {k: [round(float(x), 12) for x in v] for k, v in gm.physics_basis().items()}
    geo_root = paths.ROOT / "dataset with 3D structures"                     # block B inputs (A2)
    out["geometry_features_sha256"] = paths.sha256_file(geo_root / "features" / "features_by_safe_exp_id.parquet")
    out["accepted_geometries_sha256_lf"] = paths.sha256_text_lf(geo_root / "accepted_geometries.csv")
    out["archive_master_clean_sha256"] = paths.sha256_file(paths.ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet")
    from r4 import an_data                                                    # block C inputs (A4)
    an = an_data.an3_records()
    out["an3_records_sha256"] = hashlib.sha256(an.to_csv(index=False, lineterminator=chr(10)).encode()).hexdigest()
    out["an3_links_sha256"] = hashlib.sha256(json.dumps({e: sorted(v) for e, v in an_data.an_links().items()},
                                                        sort_keys=True).encode()).hexdigest()
    out["an3_counts"] = an_data.counts(an)
    return out


def environment() -> dict:
    import numpy, pandas, scipy, sklearn, torch, rdkit, catboost  # noqa: E401
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
            "rdkit": rdkit.__version__, "catboost": catboost.__version__}


STAGE = ("stage 1 close-out (amendment A5): blocks A, B and C scored; S11 no gain over B1, G-transfer failed (S13-S16 NOT RUN), "
         "S24 NOT RUN; no slot beats B1; final statuses")


def build() -> dict:
    code, bench = code_digests()
    return {"stage": STAGE,
            "docs": {d: paths.sha256_text_lf(paths.R4 / d) for d in DOCS},
            "design_docs": {p.name: paths.sha256_text_lf(p) for p in sorted(DESIGN.glob("*.md"))},
            "evidence": {e: paths.sha256_text_lf(paths.R4 / e) for e in EVIDENCE},
            "code": code, "inputs": input_digests(bench), "environment": environment(),
            "round3_seal": (paths.R3DIR / "manifests" / "freeze_seal.txt").read_text().split()[0]}


def verify() -> tuple[bool, str, str]:
    if not SEAL.exists() or not MANIFEST.exists():
        return False, "no stage-0 seal", ""
    seal = SEAL.read_text().split()[0]
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != seal:
        return False, "manifest digest != seal", seal
    sealed = json.loads(MANIFEST.read_text())
    live = build()
    for key in ("docs", "design_docs", "evidence", "code", "inputs", "environment", "round3_seal"):
        if live[key] != sealed[key]:
            if isinstance(live[key], dict):
                diff = sorted(k for k in set(live[key]) | set(sealed[key]) if live[key].get(k) != sealed[key].get(k))
                return False, f"{key} changed: {diff[:8]}", seal
            return False, f"{key} changed", seal
    return True, "ok", seal


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seal", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    if a.verify:
        ok, msg, seal = verify()
        print(("VERIFIED " if ok else "NOT VERIFIED ") + msg, seal)
        return 0 if ok else 1
    supersedes = None
    if SEAL.exists():
        old = SEAL.read_text().split()[0]
        amend = (paths.R4 / "AMENDMENTS.md").read_text(encoding="utf-8")
        if not a.reason or old[:12] not in amend:
            raise SystemExit("a seal exists: resealing needs --reason and an AMENDMENTS.md entry that names the old seal "
                             f"({old[:12]}) and lists every result seen")
        supersedes = old
    paths.MAN.mkdir(parents=True, exist_ok=True)
    man = build()                                    # build first: a failing build leaves the current seal in place
    if supersedes:
        MANIFEST.replace(paths.MAN / f"stage0_manifest_{supersedes[:12]}.json")     # archived, never overwritten
        SEAL.replace(paths.MAN / f"stage0_seal_{supersedes[:12]}.txt")
    man["sealed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    man["reason"] = a.reason
    man["supersedes"] = supersedes
    man["amendments_md_sha256_lf"] = paths.sha256_text_lf(paths.R4 / "AMENDMENTS.md")     # informational
    man["run_record_md_sha256_lf"] = paths.sha256_text_lf(paths.R4 / "RUN_RECORD.md")     # informational
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    seal = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    SEAL.write_text(seal + "  stage0_manifest.json\n", encoding="utf-8", newline="\n")
    with open(paths.R4 / "RUN_RECORD.md", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"- {man['sealed_utc']} SEAL {seal} ({len(man['code'])} code files, {len(man['docs'])} documents, "
                 f"{len(man['evidence'])} evidence files; supersedes {supersedes}; local only, not public "
                 f"preregistration). {a.reason}\n")
    print("sealed", seal, len(man["code"]), "code files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
