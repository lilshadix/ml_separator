#!/usr/bin/env python
"""Hash every Gen12 artefact, plus the frozen inputs it is only meaningful against.

A manifest listing only outputs cannot detect the failure that matters: an output
regenerated against a *different* cohort.  So the identity block pins the bundle
digest, the cohort fingerprint and the chemistry map alongside the artefact
digests, and ``--verify`` fails on any drift.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_manifest.py [--verify]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import paths  # noqa: E402

HASHED = {".csv", ".parquet", ".json", ".md", ".py", ".npz"}
LISTED = {".png", ".pdf", ".log", ".txt"}
MANIFEST = paths.MANIFEST_DIR / "manifest.json"
FROZEN_INPUTS = {
    "frozen_bundle": paths.BUNDLE_PARQUET,
    "ligand_2d_descriptors": paths.LIG2D_PARQUET,
    "frozen_chemistry_map": paths.CHEMISTRY_MAP_PARQUET,
    "gen7_cohort_cache": paths.GEN7_COHORT_PARQUET,
}


def digest(path: Path, chunk: int = 1 << 20) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def environment() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "sklearn", "rdkit", "catboost", "xgboost",
                 "torch", "matplotlib", "pyarrow"):
        try:
            versions[name] = __import__(name).__version__
        except Exception as error:
            versions[name] = f"unavailable ({type(error).__name__})"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=paths.REPO_ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                         cwd=paths.REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=paths.REPO_ROOT, text=True).strip())
    except Exception:
        commit, branch, dirty = "unknown", "unknown", None
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "machine": platform.machine(),
            "versions": versions, "git": {"commit": commit, "branch": branch, "dirty": dirty}}


def collect() -> dict:
    artefacts: dict = {}
    for path in sorted(paths.GEN12_ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in HASHED | LISTED or path.name == "manifest.json":
            continue
        relative = str(path.relative_to(paths.GEN12_ROOT))
        entry = {"bytes": path.stat().st_size}
        if path.suffix in HASHED:
            entry["blake2b_128"] = digest(path)
        artefacts[relative] = entry
    return artefacts


def build() -> dict:
    from gen12eu.cohort import build_cohort
    cohort = build_cohort()
    return {
        "generation": "Gen12Eu_pred",
        "identity": {
            "cohort_fingerprint": cohort.fingerprint,
            "cohort_rows": int(len(cohort.frame)),
            "cohort_extractants": int(cohort.frame["extractant"].nunique()),
            "cohort_chemotypes": int(cohort.frame["chemotype"].nunique()),
            "split_seeds": list(__import__("gen12eu.splits", fromlist=["x"]).SPLIT_SEEDS),
            "model_seed_rule": "42 + fold*1009 + 9999991",
            "bootstrap_seed": 8675309,
        },
        "frozen_inputs": {name: {"path": str(path.relative_to(paths.REPO_ROOT)),
                                 "sha256": paths.sha256_of(path)}
                          for name, path in FROZEN_INPUTS.items() if path.exists()},
        "environment": environment(),
        "artefacts": collect(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    current = build()
    if args.verify:
        if not MANIFEST.exists():
            print("no manifest to verify against"); return 1
        stored = json.loads(MANIFEST.read_text())
        problems = []
        for name, entry in stored["frozen_inputs"].items():
            if current["frozen_inputs"].get(name, {}).get("sha256") != entry["sha256"]:
                problems.append(f"frozen input drifted: {name}")
        if stored["identity"]["cohort_fingerprint"] != current["identity"]["cohort_fingerprint"]:
            problems.append("cohort fingerprint drifted")
        for path, entry in stored["artefacts"].items():
            now = current["artefacts"].get(path)
            if now is None:
                problems.append(f"artefact lost: {path}")
            elif "blake2b_128" in entry and now.get("blake2b_128") != entry["blake2b_128"]:
                problems.append(f"artefact changed: {path}")
        for problem in problems:
            print("DRIFT:", problem)
        print(f"{len(problems)} problems over {len(stored['artefacts'])} artefacts")
        return 1 if problems else 0
    MANIFEST.write_text(json.dumps(current, indent=1, default=str))
    print(f"stamped {len(current['artefacts'])} artefacts; "
          f"cohort {current['identity']['cohort_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
