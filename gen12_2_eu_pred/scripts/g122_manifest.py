#!/usr/bin/env python
"""Freeze the generation: hash every artefact, record the environment and the git commit.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_manifest.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import coordination, paths  # noqa: E402

SKIP = {".DS_Store"}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "rdkit", "catboost", "xgboost",
                 "matplotlib", "pyarrow"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


def git() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  cwd=paths.REPO_ROOT).stdout.strip()
        except OSError:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def main() -> int:
    artefacts = {}
    for path in sorted(paths.GEN122_ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name in SKIP:
            continue
        relative = str(path.relative_to(paths.GEN122_ROOT))
        if relative == "manifests/manifest.json":
            continue
        artefacts[relative] = {
            "bytes": path.stat().st_size,
            "blake2b_128": hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()}

    gen12_plan = json.loads(paths.GEN12_FOLD_PLAN_B.read_text())
    choice = json.loads((paths.MANIFEST_DIR / "level_definition_choice.json").read_text())
    audit = json.loads((paths.FEATURE_DIR / "coordination_audit.json").read_text())
    manifest = {
        "generation": "Gen12.2",
        "directory": "gen12_2_eu_pred",
        "brief_name": "Gen1Eu_pred-2 (the brief's spelling; renamed to the repository convention)",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity": {
            "cohort_fingerprint": gen12_plan["cohort_fingerprint"],
            "gen12_root": str(paths.GEN12_ROOT.relative_to(paths.REPO_ROOT)),
            "split_seeds": gen12_plan["seeds"],
            "design": "B (chemotype hold-out), inherited from Gen12 unchanged",
            "bootstrap_seed": 8675309,
        },
        "frozen_inputs": {
            "coordination_spec": {"path": "config/coordination_smarts.json",
                                  "sha256": coordination.spec_digest(),
                                  "version": coordination.spec()["spec_version"]},
            "level_definition_rules": {
                "path": "config/level_definition_rules.json",
                "sha256": paths.sha256_of(paths.CONFIG_DIR / "level_definition_rules.json")},
            "pre_registration_sha256": paths.sha256_of(paths.GEN122_ROOT / "PRE_REGISTRATION.md"),
            "descriptor_spec_sha256": paths.sha256_of(
                paths.GEN122_ROOT / "COORDINATION_DESCRIPTOR_SPEC.md"),
        },
        "decisions": {
            "primary_level_definition": choice["decision"]["primary_level_definition"],
            "secondary_level_definition": choice["decision"]["secondary_level_definition"],
            "level_reliable_threshold_cells": choice["level_reliable_cohort"]["threshold_cells"],
            "multi_arm_rule": audit["multi_arm"]["rule"],
            "n_multi_arm": audit["multi_arm"]["n_multi_arm"],
        },
        "feature_matrix": {"blake2b": audit["feature_matrix_blake2b"],
                           "columns_defined": audit["n_columns_defined"],
                           "columns_informative": audit["n_columns_informative"]},
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "versions": versions(), "git": git()},
        "artefacts": artefacts,
    }
    (paths.MANIFEST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(artefacts)} artefacts hashed")
    print(json.dumps(manifest["decisions"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
