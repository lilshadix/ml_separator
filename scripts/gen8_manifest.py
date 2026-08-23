"""Write the gen8 run manifest: environment, model registry, experiment inventory.

A leaderboard is only meaningful against a recorded environment.  This project has
already lost a completed sweep to a silent dependency change — ``pip install
tabpfn`` pulled scikit-learn 1.9.0 back to 1.6.1 and moved the reference arm by
0.0014 macro MAE — so every gen8 artefact is stamped with the library versions it
was produced under, and two runs whose ``library_versions`` differ must not share
a table.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ROOT = REPO_ROOT / "runs" / "gen8_architecture"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    import scipy
    import sklearn
    versions = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        versions["torch"] = None
    try:
        import rdkit
        versions["rdkit"] = rdkit.__version__
    except Exception:  # noqa: BLE001
        versions["rdkit"] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                    capture_output=True, text=True, check=False).stdout.strip())
    except Exception:  # noqa: BLE001
        commit, dirty = "", None

    cohort_path = REPO_ROOT / "runs/gen7_architecture/cache/cohort.parquet"
    cohort = pd.read_parquet(cohort_path)
    key = cohort[["row_id", "log_D"]].sort_values("row_id")
    payload = "|".join(f"{r}:{v:.10g}" for r, v in zip(key["row_id"], key["log_D"]))
    payload += f"||rows={len(cohort)}"
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]

    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "library_versions": versions, "git_commit": commit, "git_dirty": dirty,
        "cohort_path": str(cohort_path.relative_to(REPO_ROOT)),
        "cohort_fingerprint": fingerprint,
        "cohort_rows": int(len(cohort)),
        "cohort_extractants": int(cohort["extractant"].nunique()),
        "cohort_chemotypes": int(cohort["tanimoto_cluster"].nunique()),
        "frozen_global_model": "REC_ecfp_plus_recovered",
        "frozen_oof": "runs/gen7_architecture/finalists/oof_predictions.parquet",
        "split_seeds": [104729, 130363, 155921, 196613, 262147],
        "model_seed_formula": "42 + fold*1009 + 9999991",
    }
    (ROOT / "environment.json").write_text(json.dumps(environment, indent=2))

    from lanthanide_separation.gen8.adapters import default_adapters
    from lanthanide_separation.gen8.kshot import NON_DEPLOYABLE, POLICIES

    registry = {
        "adapters": [], "policies": [], "protocols": {
            "P1": "exhaustive, gen7-compatible: every row a candidate, scored on the rest",
            "P2": "acquisition-fair: disjoint candidate pool and evaluation set per repeat",
        }}
    for adapter in default_adapters():
        registry["adapters"].append({
            "name": adapter.name, "module": type(adapter).__module__,
            "class": type(adapter).__name__,
            "trainable": hasattr(adapter, "fit_fold"),
            "doc": (type(adapter).__doc__ or "").strip().split("\n")[0]})
    for module_name, builder in (
            ("lanthanide_separation.gen8.slope_restore", "build_slope_adapters"),
            ("lanthanide_separation.gen8.curve_baselines", "build_curve_adapters"),
            ("lanthanide_separation.gen8.physics_latent", "build_physics_adapters"),
            ("lanthanide_separation.gen8.cnp", "build_cnp_adapters")):
        try:
            import importlib
            module = importlib.import_module(module_name)
            for adapter in getattr(module, builder)():
                registry["adapters"].append({
                    "name": adapter.name, "module": module_name,
                    "class": type(adapter).__name__,
                    "trainable": hasattr(adapter, "fit_fold"),
                    "doc": (type(adapter).__doc__ or "").strip().split("\n")[0]})
        except Exception as error:  # noqa: BLE001
            registry.setdefault("unavailable", []).append(
                {"module": module_name, "error": f"{type(error).__name__}: {error}"})
    for name in sorted(POLICIES):
        registry["policies"].append({"name": name, "deployable": name not in NON_DEPLOYABLE})
    registry["policies"].append({"name": "ORACLE", "deployable": False,
                                 "note": "reads candidate targets; analysis upper bound only"})
    (ROOT / "model_registry.json").write_text(json.dumps(registry, indent=2))

    inventory = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_dir() or path.name in ("experiment_manifest.json",):
            continue
        if path.suffix in (".log",) or "__pycache__" in str(path):
            continue
        inventory.append({
            "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256_16": digest(path) if path.stat().st_size < 200_000_000 else None})
    manifest = {"created_utc": environment["created_utc"], "environment": environment,
                "n_artifacts": len(inventory), "artifacts": inventory}
    (ROOT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(versions, indent=2))
    print(f"cohort fingerprint {fingerprint} / {len(inventory)} artifacts")
    print(f"adapters registered: {len(registry['adapters'])}, "
          f"policies: {len(registry['policies'])}")
    if "unavailable" in registry:
        print("unavailable:", json.dumps(registry["unavailable"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
