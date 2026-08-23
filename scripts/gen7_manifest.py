"""Write the reproducibility artifacts the gen7 brief requires.

``environment.json``        interpreter, packages, hardware, and the two macOS
                            OpenMP fixes without which LightGBM/XGBoost deadlock
``model_registry.json``     every contender the suites can build, generated from the
                            code that actually runs — not hand-maintained
``experiment_manifest.json``every run directory found, with its artifacts and hashes
``data_audit.json``         cohort composition, block inventory and the fingerprint
                            that certifies identical test rows
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

OUT = REPO_ROOT / "runs" / "gen7_architecture"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    try:
        packages = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"],
                                  capture_output=True, text=True, timeout=120).stdout
        packages = {p["name"]: p["version"] for p in json.loads(packages)}
    except Exception:
        packages = {}
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        git, branch = None, None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "git_commit": git, "git_branch": branch,
        "notes": {
            "openmp_fix": (
                "LightGBM and XGBoost each ship their own libomp; on macOS a process "
                "that has already initialised torch's libomp deadlocks at 0 % CPU when "
                "the second of them starts a thread pool. Fixed by adding torch/lib to "
                "the rpath of lib_lightgbm.dylib and libxgboost.dylib (and re-signing) so "
                "one libomp serves all three, plus OMP_NUM_THREADS=4 and n_jobs=4 for "
                "both boosters."),
            "rdkit": "installed for gen7; the bundled 2D descriptor parquet predates it",
        },
    }


def model_registry() -> dict:
    from lanthanide_separation.gen7 import suites
    from lanthanide_separation.gen7.harness import load_cohort
    cohort = load_cohort()
    registry: dict = {}
    for suite, factory in suites.SUITES.items():
        try:
            contenders = factory(cohort)
        except Exception as error:
            registry[suite] = {"error": f"{type(error).__name__}: {error}"}
            continue
        entries = []
        for contender in contenders:
            entry = {"name": contender.name, "class": type(contender).__name__}
            for field in ("blocks", "arm", "learner", "ligand_kernel", "pca_blocks",
                          "pca_components", "add_indicator", "weighting"):
                if hasattr(contender, field):
                    value = getattr(contender, field)
                    entry[field] = list(value) if isinstance(value, tuple) else value
            config = getattr(contender, "config", None)
            if config is not None:
                entry["config"] = {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in vars(config).items()}
            entries.append(entry)
        registry[suite] = entries
    return registry


def data_audit() -> dict:
    from lanthanide_separation.gen7.harness import (
        DEFAULT_SEEDS, assert_fold_integrity, build_folds, load_cohort,
    )
    cohort = load_cohort()
    frame = cohort.frame
    folds = {}
    for seed in DEFAULT_SEEDS:
        plan = build_folds(frame, seed)
        integrity = assert_fold_integrity(frame, plan)
        folds[str(seed)] = {
            "ok": integrity["ok"],
            "test_rows": [int(f.test_index.size) for f in plan],
            "test_extractants": [int(frame.iloc[f.test_index].extractant.nunique()) for f in plan],
            "held_out_chemotypes": [len(f.held_out_chemotypes) for f in plan],
            "model_seeds": [f.model_seed for f in plan],
        }
    return {
        "cohort_fingerprint": cohort.fingerprint,
        "n_rows": int(len(frame)),
        "n_extractants": int(frame.extractant.nunique()),
        "n_ecfp_clusters": int(frame.ecfp_cluster.nunique()),
        "n_chemotypes": int(frame.tanimoto_cluster.nunique()),
        "n_metals": int(frame.metal_symbol.nunique()),
        "n_conditions": int(frame.condition_id.nunique()),
        "n_series": int(frame.series_id.nunique()),
        "target_mean": float(frame.log_D.mean()), "target_sd": float(frame.log_D.std()),
        "largest_extractant_share": float(frame.extractant.value_counts(normalize=True).iloc[0]),
        "largest_chemotype_share": float(frame.tanimoto_cluster.value_counts(normalize=True).iloc[0]),
        "blocks": {k: len(v) for k, v in cohort.blocks.items()},
        "folds": folds,
        "cohort_audit": cohort.data.audit,
    }


def experiment_manifest() -> dict:
    entries = {}
    for directory in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name != "cache"):
        files = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.stat().st_size < 200 * 1 << 20:
                files[str(path.relative_to(directory))] = {
                    "bytes": path.stat().st_size, "sha256": sha256_file(path)[:16]}
        entries[directory.name] = files
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = {"environment.json": environment, "model_registry.json": model_registry,
             "data_audit.json": data_audit, "experiment_manifest.json": experiment_manifest}
    for name, builder in tasks.items():
        if name in args.skip:
            continue
        try:
            payload = builder()
        except Exception as error:
            payload = {"error": f"{type(error).__name__}: {error}"}
        (OUT / name).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
