"""Stage 7 -- publish the deliverables and hash every artifact."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_paths import (ROOT, RAW_DIR, CLEAN_DIR, AUDIT_DIR, REPORTS_DIR,  # noqa: E402
                       RUN_DIR, ensure_dirs)

#: deliverable name -> source path, relative to the workspace root.
DELIVERABLES = {
    "master_clean.parquet": CLEAN_DIR / "master_clean.parquet",
    "master_clean.csv": CLEAN_DIR / "master_clean.csv",
    "lanthanide_clean.parquet": CLEAN_DIR / "lanthanide_clean.parquet",
    "non_lanthanide_clean.parquet": CLEAN_DIR / "non_lanthanide_clean.parquet",
    "single_component_clean.parquet": CLEAN_DIR / "single_component_clean.parquet",
    "raw_to_canonical_mapping.parquet": AUDIT_DIR / "raw_to_canonical_mapping.parquet",
    "duplicate_groups.parquet": AUDIT_DIR / "duplicate_groups.parquet",
    "conflict_groups.parquet": AUDIT_DIR / "conflict_groups.parquet",
    "condition_conflict_candidates.parquet": AUDIT_DIR / "condition_conflict_candidates.parquet",
    "manual_review_queue.csv": AUDIT_DIR / "manual_review_queue.csv",
    "curve_inventory.parquet": AUDIT_DIR / "curve_inventory.parquet",
    "curve_membership.parquet": ROOT / "intermediate" / "curve_membership.parquet",
    "metal_coverage.csv": REPORTS_DIR / "metal_coverage.csv",
    "extractant_coverage.csv": REPORTS_DIR / "extractant_coverage.csv",
    "extractant_metal_matrix.csv": REPORTS_DIR / "extractant_metal_matrix.csv",
    "field_mapping.csv": REPORTS_DIR / "field_mapping.csv",
    "normalization_log.json": REPORTS_DIR / "normalization_log.json",
    "row_accounting.json": REPORTS_DIR / "row_accounting.json",
    "quality_report.md": REPORTS_DIR / "quality_report.md",
    "schema.md": REPORTS_DIR / "schema.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ensure_dirs()
    entries = {}
    missing = []
    for name, source in DELIVERABLES.items():
        if not source.exists():
            missing.append(name)
            continue
        target = RUN_DIR / name
        shutil.copy2(source, target)
        entries[name] = {
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
            "workspace_path": str(source.relative_to(ROOT.parent)),
        }

    raw_inputs = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(RAW_DIR.glob("*.csv"))
    }

    # The Crossref cache is a real pipeline INPUT: it fills reference_* and
    # doi_primary_corrected on most rows. Without its hash here, a different
    # cache could silently produce a different dataset under an unchanged
    # manifest, so it is pinned exactly like the raw CSVs.
    cache_file = ROOT / "cache" / "crossref_metadata.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text())
        external_inputs = {
            "cache/crossref_metadata.json": {
                "sha256": sha256(cache_file),
                "bytes": cache_file.stat().st_size,
                "dois_cached": len(payload),
                "dois_resolved": sum(1 for v in payload.values() if v.get("status") == "ok"),
                "role": "pipeline input: bibliographic metadata for reference_* columns",
                "refetch_with": "scripts/fetch_crossref.py",
            }
        }
    else:
        external_inputs = {}

    import pandas as pd
    import rdkit
    import numpy

    manifest = {
        "dataset": "Separation Archive for Elements -- multi-metal clean build",
        "workspace": str(ROOT),
        "deliverables_directory": str(RUN_DIR),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": numpy.__version__,
            "rdkit": rdkit.__version__,
        },
        "raw_inputs": raw_inputs,
        "raw_input_count": len(raw_inputs),
        "external_inputs": external_inputs,
        "artifacts": entries,
        "missing_artifacts": missing,
        "pipeline": [
            "stage01_ingest.py", "stage02_normalize.py", "stage03_dedup.py",
            "stage04_series.py", "stage05_export.py", "stage06_report.py",
            "stage06b_field_mapping.py", "stage06c_quality.py", "stage07_manifest.py",
        ],
        "determinism": "no wall-clock, RNG or hash-seed dependence; "
                       "group ids are assigned by sorted identity hash",
    }
    for target in (REPORTS_DIR / "manifest.json", RUN_DIR / "manifest.json"):
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"published {len(entries)} artifacts to {RUN_DIR}")
    if missing:
        print("MISSING:", ", ".join(missing))


if __name__ == "__main__":
    main()
