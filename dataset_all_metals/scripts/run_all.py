"""Run the whole SAE cleaning pipeline end to end.

Deterministic and idempotent: re-running it must reproduce byte-identical
cleaned files.  ``--verify-determinism`` proves that by hashing the artifacts,
running everything a second time, and comparing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sae_paths import CLEAN_DIR, AUDIT_DIR, REPORTS_DIR, INTERMEDIATE_DIR  # noqa: E402

STAGES = ("stage01_ingest.py", "stage02_normalize.py", "stage03_dedup.py",
          "stage04_series.py", "stage05_export.py", "stage06_report.py",
          "stage06b_field_mapping.py", "stage06c_quality.py", "stage07_manifest.py")


def run(stage: str) -> None:
    print(f"\n=== {stage} ===", flush=True)
    result = subprocess.run([sys.executable, str(HERE / stage)], text=True)
    if result.returncode != 0:
        raise SystemExit(f"{stage} failed with exit code {result.returncode}")


def fingerprint() -> dict[str, str]:
    """Hash every produced artifact except the manifest, which embeds paths.

    ``INTERMEDIATE_DIR`` is included because one published deliverable
    (``curve_membership.parquet``) lives there; leaving it out meant the
    determinism check silently did not cover a file the manifest ships.
    """
    digests = {}
    for directory in (CLEAN_DIR, AUDIT_DIR, REPORTS_DIR, INTERMEDIATE_DIR):
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            digests[str(path)] = digest.hexdigest()
    return digests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-determinism", action="store_true",
                        help="run the pipeline twice and require identical artifacts")
    args = parser.parse_args()

    for stage in STAGES:
        run(stage)

    if args.verify_determinism:
        first = fingerprint()
        print("\n=== second pass (determinism check) ===")
        for stage in STAGES:
            run(stage)
        second = fingerprint()
        changed = sorted(k for k in first if first[k] != second.get(k))
        added = sorted(set(second) - set(first))
        removed = sorted(set(first) - set(second))
        report = {"artifacts_compared": len(first), "changed": changed,
                  "added": added, "removed": removed,
                  "deterministic": not (changed or added or removed)}
        (REPORTS_DIR / "determinism_check.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2))
        if not report["deterministic"]:
            raise SystemExit("pipeline is NOT deterministic")
        print("\nOK: two full runs produced byte-identical artifacts")


if __name__ == "__main__":
    main()
