"""Hash every gen11 artefact, plus the inputs it is only meaningful against.

A manifest that lists only the outputs cannot detect the failure that matters:
an output regenerated against a *different* cohort or a *different* archive.  So
the identity block pins the frozen bundle, the cohort fingerprint, the archive's
own published manifest and the auxiliary feature fingerprint alongside the
artefact digests.

Usage::

    PYTHONPATH=src python scripts/gen11_manifest.py [--verify]

``--verify`` re-hashes everything against the stored manifest and exits non-zero
on any drift, which is what the self-audit calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RUN = REPO_ROOT / "runs" / "gen11_transfer"
MANIFEST = RUN / "manifest.json"
#: Artefacts change; these do not, and a gen11 number is only valid against them.
FROZEN_INPUTS = {
    "frozen_bundle": REPO_ROOT / "dataset with 3D structures" / "dataset.parquet",
    "archive_master_clean": REPO_ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet",
    "archive_manifest": REPO_ROOT / "dataset_all_metals" / "reports" / "manifest.json",
    "gen7_cohort_cache": REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet",
    "gen9_curve_membership": REPO_ROOT / "runs" / "gen9_shape" / "curves" / "curve_membership.parquet",
}
#: gen11's own code.  A result is only reproducible against the module that made it.
SOURCE = sorted((REPO_ROOT / "src" / "lanthanide_separation" / "gen11").glob("*.py"))
SCRIPTS = sorted(REPO_ROOT.glob("scripts/gen11_*.py"))

#: Files above this are hashed but never inlined into a report.
LARGE_BYTES = 50_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry(path: Path) -> dict:
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path),
            "bytes": path.stat().st_size, "large": path.stat().st_size > LARGE_BYTES}


def build() -> dict:
    from lanthanide_separation.gen10.runner import COHORT_FINGERPRINT, prepared_cohort

    cohort = prepared_cohort()
    artefacts = [entry(p) for p in sorted(RUN.rglob("*"))
                 if p.is_file() and p.name != MANIFEST.name and ".log" not in p.suffixes]
    return {
        "run": "gen11_transfer",
        "identity": {
            "cohort_fingerprint": cohort.fingerprint,
            "cohort_fingerprint_expected": COHORT_FINGERPRINT,
            "cohort_rows": int(len(cohort.frame)),
            "cohort_extractants": int(cohort.frame["extractant"].nunique()),
            "cohort_chemotypes": int(cohort.frame["tanimoto_cluster"].nunique()),
        },
        "frozen_inputs": {name: (entry(path) if path.exists() else {"path": str(path),
                                                                    "missing": True})
                          for name, path in FROZEN_INPUTS.items()},
        "source": [entry(p) for p in SOURCE],
        "scripts": [entry(p) for p in SCRIPTS],
        "artefacts": artefacts,
        "counts": {"artefacts": len(artefacts), "source_files": len(SOURCE),
                   "scripts": len(SCRIPTS)},
    }


def verify() -> int:
    if not MANIFEST.exists():
        print("no manifest to verify against")
        return 1
    stored = json.loads(MANIFEST.read_text())
    drift, missing = [], []
    for group in ("frozen_inputs", "source", "scripts", "artefacts"):
        entries = stored[group]
        entries = entries.values() if isinstance(entries, dict) else entries
        for item in entries:
            if item.get("missing"):
                continue
            path = REPO_ROOT / item["path"]
            if not path.exists():
                missing.append(item["path"])
            elif sha256(path) != item["sha256"]:
                drift.append(item["path"])
    current = build()
    if current["identity"]["cohort_fingerprint"] != stored["identity"]["cohort_fingerprint"]:
        drift.append("COHORT FINGERPRINT")
    print(json.dumps({"drifted": drift, "missing": missing,
                      "n_checked": sum(len(stored[g]) for g in
                                       ("source", "scripts", "artefacts"))}, indent=2))
    return 0 if not drift and not missing else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        return verify()
    manifest = build()
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"identity": manifest["identity"], "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
