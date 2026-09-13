"""Script 1 of DESIGN.md section 11: build the extraction-systems database.

Bundle SHA check, ingest (``gen18proc.ingest.ingest_corpus``), the literature entries of
section 3.7 (``gen18proc.literature``), the phase-behaviour literature merge
(``PHASE_LITERATURE``, addenda/ORCHESTRATOR_prefit_20260913.md), validation of every written
entry, ``systems/*`` and ``results/audit/ingest_audit.json`` plus ``manifest.json``.

Usage (from the repository root, one process):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_build_db.py \\
        [--systems-dir systems] [--seed 18]

No wall-clock value is written.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc.ingest import ingest_corpus  # noqa: E402
from gen18proc.literature import PHASE_LITERATURE, literature_entries  # noqa: E402
from gen18proc.systems import (  # noqa: E402
    entry_to_json, load_registry, load_system, registry_row, validate_entry, write_system,
)


def git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=paths.REPO_ROOT, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def write_manifest(out_dir: Path, inputs: dict[str, Path], outputs: list[Path], seed: int,
                   args: dict, *, name: str = "manifest.json", script: str | None = None) -> Path:
    """``<out_dir>/<name>``: input and output SHA-256s, seed, arguments, git HEAD; no clock."""

    def rel(p: Path) -> str:
        root = paths.REPO_ROOT
        return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)

    manifest = {
        "script": script or Path(__file__).name, "seed": seed, "arguments": args,
        "git_head": git_head(),
        "inputs": {k: {"path": rel(p), "sha256": paths.sha256_of(p)} for k, p in inputs.items()},
        "outputs": {str(p.relative_to(paths.G18_ROOT)): paths.sha256_of(p)
                    for p in sorted(outputs)},
    }
    path = out_dir / name
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    systems_dir = Path(args.systems_dir)
    if not systems_dir.is_absolute():
        systems_dir = paths.G18_ROOT / systems_dir
    systems_dir.mkdir(parents=True, exist_ok=True)

    print(f"bundle sha256 {paths.assert_bundle_unchanged()}")
    audit = ingest_corpus(systems_dir, seed=args.seed)
    print(f"ingest: {audit.n_records} records, {audit.n_systems} systems, "
          f"{audit.n_publications} publications, {audit.n_loading_series_publication_aware} "
          f"loading series, {audit.n_unit_slip_rows} unit-slip rows")

    # literature entries (section 3.7)
    lit = literature_entries()
    for entry in lit:
        errors = [v for v in validate_entry(entry) if v.level == "error"]
        if errors:
            raise SystemExit(f"{entry.system_id}: literature entry fails validation: "
                             + "; ".join(f"{v.path}: {v.message}" for v in errors))
        write_system(entry, systems_dir / f"{entry.system_id}.json")
        print(f"literature entry written: {entry.system_id} ({entry.name})")

    # phase-behaviour literature merged into corpus entries
    for sid, fields in PHASE_LITERATURE.items():
        path = systems_dir / f"{sid}.json"
        if not path.exists():
            print(f"PHASE_LITERATURE: {sid} not in the database; skipped")
            continue
        entry = load_system(path)
        entry = dataclasses.replace(entry, phase=dataclasses.replace(entry.phase, **fields))
        write_system(entry, path)
        print(f"PHASE_LITERATURE merged into {sid}: {', '.join(fields)}")

    # validate everything as written; rebuild registry / INDEX with the literature rows
    warnings_by_rule: dict[str, int] = {}
    rows = []
    for path in sorted(systems_dir.glob("sys_*.json")):
        entry = load_system(path)          # raises on any error
        for v in validate_entry(entry):
            if v.level == "warning":
                rule = v.message.split(":")[0]
                warnings_by_rule[rule] = warnings_by_rule.get(rule, 0) + 1
        rows.append(registry_row(entry_to_json(entry)))
    n_warn = sum(warnings_by_rule.values())
    with open(systems_dir / "registry.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"schema_version": "gen18.1", "systems": rows}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    from gen18proc.ingest import _write_index
    _write_index(systems_dir, rows)
    reg = load_registry(systems_dir)
    print(f"validated {len(reg)} entries ({n_warn} warnings: "
          f"{dict(sorted(warnings_by_rule.items()))}); origins: "
          f"{reg['origin'].value_counts().to_dict()}")

    audit = dataclasses.replace(audit, n_literature_entries=len(lit))
    out = paths.RESULTS_AUDIT_DIR
    audit_path = out / "ingest_audit.json"
    payload = audit.to_json()
    payload["validator_warnings_by_rule"] = dict(sorted(warnings_by_rule.items()))
    with open(audit_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    outputs = sorted(systems_dir.glob("*.json")) + sorted(systems_dir.glob("*.csv")) + [
        systems_dir / "INDEX.md", audit_path]
    write_manifest(out, {"bundle": paths.BUNDLE_PARQUET,
                         "provenance": paths.GEN6_PROVENANCE_PARQUET},
                   outputs, args.seed, vars(args))
    print(f"wrote {audit_path} and {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
