#!/usr/bin/env python3
"""Losslessly compress oversized run artifacts for transport.

Cluster export and GitHub both reject files at or above 100 MB.  Run artifacts
are covered by SHA-256 contracts (``artifact_hashes.json``), so compression
must be exactly reversible: this tool gzips oversized files with a fixed
header (``mtime=0``) so output is deterministic, verifies that decompression
reproduces the original bytes (and the recorded contract hash when one
exists), and writes a ``transport_manifest.json`` next to each compressed
file mapping original name, size and SHA-256 to the compressed counterpart.

Usage:
    python scripts/compress_run_artifacts.py RUN_DIR [RUN_DIR ...]
        [--threshold-mb 95] [--delete-original] [--restore]

``--restore`` reverses the operation: every ``*.gz`` listed in a manifest is
decompressed and re-verified against the recorded original SHA-256.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

CHUNK = 1 << 20
MANIFEST_NAME = "transport_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_sha256_of_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_hash(run_dir: Path, name: str) -> str | None:
    contract = run_dir / "artifact_hashes.json"
    if not contract.is_file():
        return None
    try:
        recorded = json.loads(contract.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = recorded.get(name)
    return str(value) if isinstance(value, str) else None


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / MANIFEST_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": "transport_manifest.v1", "files": {}}


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    path = run_dir / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def compress_file(source: Path, delete_original: bool) -> dict:
    original_sha = file_sha256(source)
    original_size = source.stat().st_size
    target = source.with_name(source.name + ".gz")
    tmp = target.with_name(target.name + ".tmp")
    with source.open("rb") as src, tmp.open("wb") as raw:
        # mtime=0 keeps the gzip container byte-deterministic across reruns.
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as dst:
            shutil.copyfileobj(src, dst, CHUNK)
    if gzip_sha256_of_decompressed(tmp) != original_sha:
        tmp.unlink()
        raise RuntimeError(f"round-trip verification failed for {source}")
    tmp.replace(target)

    contract_sha = _contract_hash(source.parent, source.name)
    if contract_sha is not None and contract_sha != original_sha:
        raise RuntimeError(
            f"{source.name}: on-disk bytes do not match artifact_hashes.json "
            f"({original_sha} != {contract_sha}); refusing to continue"
        )

    entry = {
        "original": source.name,
        "original_sha256": original_sha,
        "original_bytes": original_size,
        "compressed": target.name,
        "compressed_sha256": file_sha256(target),
        "compressed_bytes": target.stat().st_size,
        "matches_artifact_hashes_json": contract_sha is not None,
    }
    if delete_original:
        source.unlink()
    return entry


def restore_run_dir(run_dir: Path) -> int:
    manifest = _load_manifest(run_dir)
    restored = 0
    for entry in manifest.get("files", {}).values():
        compressed = run_dir / entry["compressed"]
        original = run_dir / entry["original"]
        if original.exists() or not compressed.is_file():
            continue
        tmp = original.with_name(original.name + ".tmp")
        with gzip.open(compressed, "rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
        if file_sha256(tmp) != entry["original_sha256"]:
            tmp.unlink()
            raise RuntimeError(f"restore verification failed for {compressed}")
        tmp.replace(original)
        restored += 1
        print(f"restored {original} ({entry['original_bytes']} bytes, sha256 verified)")
    return restored


def process_run_dir(run_dir: Path, threshold_bytes: int, delete_original: bool) -> int:
    manifest = _load_manifest(run_dir)
    compressed = 0
    for source in sorted(run_dir.glob("*.csv")):
        if source.stat().st_size < threshold_bytes:
            continue
        entry = compress_file(source, delete_original=delete_original)
        manifest["files"][entry["original"]] = entry
        compressed += 1
        ratio = entry["compressed_bytes"] / entry["original_bytes"]
        print(
            f"{source}: {entry['original_bytes'] / 1e6:.1f} MB -> "
            f"{entry['compressed_bytes'] / 1e6:.1f} MB ({ratio:.0%}), "
            "round-trip sha256 verified"
        )
    if compressed:
        _write_manifest(run_dir, manifest)
    return compressed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--threshold-mb", type=float, default=95.0)
    parser.add_argument("--delete-original", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args(argv)

    threshold_bytes = int(args.threshold_mb * 1e6)
    touched = 0
    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            parser.error(f"not a directory: {run_dir}")
        if args.restore:
            touched += restore_run_dir(run_dir)
        else:
            touched += process_run_dir(run_dir, threshold_bytes, args.delete_original)
    if not touched:
        print("nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
