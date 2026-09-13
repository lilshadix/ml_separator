#!/usr/bin/env python
"""Verify the Gen12 and Gen12.2 manifests across the move into ``generations/``.

Commit 155dc6c moved ``gen12_eu_pred/`` and ``gen12_2_eu_pred/`` from the repository
root into ``generations/``.  Their manifests hash nearly every file in those
directories, code included, so the few lines that had to change for the code to keep
resolving the same targets are recorded in ``generations/RELOCATION.json`` as exact
text substitutions.  That record is pinned here by its SHA-256 (``RELOCATION_SHA256``)
and its entries may name only the files in ``RELOCATION_PATHS``, so a new or widened
substitution needs a visible change to this script.  This script shows that nothing
outside the pinned substitutions changed:

(0) each manifest is byte-identical to ``git show <commit_before>:<old path>``, so no
    hash in it can have been re-stamped; without that git history the run fails;
(a) for every relocation entry (each naming a file in ``RELOCATION_PATHS``), each
    ``after`` block occurs exactly once in the file, and undoing the substitutions gives
    bytes that match every hash the manifest records for that file;
(b) every other file listed in either manifest matches its recorded hashes as it
    stands.  Drift that already existed at the pre-move commit (the file is
    byte-identical to ``git show <commit_before>:<old path>``) is reported and does not
    fail; any other drift fails, and so does a missing file.
In (a) and (b) an entry recorded by size only (no content hash) must also be
byte-identical to its ``<commit_before>`` blob.

The hashes are always read live from the manifests; RELOCATION.json stores only text.
``original_bytes(path)`` returns the pre-relocation bytes of a file, for guards that
compare files byte-for-byte with a manifest.

Usage (from the repository root)::

    .venv/bin/python generations/verify_relocation.py [--root PATH]

Exit status 0 when every check passes, 1 otherwise.  Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
RELOCATION_FILE = "generations/RELOCATION.json"
# SHA-256 of the reviewed RELOCATION.json.  Any change to the record (a new entry, a widened
# substitution) must change this constant too, so it cannot pass unseen.
RELOCATION_SHA256 = "b70d5b6cc308096dfb9a74559b3104cd0f832c24b4ad89a1595d0562c0780200"
# the only files a relocation entry may name: the four .py files the move had to edit
RELOCATION_PATHS = frozenset({
    "generations/gen12_eu_pred/gen12eu/paths.py",
    "generations/gen12_eu_pred/scripts/gen12_manifest.py",
    "generations/gen12_2_eu_pred/gen122/paths.py",
    "generations/gen12_2_eu_pred/scripts/g122_self_audit.py",
})
# manifest (repository-relative, after the move) -> its generation directory before the move
MANIFESTS = {
    "generations/gen12_eu_pred/manifests/manifest.json": "gen12_eu_pred",
    "generations/gen12_2_eu_pred/manifests/manifest.json": "gen12_2_eu_pred",
}


# ---- relocation records ------------------------------------------------------ #

@lru_cache(maxsize=None)
def _relocation(root: str) -> dict:
    data = (Path(root) / RELOCATION_FILE).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != RELOCATION_SHA256:
        raise ValueError(f"SHA-256 {digest} is not the RELOCATION_SHA256 pinned in verify_relocation.py "
                         f"({RELOCATION_SHA256}), so the record was edited")
    return json.loads(data.decode("utf-8"))


@lru_cache(maxsize=None)
def _entries(root: str) -> dict:
    return {entry["path"]: entry for entry in _relocation(root)["entries"]}


def _reverse(data: bytes, substitutions: list) -> tuple:
    """Undo ``substitutions`` on ``data``.

    Returns ``(restored_bytes, [])``, or ``(None, problems)`` when an ``after`` block does
    not occur exactly once or two blocks overlap.  Nothing is undone partially.
    """
    spans, problems = [], []
    for number, substitution in enumerate(substitutions, start=1):
        after = substitution["after"].encode("utf-8")
        before = substitution["before"].encode("utf-8")
        count = data.count(after)
        if count != 1:
            problems.append(f"substitution {number}: 'after' block occurs {count} times, expected exactly 1")
            continue
        start = data.find(after)
        spans.append((start, start + len(after), before))
    spans.sort()
    for (_, end, _), (start, _, _) in zip(spans, spans[1:]):
        if start < end:
            problems.append("substitution 'after' blocks overlap")
    if problems:
        return None, problems
    pieces, cursor = [], 0
    for start, end, before in spans:
        pieces.append(data[cursor:start])
        pieces.append(before)
        cursor = end
    pieces.append(data[cursor:])
    return b"".join(pieces), []


def original_bytes(path, root=None) -> bytes:
    """The bytes ``path`` had before the relocation edits.

    ``path`` may be absolute or relative to the repository root.  A file with an entry in
    RELOCATION.json has its substitutions undone; any other file is returned as it stands.
    If an entry's substitutions cannot be undone cleanly (an ``after`` block missing or
    repeated) the current bytes are returned unchanged, so a byte-for-byte guard reports
    the file as drifted instead of crashing; ``verify_relocation.py`` reports why.
    Raises ValueError when RELOCATION.json does not match ``RELOCATION_SHA256``, so a
    guard built on it fails closed.
    """
    root = Path(root).resolve() if root is not None else DEFAULT_ROOT
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    data = path.read_bytes()
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError:
        return data
    entry = _entries(str(root)).get(relative)
    if entry is None:
        return data
    restored, problems = _reverse(data, entry["substitutions"])
    return data if problems else restored


# ---- hashing and history ----------------------------------------------------- #

def _hash_failures(data: bytes, meta: dict) -> list:
    """The recorded hashes ``data`` does not match.  Unrecognised keys fail closed."""
    failures, recognised = [], 0
    for key, expected in meta.items():
        if key == "bytes":
            actual = len(data)
        elif key == "blake2b_128":
            actual = hashlib.blake2b(data, digest_size=16).hexdigest()
        elif key == "sha256":
            actual = hashlib.sha256(data).hexdigest()
        else:
            failures.append(f"unrecognised manifest key {key!r}")
            continue
        recognised += 1
        if actual != expected:
            failures.append(key)
    if not recognised:
        failures.append("no recognised hash recorded")
    return failures


def _git_has_commit(root: Path, commit: str) -> bool:
    try:
        result = subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                                capture_output=True, check=False)
    except OSError:
        return False
    return result.returncode == 0


def _git_blob(root: Path, commit: str, path: str):
    """The file at ``commit``, or None if it did not exist there (or git failed)."""
    try:
        result = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{path}"],
                                capture_output=True, check=False)
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _generation_dir(manifest_rel: str) -> str:
    return Path(manifest_rel).parents[1].as_posix()


# ---- verification ------------------------------------------------------------ #

def verify(root: Path) -> int:
    failures: list = []
    preexisting: list = []
    print(f"Relocation verification, root {root}")
    try:
        relocation = _relocation(str(root))
    except (OSError, ValueError) as error:
        print(f"cannot use {RELOCATION_FILE}: {error}\n\nRESULT: FAIL")
        return 1
    if relocation.get("schema_version") != 1:
        print(f"{RELOCATION_FILE}: unsupported schema_version {relocation.get('schema_version')!r}\n\nRESULT: FAIL")
        return 1
    commit_before = relocation["commit_before"]
    entries = relocation["entries"]
    git_ok = _git_has_commit(root, commit_before)
    print(f"{RELOCATION_FILE}: commit_before {commit_before}, move_commit {relocation.get('move_commit')}, "
          f"{len(entries)} entries")
    print(f"git history at {commit_before}: "
          + ("available" if git_ok else "UNAVAILABLE, so no drift can be shown to predate the move"))

    # The manifests are frozen.  Pin each one byte-for-byte to commit_before, or a re-stamped hash
    # would hide any edit.
    print(f"\nmanifests, pinned byte-for-byte to {commit_before}")
    artefacts = {}
    for manifest_rel in MANIFESTS:
        manifest_path = root / manifest_rel
        if not manifest_path.is_file():
            failures.append(f"{manifest_rel}: manifest missing")
            artefacts[manifest_rel] = {}
            continue
        artefacts[manifest_rel] = json.loads(manifest_path.read_text(encoding="utf-8"))["artefacts"]
        manifest_before = f"{MANIFESTS[manifest_rel]}/manifests/manifest.json"
        if not git_ok:
            problem = f"cannot be pinned to {commit_before}: git history unavailable"
        elif manifest_path.read_bytes() != _git_blob(root, commit_before, manifest_before):
            problem = f"differs from its {commit_before} blob ({manifest_before}), so it was re-stamped or edited"
        else:
            print(f"  PASS  {manifest_rel}\n        identical to {commit_before}:{manifest_before}")
            continue
        failures.append(f"{manifest_rel}: {problem}")
        print(f"  FAIL  {manifest_rel}\n        {problem}")

    # (a) files with a relocation entry
    print("\n(a) manifest-listed files with a relocation entry: substitutions undone, then hashed")
    covered, seen = set(), set()
    for entry in entries:
        rel, manifest_rel, key = entry["path"], entry["manifest"], entry["manifest_key"]
        problems, detail = [], ""
        if rel in seen:
            problems.append("duplicate entry for this path")
        seen.add(rel)
        if rel not in RELOCATION_PATHS:
            problems.append("not one of the RELOCATION_PATHS a relocation entry may name")
        if manifest_rel not in MANIFESTS:
            problems.append(f"manifest {manifest_rel} is not one this script verifies")
        else:
            covered.add((manifest_rel, key))
            if f"{_generation_dir(manifest_rel)}/{key}" != rel:
                problems.append(f"manifest_key {key!r} does not name this path")
            if f"{MANIFESTS[manifest_rel]}/{key}" != entry["path_before"]:
                problems.append(f"path_before {entry['path_before']!r} is not the pre-move location")
        meta = artefacts.get(manifest_rel, {}).get(key)
        if meta is None:
            problems.append("not listed in its manifest")
        if not (root / rel).is_file():
            problems.append("file missing")
        if not problems:
            restored, problems = _reverse((root / rel).read_bytes(), entry["substitutions"])
            if restored is not None:
                bad = _hash_failures(restored, meta)
                blob = _git_blob(root, commit_before, entry["path_before"]) if git_ok else None
                identical = blob is not None and blob == restored
                if not bad and not {"blake2b_128", "sha256"} & set(meta) and not identical:
                    bad = [f"size-only entry and content differs from {commit_before}"]
                if not bad:
                    detail = (f"{len(entry['substitutions'])} substitution(s) undone; "
                              f"{', '.join(meta)} match the manifest"
                              + (f"; identical to {commit_before}:{entry['path_before']}" if identical else ""))
                elif identical:
                    detail = f"pre-existing drift ({', '.join(bad)})"
                    preexisting.append(f"{rel}: restored bytes fail {', '.join(bad)}, "
                                       f"as the file already did at {commit_before}")
                else:
                    problems.append(f"restored bytes fail {', '.join(bad)}")
        if problems:
            failures.extend(f"{rel}: {problem}" for problem in problems)
            print(f"  FAIL  {rel}")
            for problem in problems:
                print(f"        {problem}")
        else:
            print(f"  PASS  {rel}\n        {detail}")

    # (b) every other manifest-listed file
    print("\n(b) every other manifest-listed file, hashed as it stands")
    for manifest_rel, before_dir in MANIFESTS.items():
        generation = _generation_dir(manifest_rel)
        listed = artefacts[manifest_rel]
        n_entry = sum(1 for key in listed if (manifest_rel, key) in covered)
        n_ok = n_pre = n_new = n_size_only = 0
        for key, meta in listed.items():
            if (manifest_rel, key) in covered:
                continue
            path = root / generation / key
            data = path.read_bytes() if path.is_file() else None
            bad = ["missing"] if data is None else _hash_failures(data, meta)
            if not bad and not {"blake2b_128", "sha256"} & set(meta):
                # recorded by size only, which proves nothing: compare the content with commit_before
                n_size_only += 1
                if not git_ok or _git_blob(root, commit_before, f"{before_dir}/{key}") != data:
                    bad = [f"size-only entry and content differs from {commit_before}" if git_ok else
                           "size-only entry and no git history to compare its content with"]
            if not bad:
                n_ok += 1
                continue
            rel = f"{generation}/{key}"
            if git_ok and data is not None and _git_blob(root, commit_before, f"{before_dir}/{key}") == data:
                # unchanged since the pre-move commit: it predates the move
                n_pre += 1
                preexisting.append(f"{rel}: {', '.join(bad)} (unchanged since {commit_before})")
                continue
            n_new += 1
            failures.append(f"{rel}: new drift without a relocation entry ({', '.join(bad)})" if git_ok else
                            f"{rel}: drift ({', '.join(bad)}); git history unavailable to show it predates the move")
        print(f"  {manifest_rel}: {len(listed)} listed, {n_entry} checked in (a), {n_ok} match "
              f"({n_size_only} size-only, compared with {commit_before}), "
              f"{n_pre} pre-existing drift, {n_new} new drift")

    print()
    if preexisting:
        print(f"Pre-existing drift, reported but not failed ({len(preexisting)}):")
        for line in preexisting:
            print(f"  {line}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for line in failures:
            print(f"  {line}")
    print(f"RESULT: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify the Gen12 and Gen12.2 manifests across the "
                                                 "relocation into generations/.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="repository root (default: the parent of the generations/ directory "
                             "holding this script)")
    args = parser.parse_args(argv)
    return verify(args.root.resolve())


if __name__ == "__main__":
    sys.exit(main())
