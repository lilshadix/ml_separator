"""Run manifests: what was run, on what data, with which code, and did it finish.

Every gen2–gen5 study in this repo recorded *some* provenance, but each recorded
a different subset, and the one time it mattered — the gen5 run that silently
dropped two of four regimes — nothing in the artifacts disclosed it.  gen6
therefore has one manifest type with a **fail-closed required-key list**: a run
that cannot name its dataset hash, its split definition, its held-out chemistry
and its exact row ids does not get a ``_SUCCESS.json``.

Three ideas carry the design:

* **Hash what the science depends on, not what the filesystem happens to hold.**
  A parquet re-written with a different compression level is byte-different and
  scientifically identical, so frames are hashed through a canonical form
  (sorted columns, stable row order, fixed float formatting) rather than by file
  bytes.  File bytes are hashed too, separately, for transport integrity.
* **Record the split, not just the seed.** A seed only reproduces a split if the
  cohort, the label strings and the fold algorithm are also unchanged — three
  things that have silently moved in this project before.  So the manifest
  stores the realised partition: test row ids, test extractants, test
  super-clusters, per fold.
* **A validation report is part of the run, not a post-hoc script.**
  :func:`validate_run` re-reads what was written and checks it against the
  manifest, so a truncated artifact is caught by the run that produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: Keys a gen6 run must record before it may write ``_SUCCESS.json``.
#: Ordered by the brief's Part I list; a runner that forgets one fails closed.
REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "dataset_path",
    "dataset_file_sha256",
    "source_table_sha256",
    "feature_registry_sha256",
    "code_sha256",
    "split_definition",
    "chemistry_cluster_definition",
    "provenance_state",
    "model_seed",
    "split_seeds",
    "preprocessing",
    "folds",
)

#: Keys that must be present *inside* each entry of ``manifest["folds"]``.
REQUIRED_FOLD_KEYS: tuple[str, ...] = (
    "fold", "split_seed", "test_row_ids_sha256", "test_extractants", "test_superclusters",
    "train_extractants_by_arm", "train_superclusters_by_arm", "n_test_rows", "n_train_rows_by_arm",
)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #

def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed (the dataset parquet is 100+ MB)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a JSON-serialisable object under a canonical encoding."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default))


def sha256_frame(frame: pd.DataFrame, *, sort_rows_by: Sequence[str] | None = None) -> str:
    """Content hash of a DataFrame that ignores column order and dtype width.

    Two frames holding the same science must hash the same even if one was
    round-tripped through parquet (which can turn ``int64`` into ``Int64`` and
    reorder columns).  Floats are rendered at 12 significant digits: enough to
    catch a real change, loose enough to survive a parquet round trip.

    ``sort_rows_by`` makes the hash invariant to row order as well; pass the
    identity columns (e.g. ``["row_id"]``) when row order is not meaningful.
    """
    work = frame
    if sort_rows_by:
        missing = [c for c in sort_rows_by if c not in work.columns]
        if missing:
            raise KeyError(f"sort_rows_by columns absent from frame: {missing}")
        work = work.sort_values(list(sort_rows_by), kind="stable")
    # Duplicate labels would make `work[column]` return a DataFrame, and the join
    # below would then iterate its column NAMES rather than its values — the hash
    # would describe the schema and ignore the data.  Fail closed instead.
    labels = list(map(str, work.columns))
    duplicated = sorted({label for label in labels if labels.count(label) > 1})
    if duplicated:
        raise ValueError(f"cannot hash a frame with duplicate column labels: {duplicated[:5]}")
    digest = hashlib.sha256()
    digest.update(f"shape={work.shape[0]}x{work.shape[1]}\n".encode())
    for column in sorted(labels):
        series = work[column]
        digest.update(f"col={column}\n".encode())
        if pd.api.types.is_float_dtype(series):
            rendered = series.map(lambda v: "nan" if pd.isna(v) else f"{float(v):.12g}")
        elif pd.api.types.is_bool_dtype(series):
            rendered = series.map(lambda v: "" if pd.isna(v) else ("1" if bool(v) else "0"))
        elif pd.api.types.is_integer_dtype(series):
            rendered = series.map(lambda v: "nan" if pd.isna(v) else str(int(v)))
        else:
            rendered = series.map(lambda v: "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        digest.update("\x1f".join(rendered.astype(str)).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def code_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    """``{relative-ish path: sha256}`` for every existing file in ``paths``.

    Directories are walked for ``*.py``.  A missing path is recorded as
    ``"MISSING"`` rather than dropped, because a silently absent module is
    exactly the kind of drift this table exists to catch.
    """
    out: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                if "__pycache__" in child.parts:
                    continue
                out[str(child)] = sha256_file(child)
        elif path.is_file():
            out[str(path)] = sha256_file(path)
        else:
            out[str(path)] = "MISSING"
    return out


def git_state(repo_root: str | Path) -> dict[str, Any]:
    """Best-effort git commit / branch / dirty flag; never raises."""
    root = Path(repo_root)

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(("git", "-C", str(root), *args), capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "dirty_files": None if status is None else [line[3:] for line in status.splitlines()][:50],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"not JSON-serialisable: {type(value)!r}")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

@dataclass
class RunManifest:
    """Accumulates everything a gen6 run must be able to prove afterwards."""

    layer: str
    run_id: str
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    entries: dict[str, Any] = field(default_factory=dict)

    # -- recording ---------------------------------------------------------- #
    def record(self, key: str, value: Any) -> "RunManifest":
        self.entries[key] = value
        return self

    def record_many(self, mapping: Mapping[str, Any]) -> "RunManifest":
        for key, value in mapping.items():
            self.record(key, value)
        return self

    def record_dataset(
        self, *, dataset_path: str | Path, source_frame: pd.DataFrame,
        descriptor_path: str | Path | None = None, descriptor_frame: pd.DataFrame | None = None,
    ) -> "RunManifest":
        self.record("dataset_path", str(dataset_path))
        self.record("dataset_file_sha256", sha256_file(dataset_path))
        self.record("source_table_sha256", sha256_frame(source_frame))
        self.record("source_table_shape", list(source_frame.shape))
        if descriptor_path is not None:
            self.record("descriptor_path", str(descriptor_path))
            self.record("descriptor_file_sha256",
                        sha256_file(descriptor_path) if Path(descriptor_path).exists() else "MISSING")
        if descriptor_frame is not None:
            self.record("descriptor_table_sha256", sha256_frame(descriptor_frame))
        return self

    def record_code(self, paths: Iterable[str | Path], *, repo_root: str | Path | None = None) -> "RunManifest":
        """Hash the source files, and flag any that changed after the run started.

        The hashes are read from disk when this is called, which is usually near the
        *end* of a run — so a file edited mid-run would be certified in a state the
        process never executed.  That is not hypothetical: it happened during this
        layer's own development.  Any file whose mtime is later than the manifest's
        creation time is listed in ``code_modified_after_start``, so a reader can
        see that the certification is unreliable for that file.
        """
        hashes = code_hashes(paths)
        started = datetime.fromisoformat(self.created_utc).timestamp()
        suspect = []
        for name in hashes:
            path = Path(name)
            if path.is_file() and path.stat().st_mtime > started:
                suspect.append(name)
        self.record("code_modified_after_start", sorted(suspect))
        if suspect:
            self.record("code_hash_warning",
                        "one or more source files were modified after this run started; their "
                        "recorded hash may not be the code that ran")
        self.record("code_sha256", hashes)
        self.record("code_sha256_combined", sha256_json(hashes))
        if repo_root is not None:
            self.record("git", git_state(repo_root))
        self.record("python", {"version": platform.python_version(), "platform": platform.platform()})
        return self

    def record_features(self, *, feature_sets: Mapping[str, Sequence[str]]) -> "RunManifest":
        """Feature registry: the exact ordered column list behind every arm name.

        Order matters — ``max_features`` samples columns, so a permuted column
        list is a different model.  The hash therefore keeps the order.
        """
        registry = {name: list(cols) for name, cols in feature_sets.items()}
        self.record("feature_registry", {name: {"n_columns": len(cols), "sha256": sha256_json(cols)}
                                         for name, cols in registry.items()})
        self.record("feature_registry_sha256", sha256_json(registry))
        self.record("feature_registry_columns", registry)
        return self

    def record_split(
        self, *, definition: Mapping[str, Any], folds: Sequence[Mapping[str, Any]],
    ) -> "RunManifest":
        """The *realised* partition, not just the recipe that produced it."""
        for entry in folds:
            missing = [k for k in REQUIRED_FOLD_KEYS if k not in entry]
            if missing:
                raise ValueError(f"fold record is missing required keys {missing}: {sorted(entry)}")
        self.record("split_definition", dict(definition))
        self.record("folds", [dict(entry) for entry in folds])
        self.record("split_sha256", sha256_json({"definition": dict(definition),
                                                 "folds": [dict(f) for f in folds]}))
        return self

    def record_chemistry(self, *, definition: Mapping[str, Any]) -> "RunManifest":
        self.record("chemistry_cluster_definition", dict(definition))
        return self

    def record_provenance(self, *, state: Mapping[str, Any]) -> "RunManifest":
        self.record("provenance_state", dict(state))
        return self

    def record_preprocessing(self, steps: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> "RunManifest":
        self.record("preprocessing", steps if isinstance(steps, Mapping) else list(steps))
        return self

    # -- output ------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "run_id": self.run_id, "created_utc": self.created_utc,
                **self.entries}

    def missing_required(self) -> list[str]:
        return [key for key in REQUIRED_MANIFEST_KEYS if key not in self.entries]

    def write(self, output_dir: str | Path) -> dict[str, Any]:
        """Write ``manifest.json`` and ``artifact_hashes.json``; return the manifest."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        (out / "manifest.json").write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
        # A first pass, so a run that dies before validation still ships hashes.
        # `write_success` refreshes it afterwards, because artifacts written after
        # this point — validation.json, and the log's final lines — would otherwise
        # be missing from the very record that is supposed to cover them.
        refresh_artifact_hashes(out)
        return payload


def refresh_artifact_hashes(output_dir: str | Path) -> dict[str, Any]:
    """(Re)write ``artifact_hashes.json`` over everything currently in the directory."""
    out = Path(output_dir)
    artifacts = artifact_hashes(out, exclude={"artifact_hashes.json"})
    (out / "artifact_hashes.json").write_text(
        json.dumps(artifacts, indent=2, default=_json_default) + "\n")
    return artifacts


def artifact_hashes(output_dir: str | Path, *, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """SHA-256 and byte size of every file written into a run directory."""
    out = Path(output_dir)
    skip = set(exclude)
    files: dict[str, Any] = {}
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(out))
        if rel in skip:
            continue
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"n_files": len(files), "files": files}


# --------------------------------------------------------------------------- #
# Validation and completion
# --------------------------------------------------------------------------- #

def validate_run(
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    required_artifacts: Iterable[str] = (),
    checks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-read the run directory and check it against its own manifest.

    Returns the validation record (also written to ``validation.json``).  It is
    *fail-closed*: any missing required manifest key, missing artifact, empty
    artifact or failed extra check sets ``ok = False``.  ``checks`` carries
    run-specific assertions the runner already evaluated, as
    ``{name: bool | {"ok": bool, ...}}``.
    """
    out = Path(output_dir)
    missing_keys = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    artifacts: dict[str, Any] = {}
    for name in required_artifacts:
        path = out / name
        exists = path.is_file()
        artifacts[name] = {"exists": exists, "bytes": path.stat().st_size if exists else 0}
    missing_artifacts = [n for n, a in artifacts.items() if not a["exists"]]
    empty_artifacts = [n for n, a in artifacts.items() if a["exists"] and a["bytes"] == 0]

    normalised_checks: dict[str, Any] = {}
    failed_checks: list[str] = []
    for name, value in (checks or {}).items():
        # Fail closed on the *type*, not just the value.  `bool("FAIL")` is True and
        # `bool(0.0001)` is True, so a check reporting a string verdict or a small
        # error magnitude used to pass silently.  Only a real bool (or a mapping
        # whose "ok" is a real bool) can pass.
        if isinstance(value, Mapping):
            raw = value.get("ok", None)
            ok = raw is True or (isinstance(raw, (bool, np.bool_)) and bool(raw))
            if raw is not None and not isinstance(raw, (bool, np.bool_)):
                ok = False
                value = {**value, "_rejected": f"non-boolean ok: {raw!r}"}
            normalised_checks[name] = dict(value)
        elif isinstance(value, (bool, np.bool_)):
            ok = bool(value)
            normalised_checks[name] = {"ok": ok}
        else:
            ok = False
            normalised_checks[name] = {"ok": False,
                                       "_rejected": f"check value must be a bool or a mapping "
                                                    f"with a boolean 'ok'; got {type(value).__name__}"}
        if not ok:
            failed_checks.append(name)

    record = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out),
        "missing_manifest_keys": missing_keys,
        "artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "empty_artifacts": empty_artifacts,
        "checks": normalised_checks,
        "failed_checks": failed_checks,
        "ok": not (missing_keys or missing_artifacts or empty_artifacts or failed_checks),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    return record


def write_success(
    output_dir: str | Path, *, manifest: Mapping[str, Any], validation: Mapping[str, Any],
) -> Path | None:
    """Write ``_SUCCESS.json`` — but only for a run that actually validated.

    Returns the path, or ``None`` when validation failed (in which case a
    ``_FAILED.json`` is written instead, so the failure is discoverable by the
    same directory listing that would have found the success marker).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer": manifest.get("layer"),
        "run_id": manifest.get("run_id"),
        "created_utc": manifest.get("created_utc"),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_file_sha256": manifest.get("dataset_file_sha256"),
        "source_table_sha256": manifest.get("source_table_sha256"),
        "code_sha256_combined": manifest.get("code_sha256_combined"),
        "split_sha256": manifest.get("split_sha256"),
        "feature_registry_sha256": manifest.get("feature_registry_sha256"),
        "provenance_state": manifest.get("provenance_state"),
        "validation_ok": bool(validation.get("ok")),
    }
    if not validation.get("ok"):
        failed = out / "_FAILED.json"
        failed.write_text(json.dumps(
            {**payload, "missing_manifest_keys": validation.get("missing_manifest_keys"),
             "missing_artifacts": validation.get("missing_artifacts"),
             "failed_checks": validation.get("failed_checks")},
            indent=2, default=_json_default) + "\n")
        return None
    success = out / "_SUCCESS.json"
    success.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    # Hash everything that exists now, including validation.json — the record that
    # gates this very marker — and excluding only the two files that cannot cover
    # themselves.
    refresh_artifact_hashes(out)
    return success
