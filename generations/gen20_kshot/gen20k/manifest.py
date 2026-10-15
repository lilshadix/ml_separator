"""Provenance for every gen20 run: git state, environment, dataset digests, seeds, arguments,
output digests and a log, written as ``manifests/<name>__<label>.json`` and alongside the outputs.

``require_sealed_preregistration`` refuses an outcome-bearing run unless ``PRE_REGISTRATION.md``
carries a sealed SHA-256 footer that matches the file's own content (gen16's convention).
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from . import bootstrap
from .data import MANIFEST_PARQUET, sha256_of

PREREG = bootstrap.G20 / "PRE_REGISTRATION.md"
SEAL_LINE = "SEALED-SHA256:"


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=bootstrap.ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception as exc:  # pragma: no cover
            return f"unavailable: {exc}"
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")), "describe": run("describe", "--always", "--dirty")}


def environment() -> dict:
    import numpy, pandas, scipy, sklearn, torch
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads()}


def dataset_digests() -> dict:
    from gen13sep import paths
    return {"bundle_parquet": str(paths.BUNDLE_PARQUET), "bundle_sha256": sha256_of(paths.BUNDLE_PARQUET),
            "bundle_sha256_pinned": getattr(paths, "BUNDLE_SHA256", None),
            "cohort_manifest_parquet": str(MANIFEST_PARQUET), "cohort_manifest_sha256": sha256_of(MANIFEST_PARQUET)}


def preregistration_state() -> dict:
    if not PREREG.exists():
        return {"exists": False, "sealed": False}
    text = PREREG.read_text(encoding="utf-8")
    lines = text.rstrip("\n").split("\n")
    sealed = lines and lines[-1].startswith(SEAL_LINE)
    body = "\n".join(lines[:-1]) + "\n" if sealed else text
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    recorded = lines[-1].split(":", 1)[1].strip() if sealed else None
    return {"exists": True, "sealed": bool(sealed), "sha256_of_body": digest, "recorded": recorded,
            "valid": bool(sealed and recorded == digest)}


def require_sealed_preregistration() -> dict:
    st = preregistration_state()
    if not st.get("valid"):
        raise RuntimeError("PRE_REGISTRATION.md is not sealed (or its seal does not match); outcome-bearing runs are refused. "
                           "Use --smoke for a labelled smoke run.")
    return st


def seal_preregistration() -> str:
    """Append the SHA-256 footer (idempotent: an existing footer is replaced)."""
    text = PREREG.read_text(encoding="utf-8")
    lines = text.rstrip("\n").split("\n")
    if lines[-1].startswith(SEAL_LINE):
        lines = lines[:-1]
    body = "\n".join(lines) + "\n"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    PREREG.write_text(body + f"{SEAL_LINE} {digest}\n", encoding="utf-8")
    return digest


class Run:
    def __init__(self, name: str, out_dir: Path, *, args: dict | None = None, seeds=None):
        self.name, self.out_dir, self.args, self.seeds = name, Path(out_dir), args or {}, list(seeds or [])
        self.records, self.lines, self.t0 = {}, [], time.time()

    def __enter__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {"name": self.name, "label": self.args.get("label"), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "git": git_state(), "environment": environment(), "dataset": dataset_digests(),
                         "preregistration": preregistration_state(), "seeds": self.seeds, "args": self.args}
        return self

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.0f}s] {msg}"
        print(line, flush=True)
        self.lines.append(line)

    def record(self, key: str, value) -> None:
        self.records[key] = value

    def __exit__(self, exc_type, exc, tb):
        outputs = {}
        for p in sorted(self.out_dir.rglob("*")):
            if p.is_file() and p.suffix in (".parquet", ".csv", ".json") and p.name != "manifest.json":
                outputs[str(p.relative_to(self.out_dir))] = {"sha256": sha256_of(p), "bytes": p.stat().st_size}
        self.manifest.update({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": time.time() - self.t0,
                              "status": "ok" if exc is None else f"error: {exc!r}", "records": self.records,
                              "outputs": outputs, "log": self.lines})
        with open(self.out_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=1, default=str)
        bootstrap.MANIFESTS.mkdir(parents=True, exist_ok=True)
        with open(bootstrap.MANIFESTS / f"{self.name}__{self.args.get('label', 'run')}.json", "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in self.manifest.items() if k != "log"}, fh, indent=1, default=str)
        return False
