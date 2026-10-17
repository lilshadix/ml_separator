"""Run provenance for gen20.1 scripts: git state, environment, dataset digests, seeds, arguments,
records, output digests and a log, written to ``generations/gen20_1/manifests/<name>__<label>.json``
(and ``<out_dir>/manifest.json`` when an output directory is given).  Re-implemented: gen20k's
``Run.__exit__`` writes into gen20_kshot/manifests and is forbidden (IMPLEMENTATION_PLAN.md §8)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import bootstrap
from .env import environment, git_state
from .seal import preregistration_state


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_digests() -> dict:
    from gen20k.data import MANIFEST_PARQUET
    from gen13sep import paths
    return {"bundle_parquet": Path(paths.BUNDLE_PARQUET).relative_to(bootstrap.ROOT).as_posix(),
            "bundle_sha256": sha256_file(paths.BUNDLE_PARQUET),
            "bundle_sha256_pinned": paths.BUNDLE_SHA256,
            "cohort_manifest_parquet": Path(MANIFEST_PARQUET).relative_to(bootstrap.ROOT).as_posix(),
            "cohort_manifest_sha256": sha256_file(MANIFEST_PARQUET),
            "bench_pkl": bootstrap.BENCH_PKL.relative_to(bootstrap.ROOT).as_posix(),
            "bench_pkl_sha256": sha256_file(bootstrap.BENCH_PKL),
            "bench_pkl_bytes": bootstrap.BENCH_PKL.stat().st_size}


class Run:
    """``with Run("g201_counts", label="S3", args={...}) as run: run.log(...); run.record(k, v)``."""

    def __init__(self, name: str, *, label: str = "run", out_dir=None, args: dict | None = None,
                 seeds=None, manifest_dir=None):
        self.name, self.label = name, label
        self.out_dir = bootstrap.assert_under_g201(out_dir) if out_dir is not None else None
        self.manifest_dir = bootstrap.assert_under_g201(manifest_dir or bootstrap.MANIFESTS)
        self.args, self.seeds = dict(args or {}), list(seeds or [])
        self.records: dict = {}
        self.lines: list[str] = []
        self.t0 = time.time()
        self.manifest: dict = {}

    def __enter__(self):
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {"name": self.name, "label": self.label,
                         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "git": git_state(), "environment": environment(), "dataset": dataset_digests(),
                         "preregistration": preregistration_state(bootstrap.PREREG),
                         "seeds": self.seeds, "args": self.args}
        return self

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        self.lines.append(line)

    def record(self, key: str, value) -> None:
        self.records[key] = value

    def outputs(self, paths) -> dict:
        out = {}
        for p in paths:
            p = Path(p)
            if p.is_file():
                try:
                    key = str(p.resolve().relative_to(bootstrap.G201.resolve())).replace("\\", "/")
                except ValueError:
                    key = str(p.resolve())
                out[key] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        return out

    def __exit__(self, exc_type, exc, tb):
        outputs = {}
        if self.out_dir is not None:
            outputs = self.outputs(sorted(q for q in self.out_dir.rglob("*")
                                          if q.suffix in (".parquet", ".csv", ".json") and q.name != "manifest.json"))
        self.manifest.update({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "seconds": time.time() - self.t0,
                              "status": "ok" if exc is None else f"error: {exc!r}",
                              "environment_at_exit": {"torch_loaded": environment()["torch_loaded"]},
                              "records": self.records, "outputs": outputs, "log": self.lines})
        if self.out_dir is not None:
            with open(self.out_dir / "manifest.json", "w", encoding="utf-8", newline="\n") as fh:
                json.dump(self.manifest, fh, indent=1, default=str)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_dir / f"{self.name}__{self.label}.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.manifest, fh, indent=1, default=str)
        return False
