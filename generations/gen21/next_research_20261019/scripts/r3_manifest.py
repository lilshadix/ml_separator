#!/usr/bin/env python
"""Manifests for round 3: environment versions, input data and fold-plan digests, cache content
digests (content, not wall-clock metadata), and the code tree.  Writes manifests/*.json.

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_manifest.py [--code-only]
"""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from r3 import paths  # noqa: E402


def content_digest(obj) -> str:
    """Order-stable digest of a cache record's content (numpy arrays by bytes, no timings)."""
    h = hashlib.sha256()

    def walk(o):
        if isinstance(o, dict):
            for k in sorted(o, key=str):
                if k == "seconds":
                    continue
                h.update(str(k).encode()); walk(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(b"[")
            for v in o:
                walk(v)
            h.update(b"]")
        elif isinstance(o, np.ndarray):
            h.update(str(o.dtype).encode()); h.update(np.ascontiguousarray(o).tobytes())
        else:
            h.update(repr(o).encode())
    walk(obj)
    return h.hexdigest()


def main() -> int:
    code_only = "--code-only" in sys.argv
    paths.MAN.mkdir(parents=True, exist_ok=True)
    code = {str(p.relative_to(paths.R3)).replace("\\", "/"): paths.sha256_file(p)
            for sub in ("r3", "scripts", "tests") for p in sorted((paths.R3 / sub).glob("*.py"))}
    code_g21 = {str(p.relative_to(paths.G21)).replace("\\", "/"): paths.sha256_file(p) for p in sorted((paths.G21 / "g21").glob("*.py"))}
    (paths.MAN / "code_sha256.json").write_text(json.dumps({"round3": code, "g21_imported": code_g21}, indent=1), encoding="utf-8")
    if code_only:
        print("code manifest written")
        return 0
    import numpy, pandas, scipy, sklearn, torch  # noqa: E401
    env = {"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__,
           "pandas": pandas.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
           "interpreter": sys.executable, "threads_note": "OMP/LOKY capped by gen20k.bootstrap (4 / 2); torch 4 threads per process"}
    (paths.MAN / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    from gen20k import data as D
    from gen20k.splits import fold_record, folds_for
    bench = D.load_bench()
    inputs = {"cohort_manifest_sha256": paths.sha256_file(paths.ROOT / "generations/gen13_separation/manifests/cohort_exact.parquet"),
              "logD_matrix_sha256": hashlib.sha256(np.ascontiguousarray(np.nan_to_num(bench.Y, nan=-999.0)).tobytes()).hexdigest(),
              "n_cells": int(bench.Y.shape[0]), "folds": {}, "cache_content": {}}
    for d in ("BP", "BPG"):
        fr = fold_record(bench.frame, folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS)))
        inputs["folds"][d] = {"seeds": list(paths.DEV_SEEDS), "n_rows": int(len(fr)),
                              "sha256": hashlib.sha256(fr.to_csv(index=False, lineterminator="\n").encode()).hexdigest()}
    for p in sorted(paths.CACHE.glob("*.pkl")):
        with open(p, "rb") as fh:
            inputs["cache_content"][p.name] = content_digest(pickle.load(fh))
    (paths.MAN / "inputs_sha256.json").write_text(json.dumps(inputs, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in inputs.items() if k != "cache_content"}, indent=1))
    print(len(inputs["cache_content"]), "cache files digested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
