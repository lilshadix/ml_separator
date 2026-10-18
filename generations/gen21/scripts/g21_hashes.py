#!/usr/bin/env python
"""Record SHA-256 digests of the inputs, the fold plans and the cache (PROTOCOL.md section 7)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from g21 import cache as C  # noqa: E402
from g21 import paths  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import fold_record, folds_for  # noqa: E402

bench = D.load_bench()
rec = {"cohort_manifest_parquet": "generations/gen13_separation/manifests/cohort_exact.parquet",
       "cohort_manifest_sha256": paths.sha256_file(paths.ROOT / "generations/gen13_separation/manifests/cohort_exact.parquet"),
       "logD_matrix_sha256": hashlib.sha256(np.ascontiguousarray(np.nan_to_num(bench.Y, nan=-999.0)).tobytes()).hexdigest(),
       "n_cells": int(bench.Y.shape[0]), "folds": {}, "cache": {}}
for d in ("BP", "BPG"):
    fr = fold_record(bench.frame, folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS)))
    rec["folds"][d] = {"seeds": list(paths.DEV_SEEDS), "n_rows": int(len(fr)),
                       "sha256": hashlib.sha256(fr.to_csv(index=False, lineterminator="\n").encode()).hexdigest()}
for p in sorted(paths.CACHE.glob("*.pkl")):
    rec["cache"][p.name] = paths.sha256_file(p)
code = {}
for p in sorted((paths.G21 / "g21").glob("*.py")):
    code[p.name] = paths.sha256_file(p)
rec["code_at_protocol"] = code
out = paths.MANIFESTS / "inputs_sha256.json"
out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in rec.items() if k not in ("cache", "code_at_protocol")}, indent=1))
