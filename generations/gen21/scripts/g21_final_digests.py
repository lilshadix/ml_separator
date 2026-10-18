#!/usr/bin/env python
"""SHA-256 of the code, predictions and result tables at the end of a round
(manifests/round1_sha256.json).  The protocol-time digests in inputs_sha256.json are left as recorded."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g21 import paths  # noqa: E402

rec = {}
for sub, pat in (("g21", "*.py"), ("scripts", "*.py"), ("tests", "*.py"), ("predictions", "*.parquet"), ("results", "**/*.csv"),
                 ("results", "**/*.json")):
    for p in sorted((paths.G21 / sub).glob(pat)):
        rec[str(p.relative_to(paths.G21)).replace("\\", "/")] = paths.sha256_file(p)
out = paths.MANIFESTS / "round1_sha256.json"
out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(len(rec), "files ->", out)
