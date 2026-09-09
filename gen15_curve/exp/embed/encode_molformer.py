"""Add MoLFormer-XL to ``ligand_embeddings.parquet`` without re-encoding the ChemBERTa columns.

The first pass recorded MoLFormer as failed with an ``OSError`` naming ``pytorch_model.bin``.  That
diagnosis was wrong: the repo does ship ``model.safetensors``, and the real cause was that the
187 MB weight download kept stalling on this machine, after which ``transformers`` fell back to
looking for a ``.bin`` and reported the fallback's absence.  Retrying ``hf_hub_download`` until the
blob completed fixed it, and the model then loads on transformers 5.16.1 with
``trust_remote_code=True`` (five ``lm_head.*`` keys are unexpected for ``AutoModel``, which is
expected when loading the encoder out of an MLM checkpoint).

Usage:  python encode_molformer.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from encode import cohort_smiles, encode                          # noqa: E402

TAG = "molformer"
MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
PARQUET = HERE / "ligand_embeddings.parquet"
MANIFEST = HERE / "embeddings_manifest.json"


def main() -> None:
    import transformers
    smiles = cohort_smiles()
    df = pd.read_parquet(PARQUET)
    assert list(df.index) == smiles, "cohort SMILES changed since the ChemBERTa pass"
    t0 = time.time()
    mean, cls, info = encode(MODEL_ID, smiles)
    info["seconds"] = round(time.time() - t0, 1)
    add = {}
    for k in range(mean.shape[1]):
        add[f"{TAG}__mean__{k:04d}"] = mean[:, k]
    for k in range(cls.shape[1]):
        add[f"{TAG}__cls__{k:04d}"] = cls[:, k]
    new = pd.DataFrame(add, index=df.index).astype("float32")
    df = pd.concat([df.drop(columns=[c for c in df.columns if c.startswith(TAG + "__")]), new],
                   axis=1)
    df.to_parquet(PARQUET)

    m = json.loads(MANIFEST.read_text())
    m["transformers"] = transformers.__version__
    m["torch"] = torch.__version__
    m["encoders"][TAG] = info
    m["encoders"][TAG]["note"] = ("loaded with trust_remote_code=True after resuming the stalled "
                                  "187 MB model.safetensors download; the first pass's "
                                  "'pytorch_model.bin' OSError was that stall, not an API mismatch")
    m.pop("failed", None)
    m["n_columns"] = int(df.shape[1])
    MANIFEST.write_text(json.dumps(m, indent=2))
    print(f"[molformer] dim={info['dim']} distinct={info['n_distinct_mean_rows']} "
          f"{info['seconds']}s -> {df.shape}", flush=True)


if __name__ == "__main__":
    main()
