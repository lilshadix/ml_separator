"""Frozen pretrained molecular embeddings for the 190 extractants.

The brief's World A asks whether hand-engineered descriptors simply cannot carry a
ligand's absolute extraction level while a representation pretrained on a large
unlabelled corpus can.  With only 152 supervised extractants that is exactly the
kind of question a frozen encoder is supposed to answer: the encoder learned
structural chemistry from millions of molecules, and we spend none of our 5,248
rows teaching it what a molecule is.

Three encoders, chosen because their weights are downloadable without auth and
they are the ones the low-data literature actually benchmarks:

``chemberta``   ``DeepChem/ChemBERTa-77M-MTR`` — RoBERTa over SMILES, 77 M
                molecules from PubChem, 384-dim.  The MTR checkpoint is
                multi-task-regression pretrained, which the ChemBERTa-2 paper
                reports beats the MLM checkpoint on regression transfer.
``chemberta_mlm`` ``DeepChem/ChemBERTa-77M-MLM`` — the masked-LM checkpoint, kept
                as the control for that claim.
``molformer``   ``ibm-research/MoLFormer-XL-both-10pct`` — linear-attention
                transformer, 1.1 B molecules (ZINC + PubChem), 768-dim.

Pooling matters and is not free: mean-pooling over non-pad tokens is used
throughout (the ``[CLS]`` vector of an MLM checkpoint is not trained to summarise
anything).  Both poolings are written so the choice is an ablation, not an
assumption.

Nothing here reads ``log_D``: the table is keyed on ``canonical_smiles`` and built
from structure alone, so it cannot leak the target and can be built once and
frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
OUT_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_pretrained_embeddings.parquet"

ENCODERS: dict[str, str] = {
    "chemberta": "DeepChem/ChemBERTa-77M-MTR",
    "chemberta_mlm": "DeepChem/ChemBERTa-77M-MLM",
    "molformer": "ibm-research/MoLFormer-XL-both-10pct",
}


def embed(smiles: list[str], model_id: str, *, batch: int = 16) -> tuple[np.ndarray, np.ndarray, int]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # ``deterministic_eval`` is a MoLFormer-only argument (it switches its linear
    # attention from the sampled feature map to the exact one); passing it to a
    # RoBERTa checkpoint is a TypeError, so it is offered and not assumed.
    try:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                          deterministic_eval=True).eval()
    except TypeError:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
    mean_parts, cls_parts = [], []
    with torch.no_grad():
        for start in range(0, len(smiles), batch):
            chunk = smiles[start:start + batch]
            encoded = tokenizer(chunk, padding=True, truncation=True, max_length=512,
                                return_tensors="pt")
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            mean_parts.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy())
            cls_parts.append(hidden[:, 0, :].numpy())
    mean = np.concatenate(mean_parts, axis=0)
    return mean, np.concatenate(cls_parts, axis=0), int(mean.shape[1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    frame = pd.read_parquet(DATASET_PATH, columns=["canonical_smiles"])
    smiles = sorted(frame["canonical_smiles"].astype(str).unique())
    print(f"{len(smiles)} unique canonical SMILES", flush=True)

    table = pd.DataFrame({"canonical_smiles": smiles})
    manifest: dict = {"n_smiles": len(smiles), "encoders": {}}
    for key in args.encoders:
        model_id = ENCODERS[key]
        print(f"  {key} <- {model_id}", flush=True)
        try:
            mean, cls, dim = embed(smiles, model_id)
        except Exception as error:
            print(f"    FAILED: {type(error).__name__}: {error}", flush=True)
            manifest["encoders"][key] = {"model_id": model_id, "error": str(error)[:300]}
            continue
        for i in range(dim):
            table[f"emb__{key}__mean_{i:03d}"] = mean[:, i]
        for i in range(dim):
            table[f"emb__{key}__cls_{i:03d}"] = cls[:, i]
        manifest["encoders"][key] = {
            "model_id": model_id, "dim": dim,
            "mean_norm": float(np.linalg.norm(mean, axis=1).mean()),
            "n_distinct_rows": int(len(np.unique(mean.round(5), axis=0))),
        }
        print(f"    dim {dim}, {manifest['encoders'][key]['n_distinct_rows']}"
              f"/{len(smiles)} distinct vectors", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out} ({table.shape[0]}x{table.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
