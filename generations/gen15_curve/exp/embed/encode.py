"""Encode every distinct extractant SMILES in the gen13 cohort with pretrained chemical LMs.

Rebuilds the table recorded by ``dataset with 3D structures/ligand_pretrained_embeddings.manifest.json``
but for the 90 SMILES that actually appear in the gen15 value bench.  For each model we keep both the
mean-pooled last hidden state over non-padding tokens and the CLS/first-token vector, because the two
are genuinely different objects for a masked LM (MLM heads sit on the token stream, the pooled CLS of
a regression-tuned model such as ChemBERTa-77M-MTR carries the property head's view).

Output: ``ligand_embeddings.parquet`` (index = SMILES, columns ``<tag>__<pool>__<k>``) plus
``embeddings_manifest.json``.
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
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))

MODELS = {
    "chemberta_mtr": "DeepChem/ChemBERTa-77M-MTR",
    "chemberta_mlm": "DeepChem/ChemBERTa-77M-MLM",
    "molformer": "ibm-research/MoLFormer-XL-both-10pct",
}


def cohort_smiles() -> list[str]:
    from gen15 import valuebench as V
    bench = V.load()
    sm = sorted(set(bench.frame.extractant.astype(str).tolist()))
    return sm


@torch.no_grad()
def encode(model_id: str, smiles: list[str], batch: int = 8) -> tuple[np.ndarray, np.ndarray, dict]:
    from transformers import AutoModel, AutoTokenizer
    kw = {}
    if "MoLFormer" in model_id:
        kw = {"trust_remote_code": True}
    tok = AutoTokenizer.from_pretrained(model_id, **kw)
    mdl = AutoModel.from_pretrained(model_id, **kw)
    mdl.eval()
    mean_rows, cls_rows = [], []
    for i in range(0, len(smiles), batch):
        chunk = smiles[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        out = mdl(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")})
        h = out.last_hidden_state                       # (b, t, d)
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        mean_rows.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        cls_rows.append(h[:, 0, :].cpu().numpy())
    mean = np.concatenate(mean_rows, 0)
    cls = np.concatenate(cls_rows, 0)
    info = {"model_id": model_id, "dim": int(mean.shape[1]),
            "max_len_tokens": int(max(len(tok(s)["input_ids"]) for s in smiles)),
            "mean_norm": float(np.linalg.norm(mean, axis=1).mean()),
            "n_distinct_mean_rows": int(len(np.unique(np.round(mean, 5), axis=0)))}
    return mean, cls, info


def main() -> None:
    import transformers
    smiles = cohort_smiles()
    print(f"[encode] {len(smiles)} distinct extractant SMILES", flush=True)
    cols: dict[str, np.ndarray] = {}
    manifest = {"transformers": transformers.__version__, "torch": torch.__version__,
                "n_smiles": len(smiles), "encoders": {}, "failed": {}}
    for tag, mid in MODELS.items():
        t0 = time.time()
        try:
            mean, cls, info = encode(mid, smiles)
        except Exception as exc:                                    # noqa: BLE001
            manifest["failed"][tag] = {"model_id": mid, "error": f"{type(exc).__name__}: {exc}"[:800]}
            print(f"[encode] FAILED {tag} {mid}: {type(exc).__name__}: {exc}"[:400], flush=True)
            continue
        for k in range(mean.shape[1]):
            cols[f"{tag}__mean__{k:04d}"] = mean[:, k]
        for k in range(cls.shape[1]):
            cols[f"{tag}__cls__{k:04d}"] = cls[:, k]
        info["seconds"] = round(time.time() - t0, 1)
        manifest["encoders"][tag] = info
        print(f"[encode] {tag} dim={info['dim']} distinct={info['n_distinct_mean_rows']} "
              f"{info['seconds']}s", flush=True)
    df = pd.DataFrame(cols, index=pd.Index(smiles, name="smiles")).astype("float32")
    df.to_parquet(HERE / "ligand_embeddings.parquet")
    manifest["n_columns"] = int(df.shape[1])
    (HERE / "embeddings_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[encode] wrote {df.shape} -> {HERE / 'ligand_embeddings.parquet'}", flush=True)


if __name__ == "__main__":
    main()
