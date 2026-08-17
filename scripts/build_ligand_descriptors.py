#!/usr/bin/env python3
"""Build the extended 2D ligand-descriptor table for every unique extractant.

Reads ``canonical_smiles`` from the frozen dataset, computes the full RDKit
descriptor list plus the hand-crafted DGA/amide descriptors from
:mod:`lanthanide_separation.ligand_descriptors`, and writes one row per unique
SMILES to a parquet file with a sibling manifest recording versions, columns
and the parquet SHA-256.

RDKit is imported inside :func:`main` so this module imports cleanly in an
environment without RDKit; run it with an interpreter that has RDKit, e.g.::

    /Users/lilshadix/miniforge3/envs/lanth/bin/python scripts/build_ligand_descriptors.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lanthanide_separation.ligand_descriptors import (  # noqa: E402
    HAND_CRAFTED_DESCRIPTOR_NAMES,
    HAND_CRAFTED_PREFIX,
    RDKIT_PREFIX,
    SMILES_COLUMN,
    build_descriptor_table,
    descriptor_columns,
)


DATASET_DIR = REPO_ROOT / "dataset with 3D structures"
DEFAULT_DATASET = DATASET_DIR / "dataset.parquet"
DEFAULT_OUTPUT = DATASET_DIR / "ligand_2d_descriptors.parquet"

PREVIEW_SMILES: tuple[tuple[str, str], ...] = (
    ("TODGA", "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"),
    ("N-methyl-N-octyl DGA", "CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSON path; defaults to <output stem>.manifest.json.",
    )
    parser.add_argument("--smiles-column", default=SMILES_COLUMN)
    parser.add_argument(
        "--keep-uninformative",
        action="store_true",
        help="Keep RDKit descriptor columns that are constant or all-NaN.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preview(frame: pd.DataFrame, smiles: str) -> str:
    matches = frame.loc[frame[SMILES_COLUMN] == smiles]
    if matches.empty:
        return "    (not present in the dataset)"
    row = matches.iloc[0]
    lines = []
    for column in descriptor_columns(frame, "hc"):
        lines.append(f"    {column[len(HAND_CRAFTED_PREFIX):]:<40s} {row[column]:g}")
    for name in ("MolWt", "MolLogP", "NumRotatableBonds", "FractionCSP3"):
        column = f"{RDKIT_PREFIX}{name}"
        if column in frame.columns:
            lines.append(f"    {name:<40s} {row[column]:.4g}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    # Imported here so that the module itself is importable without RDKit.
    import rdkit
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    output_path = args.output.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_path.with_name(f"{output_path.stem}.manifest.json")
    )

    source = pd.read_parquet(dataset_path, columns=[args.smiles_column])
    smiles = source[args.smiles_column].dropna().astype(str)
    frame = build_descriptor_table(smiles, drop_uninformative=not args.keep_uninformative)
    if args.smiles_column != SMILES_COLUMN:
        frame = frame.rename(columns={args.smiles_column: SMILES_COLUMN})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    rd_columns = descriptor_columns(frame, "rd")
    hc_columns = descriptor_columns(frame, "hc")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rdkit_version": rdkit.__version__,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_rows": int(len(source)),
        "smiles_column": args.smiles_column,
        "parquet_path": str(output_path),
        "parquet_sha256": sha256_file(output_path),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "n_rdkit_columns": len(rd_columns),
        "n_hand_crafted_columns": len(hc_columns),
        "hand_crafted_descriptor_names": list(HAND_CRAFTED_DESCRIPTOR_NAMES),
        "columns": [str(column) for column in frame.columns],
        "dropped_rdkit_columns": list(frame.attrs.get("dropped_rdkit_columns", [])),
        "constant_hand_crafted_columns": list(
            frame.attrs.get("constant_hand_crafted_columns", [])
        ),
        "unparsed_smiles": list(frame.attrs.get("unparsed_smiles", [])),
        "nan_counts": {
            str(column): int(frame[column].isna().sum())
            for column in frame.columns
            if column != SMILES_COLUMN and frame[column].isna().any()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"dataset:            {dataset_path} ({len(source)} rows)")
    print(f"unique ligands:     {len(frame)}")
    print(f"columns:            {frame.shape[1]} (1 key + {len(rd_columns)} rdkit + "
          f"{len(hc_columns)} hand-crafted)")
    print(f"dropped rdkit cols: {len(manifest['dropped_rdkit_columns'])} "
          f"{manifest['dropped_rdkit_columns']}")
    print(f"constant hc cols:   {manifest['constant_hand_crafted_columns']}")
    print(f"unparsed SMILES:    {len(manifest['unparsed_smiles'])}")
    print(f"parquet:            {output_path}")
    print(f"manifest:           {manifest_path}")
    for label, value in PREVIEW_SMILES:
        print(f"\n{label}: {value}")
        print(_preview(frame, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
