"""Deployable Gen13 separation predictor.

Fits the ensemble on all 521 cells (no hold-out) and predicts, for new (SMILES, conditions)
rows, the centred lanthanide-axis curve and every pairwise log SF(A/B) = log D(A) - log D(B).

    # fit once (writes gen13_separation/models/deploy.joblib)
    .venv/Scripts/python.exe generations/gen13_separation/scripts/g13_predict.py fit [--arms direct@lean,physics@lean,lowrank1@lean,lowrank2@lean]

    # predict: CSV with columns smiles, any cond__* columns (missing -> NaN), optional DENTATE,
    #          optional measured_A, measured_B, measured_logSF (one pair for calibration)
    .venv/Scripts/python.exe generations/gen13_separation/scripts/g13_predict.py predict --input new.csv --output pred.csv

The default configuration is `V2_BAG4@lean` (direct row model + physics-basis, rank-1 and rank-2
curves, all on conditions + physchem + donors + coordination).  Blocks that do not transfer across
laboratories (ECFP, the 206 RDKit descriptors) and a conditions-only member are deliberately out:
see DECISION_REPORT.md §5a.

What the model can and cannot do: zero-shot on chemically unseen extractants the macro MAE over
all pairs is about 0.47 log units, and about 0.54 when the held-out chemistry's laboratories are
also removed from training (the honest number for genuinely new chemistry from a new group);
about 0.19 for neighbouring lanthanides; a single measured pair of the new extractant reduces that to
about 0.25 when that pair is the widest available (La against Lu; the conditional adapter) and to
about 0.36 for a random pair.  The ``max_train_tanimoto`` column says how far the query is from
the training chemistry; ``suggested_first_pair`` says which pair to measure first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "gen12_2_eu_pred"))
sys.path.insert(0, str(ROOT.parent / "gen12_eu_pred"))

from gen13sep import paths  # noqa: E402
from gen13sep.cohort import build_cohort  # noqa: E402
from gen13sep.features import CONTINUOUS_CONDITIONS, DONOR_COLUMNS, PHYSCHEM_COLUMNS, build_features  # noqa: E402
from gen13sep.fewshot import adapt_basis_shift, adapt_rescale  # noqa: E402
from gen13sep.fewshot_stage2 import (adapt_conditional, centred_residual,  # noqa: E402
                                     greedy_support_design, residual_covariance)
from gen13sep.metals import ATOMIC_NUMBER, LANTHANIDES  # noqa: E402
from gen13sep.models import BagArm, BlockSubsetArm, DirectRowArm, FitContext, LowRankArm, PhysicsBasisArm  # noqa: E402
from gen13sep.basis import physics_basis_matrix  # noqa: E402

MODEL_DIR = paths.GEN13_ROOT / "models"
COND = ("COND", "MASSACT")
LEAN = ("COND", "MASSACT", "PHYSCHEM", "DONORS", "COORD")
ARM_FACTORY = {
    "direct@cond": lambda: BlockSubsetArm(DirectRowArm(), COND, name="direct@cond"),
    "direct@lean": lambda: BlockSubsetArm(DirectRowArm(), LEAN, name="direct@lean"),
    "physics@lean": lambda: BlockSubsetArm(PhysicsBasisArm(("radius", "radius_sq")), LEAN, name="physics@lean"),
    "lowrank2@lean": lambda: BlockSubsetArm(LowRankArm(rank=2), LEAN, name="lowrank2@lean"),
    "lowrank1@lean": lambda: BlockSubsetArm(LowRankArm(rank=1), LEAN, name="lowrank1@lean"),
}


def fit(arms: list[str]) -> None:
    cohort = build_cohort("exact")
    features = build_features(cohort)
    blocks = ["COND", "MASSACT", "PHYSCHEM", "DONORS", "ECFP", "LIG2D", "COORD"]
    X = features.matrix(blocks).to_numpy(float)
    offsets, start = {}, 0
    for b in blocks:
        n = len(features.blocks[b]); offsets[b] = np.arange(start, start + n); start += n
    fp = features.matrix(["ECFP"]).to_numpy(float)
    ctx = FitContext(X_train=X, Y_train=cohort.target_matrix, groups_train=cohort.frame["chemotype"].to_numpy(),
                     fingerprints_train=fp, seed=42, extra={"blocks": offsets})
    bag = BagArm([ARM_FACTORY[a]() for a in arms], name="deploy")
    bag.fit(ctx)
    MODEL_DIR.mkdir(exist_ok=True)
    # residual covariance between metals for the one-measurement update: built once from the
    # cross-validated (held-out) curves of the same configuration, never from in-sample fits
    cov_source = paths.PREDICTION_DIR / "B_v2" / "curves" / "V2_BAG_MIX3.parquet"
    residual_cov = None
    if cov_source.exists():
        cv = pd.read_parquet(cov_source)
        Yc = cohort.frame.set_index("cell_id")[[f"logD__{m}" for m in LANTHANIDES]].reindex(cv["cell_id"]).to_numpy(float)
        Cc = cv[[f"c__{m}" for m in LANTHANIDES]].to_numpy(float)
        residual_cov = residual_covariance(np.vstack([centred_residual(Yc[i], Cc[i]) for i in range(len(cv))]))
    all_pairs = [(a, b) for a in range(14) for b in range(a + 1, 14)]
    design = greedy_support_design(residual_cov, all_pairs, 3) if residual_cov is not None else []
    payload = {
        "bag": bag, "arms": arms, "blocks": blocks, "offsets": {k: v.tolist() for k, v in offsets.items()},
        "columns": {b: list(features.blocks[b]) for b in blocks},
        "train_extractants": sorted(cohort.frame["extractant"].unique()),
        "train_fps": {s: fp[i] for s, i in zip(cohort.frame["extractant"], range(len(fp)))},
        "cohort_fingerprint": cohort.fingerprint(),
        "cond_medians": features.frame[list(features.blocks["COND"])].median().to_dict(),
        "residual_cov": residual_cov, "residual_cov_source": str(cov_source.relative_to(paths.REPO_ROOT / "generations")) if residual_cov is not None else None,
        "suggested_pairs": [(LANTHANIDES[a], LANTHANIDES[b]) for a, b in design],
    }
    joblib.dump(payload, MODEL_DIR / "deploy.joblib")
    with open(MODEL_DIR / "deploy.json", "w", encoding="utf-8") as fh:
        json.dump({k: payload[k] for k in ("arms", "blocks", "cohort_fingerprint", "residual_cov_source", "suggested_pairs")}
                  | {"n_cells": int(len(cohort.frame))}, fh, indent=2)
    print("fitted", arms, "on", len(cohort.frame), "cells ->", MODEL_DIR / "deploy.joblib")


def _ligand_features(smiles: str, columns: dict) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from gen122 import coordination
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES {smiles!r}")
    out: dict = {}
    fp = np.zeros(2048)
    for i in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048).GetOnBits():
        fp[i] = 1.0
    for c, v in zip(columns["ECFP"], fp):
        out[c] = v
    phys = {"MolWt": Descriptors.MolWt(mol), "TPSA": Descriptors.TPSA(mol), "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol), "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol), "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "RingCount": Descriptors.RingCount(mol), "FractionCSP3": Descriptors.FractionCSP3(mol), "MolLogP": Descriptors.MolLogP(mol)}
    out.update(phys)
    coord = coordination.descriptors_for(smiles)
    out.update({k: float(v) for k, v in coord.items()})
    # donor census approximated from the coordination donor counts (Architector census unavailable)
    donors = {
        "chem__donor__O(amide_carbonyl)": coord.get("coord__donor__n_O_amide_carbonyl", 0.0),
        "chem__donor__O(ether)": coord.get("coord__donor__n_O_ether", 0.0),
        "chem__donor__N(aromatic)": coord.get("coord__donor__n_N_aromatic", 0.0),
        "chem__donor__N(amine)": coord.get("coord__donor__n_N_amine", 0.0),
        "chem__donor__S(donor)": coord.get("coord__donor__n_soft_S", 0.0),
        "chem__donor__O(hydroxyl)": coord.get("coord__donor__n_O_hydroxyl", 0.0),
        "chem__donor__O(ester_carbonyl)": coord.get("coord__donor__n_O_ester_carbonyl", 0.0),
        "chem__donor__O(carbonyl)": coord.get("coord__donor__n_O_ketone", 0.0),
    }
    dent = float(np.clip(round(coord.get("coord__arm__estimated_denticity_per_pocket", 3.0)), 2, 8))
    donors["chem__donor__n_total"] = float(sum(v for v in donors.values()))
    donors["chem__dentate"] = dent
    donors["chem__core_cn"] = 9.0
    donors["chem__n_ligands"] = float(np.clip(round(9.0 / dent), 1, 4))
    donors["chem__n_fill"] = float(max(0.0, 9.0 - donors["chem__n_ligands"] * dent))
    out.update(donors)
    out["_fp"] = fp; out["_dentate"] = dent
    return out


def _mass_action(cond: dict, dentate: float, core_cn: float) -> dict:
    logs = {}
    for col in CONTINUOUS_CONDITIONS:
        v = cond.get(col, np.nan)
        logs[col] = np.log10(v) if (v is not None and np.isfinite(v) and v > 0) else np.nan
    out = {f"massact__log10_{c}": logs[c] for c in CONTINUOUS_CONDITIONS}
    ll, lh = logs["cond__extractant_concentration_M"], logs["cond__acid_concentration_M"]
    out["massact__logL_x_DENTATE"] = ll * dentate
    out["massact__logL_x_coreCN"] = ll * core_cn
    out["massact__logL_x_logH"] = ll * lh
    return out


def predict(input_csv: Path, output_csv: Path, adapter: str = "conditional") -> None:
    payload = joblib.load(MODEL_DIR / "deploy.joblib")
    cov = payload.get("residual_cov")
    bag, blocks, columns = payload["bag"], payload["blocks"], payload["columns"]
    order = [c for b in blocks for c in columns[b]]
    inp = pd.read_csv(input_csv)
    train_fp = np.vstack(list(payload["train_fps"].values())).astype(bool)
    curves, meta = [], []
    for _, row in inp.iterrows():
        lig = _ligand_features(str(row["smiles"]), columns)
        cond = {c: (float(row[c]) if c in inp.columns and pd.notna(row[c]) else np.nan) for c in columns["COND"]}
        dent = float(row["DENTATE"]) if "DENTATE" in inp.columns and pd.notna(row.get("DENTATE")) else lig["_dentate"]
        feats = {**cond, **_mass_action(cond, dent, 9.0), **{k: v for k, v in lig.items() if not k.startswith("_")}}
        # LIG2D block is not recomputed (not used by the lean arms): NaN, imputed inside the pipelines
        x = np.array([feats.get(c, np.nan) for c in order], dtype=float)[None, :]
        q = lig["_fp"].astype(bool)
        inter = (train_fp & q).sum(1); union = (train_fp | q).sum(1)
        sim = float((inter / np.maximum(union, 1)).max())
        c = bag.predict_curves(x, lig["_fp"][None, :])[0]
        if {"measured_A", "measured_B", "measured_logSF"} <= set(inp.columns) and pd.notna(row.get("measured_logSF")):
            a, b = LANTHANIDES.index(row["measured_A"]), LANTHANIDES.index(row["measured_B"])
            if a > b:
                a, b = b, a
                y_ab = -float(row["measured_logSF"])
            else:
                y_ab = float(row["measured_logSF"])
            if adapter == "conditional" and cov is not None:
                c = adapt_conditional(c, cov, a, b, y_ab)
            else:
                c = adapt_basis_shift(c, physics_basis_matrix(("radius", "radius_sq")), a, b, y_ab)
            calibrated = True
        else:
            calibrated = False
        curves.append(c); meta.append({"max_train_tanimoto": sim, "band": "far" if sim <= 0.4 else ("mid" if sim <= 0.6 else "near"),
                                       "dentate_used": dent, "calibrated_one_pair": calibrated,
                                       "adapter": (adapter if cov is not None else "basis_shift") if calibrated else "",
                                       "suggested_first_pair": "-".join(payload.get("suggested_pairs", [("La", "Lu")])[0])})
    out = inp.copy()
    for m, col in zip(LANTHANIDES, np.array(curves).T):
        out[f"curve__{m}"] = col
    for k in meta[0]:
        out[k] = [m[k] for m in meta]
    pairs = []
    for i, c in enumerate(curves):
        for a in range(14):
            for b in range(a + 1, 14):
                pairs.append({"row": i, "smiles": inp["smiles"].iat[i], "A": LANTHANIDES[a], "B": LANTHANIDES[b],
                              "dZ": ATOMIC_NUMBER[LANTHANIDES[b]] - ATOMIC_NUMBER[LANTHANIDES[a]], "pred_logSF_A_over_B": float(c[a] - c[b])})
    out.to_csv(output_csv, index=False)
    pd.DataFrame(pairs).to_csv(Path(output_csv).with_name(Path(output_csv).stem + "_pairs.csv"), index=False)
    print(out[["smiles"] + [f"curve__{m}" for m in LANTHANIDES] + ["max_train_tanimoto", "band"]].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--arms", default="direct@lean,physics@lean,lowrank1@lean,lowrank2@lean")
    p = sub.add_parser("predict"); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--adapter", default="conditional", choices=["conditional", "basis_shift"],
                   help="one-pair calibration: conditional (BLUP with the frozen residual covariance) or basis_shift")
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.arms.split(","))
    else:
        predict(Path(args.input), Path(args.output), adapter=args.adapter)


if __name__ == "__main__":
    main()
