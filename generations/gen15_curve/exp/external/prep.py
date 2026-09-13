"""Build the two aligned tables for the external-validation test.

Side E (extraction): one row per extractant with a well-determined curve (>= 5 measured metals),
carrying its chemotype-weighted mean radius coefficient ``amp`` -- the quantity gen14 classifies.
Side K (aqueous logK): the 273 external ligands of ``logk_external_series.parquet`` with their
fitted ``slope`` on the *same* standardised Shannon radius basis.

Both sides get the identical 39-column donor-topology block, computed by the *same* frozen builder
(``gen122.coordination.build_table``), which is a pure function of one SMILES string.

Nothing here reads any log D or any logK value into the feature block.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "data"
for p in (ROOT / "generations" / "gen15_curve", ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction",
          ROOT / "generations" / "gen12_2_eu_pred"):
    sys.path.insert(0, str(p))

from gen15 import valuebench as V           # noqa: E402
from gen14.dirbench import feature_sets     # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen122.coordination import build_table, active_spec_path, spec_digest  # noqa: E402

MIN_METALS = 5


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = V.load()
    fs = feature_sets(bench)
    lean_cols = bench.columns(LEAN_BLOCKS)
    topo_cols = [lean_cols[i] for i in fs["TOPO39"]]
    print(f"TOPO39 = {len(topo_cols)} columns; spec {active_spec_path().name} {spec_digest()[:12]}")

    frame = bench.frame
    amp = bench.coef[:, 0]
    rich = frame.n_metals.to_numpy() >= MIN_METALS

    # ---- side E: extractant-level direction label -------------------------------------
    sub = frame[rich].copy()
    sub["amp"] = amp[rich]
    w = cell_weights(bench.groups[rich], bench.n_obs[rich])          # chemotype-balanced
    sub["w"] = w
    g = sub.groupby("extractant")
    E = pd.DataFrame({
        "amp": g.apply(lambda d: float(np.average(d.amp, weights=d.w)), include_groups=False),
        "chemotype": g.chemotype.first(),
        "n_cells": g.size(),
        "n_metals_max": g.n_metals.max(),
        "w": g.w.sum(),
    }).reset_index().rename(columns={"extractant": "smiles"})
    E["y_heavy"] = (E.amp < 0).astype(int)
    print(f"side E: {len(E)} extractants, {E.chemotype.nunique()} chemotypes, "
          f"heavy base rate {E.y_heavy.mean():.3f}")

    # ---- side K: external aqueous logK series ------------------------------------------
    K = pd.read_parquet(ROOT / "generations" / "gen13_separation" / "features" / "logk_external_series.parquet")
    K = K.reset_index(drop=True)
    K["y_heavy"] = (K.slope < 0).astype(int)
    print(f"side K: {len(K)} ligands, heavy base rate {K.y_heavy.mean():.3f}")

    # ---- the shared topology block ------------------------------------------------------
    all_smi = sorted(set(E.smiles) | set(K.smiles))
    T = build_table(all_smi)
    missing = [c for c in topo_cols if c not in T.columns]
    if missing:
        raise SystemExit(f"builder does not produce {missing}")
    T = T[topo_cols]
    print(f"built topology for {len(T)} distinct SMILES; NaN cells {int(T.isna().sum().sum())}")

    # ---- reproduction check against the frozen parquet ----------------------------------
    Xb = bench.matrix(LEAN_BLOCKS)[:, fs["TOPO39"]]
    ext = frame.extractant.to_numpy()
    keep = np.array([e in T.index for e in ext])
    Xr = T.loc[ext[keep]].to_numpy(float)
    d = np.abs(np.nan_to_num(Xb[keep]) - np.nan_to_num(Xr))
    rep = {"max_abs_diff_vs_frozen_parquet": float(d.max()),
           "n_cells_checked": int(keep.sum()),
           "n_mismatched_cells": int((d.max(axis=1) > 1e-9).sum())}
    print("reproduction check:", rep)

    E = E.join(T.reindex(E.smiles).reset_index(drop=True))
    K = K.join(T.reindex(K.smiles).reset_index(drop=True))

    # ---- chemical-space overlap: Tanimoto ECFP4 2048 ------------------------------------
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    def fps(smis):
        out = []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            out.append(None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
        return out

    fe, fk = fps(E.smiles.tolist()), fps(K.smiles.tolist())
    ok_e = [i for i, f in enumerate(fe) if f is not None]
    nn_sim, nn_idx = [], []
    for f in fk:
        if f is None:
            nn_sim.append(np.nan); nn_idx.append(-1); continue
        s = np.array(DataStructs.BulkTanimotoSimilarity(f, [fe[i] for i in ok_e]))
        j = int(s.argmax()); nn_sim.append(float(s[j])); nn_idx.append(ok_e[j])
    K["nn_tanimoto"] = nn_sim
    K["nn_extractant"] = [E.smiles.iat[i] if i >= 0 else None for i in nn_idx]
    K["nn_chemotype"] = [E.chemotype.iat[i] if i >= 0 else None for i in nn_idx]

    ok_k = [i for i, f in enumerate(fk) if f is not None]
    e_sim = []
    for f in fe:
        if f is None:
            e_sim.append(np.nan); continue
        s = np.array(DataStructs.BulkTanimotoSimilarity(f, [fk[i] for i in ok_k]))
        e_sim.append(float(s.max()))
    E["nn_tanimoto"] = e_sim

    # exact structural overlap
    shared = sorted(set(E.smiles) & set(K.smiles))
    print(f"exact SMILES shared between the two sets: {len(shared)}")

    E.to_parquet(OUT / "side_E_extraction.parquet", index=False)
    K.to_parquet(OUT / "side_K_logk.parquet", index=False)
    meta = {
        "topo39_columns": topo_cols,
        "coordination_spec": str(active_spec_path().name),
        "coordination_spec_sha256": spec_digest(),
        "reproduction_check": rep,
        "n_E": int(len(E)), "n_K": int(len(K)),
        "E_heavy_base_rate": float(E.y_heavy.mean()),
        "K_heavy_base_rate": float(K.y_heavy.mean()),
        "E_chemotypes": int(E.chemotype.nunique()),
        "exact_shared_smiles": shared,
        "K_nn_tanimoto_quantiles": {q: float(np.nanquantile(K.nn_tanimoto, q))
                                    for q in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
        "E_nn_tanimoto_quantiles": {q: float(np.nanquantile(E.nn_tanimoto, q))
                                    for q in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
    }
    (OUT / "prep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "topo39_columns"}, indent=1)[:2000])


if __name__ == "__main__":
    main()
