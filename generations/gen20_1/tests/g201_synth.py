"""Synthetic long tables in the CONTRACT schemas (tests only; no real target value is ever read).

``make_cells`` deals fake cells (cell, extractant, chemotype, publication, measured metals); ``make_tables`` gives
the logd and logsf long tables of one (design, variant) for several split seeds with synthetic y and synthetic
arm columns whose quality is controlled by ``skill``; ``manifests_for`` builds population keys like the S3 row
manifests.  The metals and Z come from gen201.data (label-free constants).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gen201.data import LANTHANIDES, Z

SEEDS = (11, 22, 33, 44, 55)
LOGD_ARMS = ("A", "A_K1", "A_K2", "A_K3", "A_K4", "A_PRIME", "BD_MEAN", "BD_METAL", "BD_COND", "BD_NN", "BD_LEVEL",
             "BD_LEVEL_CURVE", "C", "C_K1", "C_K2", "C_K3", "C_K4", "C_ACFG", "C_selSF", "B_selSF_raw")
LOGSF_ARMS = ("B", "B_K1", "B_K2", "B_K3", "B_K4", "B_selSF", "C", "C_K1", "C_K2", "C_K3", "C_K4", "C_ACFG", "C_selSF",
              "C_PAIR", "FLAT", "PAIRMEAN", "G14_ZERO", "MEAN_CURVE_REF", "NN_TANIMOTO_REF", "HEAVIER_ALWAYS",
              "BD_METAL_SF", "BD_COND_SF", "BD_NN_SF", "BD_LEVEL_CURVE_SF")


def make_cells(n_chem: int = 14, *, seed: int = 0, n_pubs: int = 12, max_ext: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ci = 0
    for g in range(n_chem):
        chem = f"sc{g:03d}"
        for j in range(int(rng.integers(1, max_ext + 1))):
            ext = f"EXT{g:02d}_{j}"
            pubs = rng.choice(n_pubs, size=int(rng.integers(1, 3)), replace=False)
            for p in pubs:
                for _ in range(int(rng.integers(1, 3))):
                    start = int(rng.integers(0, 6))
                    stop = int(rng.integers(start + 5, 15))
                    rows.append({"ci": ci, "cell_id": f"cell{ci:04d}", "extractant": ext, "chemotype": chem,
                                 "publication_id": f"pub_{int(p):02d}", "metals": list(range(start, min(stop, 14)))})
                    ci += 1
    return pd.DataFrame(rows)


def _truth(cells: pd.DataFrame, seed: int = 1):
    """Fixed synthetic y per (cell, metal): level + chemotype curve + noise."""
    rng = np.random.default_rng(seed)
    chem_curve = {c: rng.normal(0, 0.15, 14).cumsum() for c in sorted(cells["chemotype"].unique())}
    y = {}
    level = {}
    for r in cells.itertuples():
        level[r.ci] = rng.normal(0.5, 1.2)
        for m in r.metals:
            y[(r.ci, m)] = level[r.ci] + chem_curve[r.chemotype][m] + rng.normal(0, 0.12)
    return y, level


def make_tables(cells: pd.DataFrame, *, seeds=SEEDS, design: str = "BP", variant: str = "primary",
                skill: float = 0.5, truth_seed: int = 1, noise_seed: int = 100, drop_cells=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """logd and logsf long tables.  ``skill`` in [0, 1]: 1 makes A and C near-perfect, 0 makes them as poor as the
    baselines.  Predictions differ by seed (seed-specific noise), y does not."""
    y, level = _truth(cells, truth_seed)
    cells = cells[~cells["ci"].isin(list(drop_cells))].reset_index(drop=True)
    metals_all = {m for ms in cells["metals"] for m in ms}
    ymean = np.mean(list(y.values()))
    metal_mean = {m: np.mean([v for (c, mm), v in y.items() if mm == m]) for m in metals_all}
    logd_parts, logsf_parts = [], []
    chem_list = sorted(cells["chemotype"].unique())
    for s in seeds:
        rng = np.random.default_rng(noise_seed + int(s) * 7 + sum(map(ord, design)) % 97)
        fold_of_chem = {c: int(rng.integers(0, 5)) for c in chem_list}
        lev_err = {r.ci: rng.normal(0, 0.9 * (1 - skill) + 0.05) for r in cells.itertuples()}
        rows = []
        for r in cells.itertuples():
            for m in r.metals:
                yt = y[(r.ci, m)]
                e_small = rng.normal(0, 0.25 * (1 - skill) + 0.01)
                a = yt + lev_err[r.ci] + e_small
                d = {"design": design, "variant": variant, "split_seed": int(s), "fold": fold_of_chem[r.chemotype],
                     "ci": int(r.ci), "cell_id": r.cell_id, "extractant": r.extractant, "chemotype": r.chemotype,
                     "publication_id": r.publication_id, "n_metals": len(r.metals), "metal": LANTHANIDES[m],
                     "im": int(m), "y": float(yt), "A": a, "A_PRIME": a + rng.normal(0, 0.002),
                     "BD_MEAN": ymean, "BD_METAL": metal_mean[m],
                     "BD_COND": yt + rng.normal(0, 0.9), "BD_NN": metal_mean[m] + rng.normal(0, 0.8),
                     "BD_LEVEL": level[r.ci] + lev_err[r.ci] * 1.5,
                     "BD_LEVEL_CURVE": level[r.ci] + lev_err[r.ci] * 1.5 + (metal_mean[m] - ymean)}
                for k in range(1, 5):
                    d[f"A_K{k}"] = a + rng.normal(0, 0.03 * k)
                d["B_selSF_raw"] = d["A_K2"]
                rows.append(d)
        t = pd.DataFrame(rows)
        # centred predictions (C family): truth curve within the cell + small error
        for col, sd in (("C", 0.06), ("C_K1", 0.08), ("C_K2", 0.07), ("C_K3", 0.09), ("C_K4", 0.1), ("C_ACFG", 0.07),
                        ("C_selSF", 0.065)):
            raw = t["y"] + rng.normal(0, sd * (2 - skill), len(t))
            t[col] = raw - t.groupby("ci")["y"].transform("mean") + rng.normal(0, 0.05, len(t))
        logd_parts.append(t)
        # pairs: consecutive measured metals with dZ = 1 (strict) plus Nd-Sm (gen13 adjacency)
        lk = {(int(r["ci"]), int(r["im"])): r for r in t.to_dict("records")}
        prs = []
        for r in cells.itertuples():
            ms = sorted(r.metals)
            for ia, ib in zip(ms[:-1], ms[1:]):
                dz = int(Z[ib] - Z[ia])
                strict = dz == 1
                g13 = strict or (LANTHANIDES[ia], LANTHANIDES[ib]) == ("Nd", "Sm")
                if not g13:
                    continue
                A_, B_ = lk[(r.ci, ia)], lk[(r.ci, ib)]
                d = {"design": design, "variant": variant, "split_seed": int(s), "fold": fold_of_chem[r.chemotype],
                     "ci": int(r.ci), "cell_id": r.cell_id, "extractant": r.extractant, "chemotype": r.chemotype,
                     "publication_id": r.publication_id, "n_metals": len(r.metals), "a": LANTHANIDES[ia],
                     "b": LANTHANIDES[ib], "ia": int(ia), "ib": int(ib), "dZ": dz,
                     "pair": f"{LANTHANIDES[ia]}/{LANTHANIDES[ib]}", "strict": bool(strict), "gen13adj": bool(g13),
                     "y": float(A_["y"] - B_["y"])}
                d["B"] = A_["A"] - B_["A"]
                for k in range(1, 5):
                    d[f"B_K{k}"] = A_[f"A_K{k}"] - B_[f"A_K{k}"]
                d["B_selSF"] = A_["B_selSF_raw"] - B_["B_selSF_raw"]
                for col in ("C", "C_K1", "C_K2", "C_K3", "C_K4", "C_ACFG", "C_selSF"):
                    d[col] = A_[col] - B_[col]
                d["C_PAIR"] = d["C"] + rng.normal(0, 0.02)
                d["FLAT"] = 0.0
                d["MEAN_CURVE_REF"] = metal_mean[ia] - metal_mean[ib]
                d["NN_TANIMOTO_REF"] = d["MEAN_CURVE_REF"] + rng.normal(0, 0.1)
                d["G14_ZERO"] = 0.6 * d["MEAN_CURVE_REF"]
                d["HEAVIER_ALWAYS"] = -1.0
                for bd in ("BD_METAL", "BD_COND", "BD_NN", "BD_LEVEL_CURVE"):
                    d[f"{bd}_SF"] = A_[bd] - B_[bd]
                prs.append(d)
        q = pd.DataFrame(prs)
        q["PAIRMEAN"] = q.groupby("pair")["y"].transform("mean") * 0.5
        logsf_parts.append(q)
    return pd.concat(logd_parts, ignore_index=True), pd.concat(logsf_parts, ignore_index=True)


def manifests_for(logd: pd.DataFrame, logsf: pd.DataFrame, *, dmd_cells=(), sc009: str = "sc009") -> dict:
    """Population keys like manifests/row_manifests (one seed's keys; seed-invariant)."""
    s0 = logd["split_seed"].iloc[0]
    d = logd[logd["split_seed"] == s0]
    q = logsf[logsf["split_seed"] == s0]
    dcols = ["ci", "im", "cell_id", "extractant", "chemotype", "publication_id", "metal"]
    qcols = ["ci", "ia", "ib", "cell_id", "extractant", "chemotype", "publication_id", "a", "b", "pair"]
    strict = q[q["strict"]]
    in_pair = set(zip(strict["ci"], strict["ia"])) | set(zip(strict["ci"], strict["ib"]))
    c3m = d[[k in in_pair for k in zip(d["ci"], d["im"])]]
    pops = {"COH-D-C3": d[dcols], "COH-SF-C3": strict[qcols], "COH-D-C3M": c3m[dcols], "COH-D-ALL": d[dcols],
            "COH-SF-ANY": strict[qcols], "COH-SF-C1": strict[qcols], "COH-SF-NDSM": q[q["gen13adj"]][qcols],
            "COH-SC009-D": d[d["chemotype"] == sc009][dcols], "COH-SC009-SF": strict[strict["chemotype"] == sc009][qcols],
            "COH-NOSC009-D": d[d["chemotype"] != sc009][dcols],
            "COH-NOSC009-SF": strict[strict["chemotype"] != sc009][qcols]}
    out = {}
    for k, v in pops.items():
        keys = ["ci", "ia", "ib"] if "ia" in v.columns else ["ci", "im"]
        out[k] = v.sort_values(keys).reset_index(drop=True)
        nod = v[~v["ci"].isin(list(dmd_cells))]
        out[f"{k}@NODMD"] = nod.sort_values(keys).reset_index(drop=True)
    return out
