"""D1 add-on: extractant-macro nearest-neighbour transfer error.

The programme scores by averaging within an extractant first, then over
extractants.  This script asks the atlas question in exactly that unit:

    if you predict a held-out cell's log SF values by copying another cell of
    the SAME extractant, what extractant-macro MAE do you get?

That number is the empirical ceiling for any model whose only handle on a cell
is "which extractant is this" - it is the error you make even when you know the
extractant perfectly and have a real measured curve for it.

Run from the repo root:
    .venv/Scripts/python.exe generations/gen13_separation/analysis/stage2/d1_curve_atlas/d1_macro_transfer.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from gen13sep.metals import LANTHANIDES  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "stage2", "d1_curve_atlas")
LN = list(LANTHANIDES)
MIN_SHARED = 2


def load():
    df = pd.read_parquet(os.path.join(ROOT, "manifests", "cohort_exact.parquet"))
    logd = df[[f"logD__{m}" for m in LN]].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        C = logd - np.nanmean(logd, axis=1, keepdims=True)
    return df, C


def sf_errors(ci, cj):
    """|log SF(i) - log SF(j)| over every metal pair both cells observed."""
    ok = np.isfinite(ci) & np.isfinite(cj)
    k = int(ok.sum())
    if k < MIN_SHARED:
        return None
    d = ci[ok] - cj[ok]
    iu = np.triu_indices(k, 1)
    return np.abs(d[:, None] - d[None, :])[iu]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    df, C = load()
    pub = df["publication_id"].astype(str).to_numpy()
    ct = df["chemotype"].to_numpy()
    name = df["extractant_name"].to_numpy()

    rows = []
    for smi, g in df.groupby("extractant", sort=False):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            continue
        per_cell_all, per_cell_diffpub = [], []
        for a in idx:
            errs_all, errs_dp = [], []
            for b in idx:
                if a == b:
                    continue
                e = sf_errors(C[a], C[b])
                if e is None:
                    continue
                errs_all.append(e.mean())
                if pub[a] != pub[b]:
                    errs_dp.append(e.mean())
            if errs_all:
                # a random sibling cell of the same extractant
                per_cell_all.append(float(np.mean(errs_all)))
            if errs_dp:
                per_cell_diffpub.append(float(np.mean(errs_dp)))
        if not per_cell_all:
            continue
        rows.append(dict(
            extractant=smi, extractant_name=name[idx[0]], chemotype=ct[idx[0]],
            n_cells=len(idx), n_cells_scored=len(per_cell_all),
            n_publications=len(set(pub[idx])),
            transfer_mae=float(np.mean(per_cell_all)),
            transfer_mae_diffpub=float(np.mean(per_cell_diffpub)) if per_cell_diffpub else np.nan,
            n_cells_scored_diffpub=len(per_cell_diffpub),
        ))
    T = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
    T.to_csv(os.path.join(OUT, "d1_extractant_macro_transfer.csv"), index=False)

    macro = T["transfer_mae"].mean()
    macro_dp = T["transfer_mae_diffpub"].dropna().mean()
    pooled = np.average(T["transfer_mae"], weights=T["n_cells_scored"])
    print(f"[macro] extractants with >= 2 comparable cells: {len(T)} of "
          f"{df.extractant.nunique()}; cells they hold: {int(T.n_cells.sum())}/{len(df)}")
    print(f"[macro] EXTRACTANT-MACRO transfer MAE (copy a sibling cell of the same "
          f"extractant) = {macro:.3f} log units over {len(T)} extractants")
    print(f"[macro] pooled (cell-weighted) = {pooled:.3f}")
    print(f"[macro] restricted to sibling cells from a DIFFERENT publication = "
          f"{macro_dp:.3f} over {int(T['transfer_mae_diffpub'].notna().sum())} extractants")
    print("[macro] reference numbers: best ensemble arm 0.481, C_DIRECT_ROW 0.495, "
          "corpus mean curve 0.603, heavier-always 0.674")

    # cross-extractant-inside-chemotype control, same macro unit
    rows2 = []
    for smi, g in df.groupby("extractant", sort=False):
        idx = g.index.to_numpy()
        my_ct = ct[idx[0]]
        other = np.where((ct == my_ct) & (df["extractant"].to_numpy() != smi))[0]
        if len(other) == 0:
            continue
        per_cell = []
        for a in idx:
            errs = [e.mean() for b in other if (e := sf_errors(C[a], C[b])) is not None]
            if errs:
                per_cell.append(float(np.mean(errs)))
        if per_cell:
            rows2.append(dict(extractant=smi, extractant_name=name[idx[0]],
                              chemotype=my_ct, n_cells_scored=len(per_cell),
                              transfer_mae_other_extractants=float(np.mean(per_cell))))
    T2 = pd.DataFrame(rows2)
    T2.to_csv(os.path.join(OUT, "d1_extractant_macro_transfer_control.csv"), index=False)
    print(f"[macro] CONTROL - copy a cell of a DIFFERENT extractant in the same chemotype: "
          f"{T2.transfer_mae_other_extractants.mean():.3f} over {len(T2)} extractants")

    both = T.merge(T2[["extractant", "transfer_mae_other_extractants"]], on="extractant")
    print(f"[macro] on the {len(both)} extractants where both are defined: own-extractant "
          f"{both.transfer_mae.mean():.3f} vs other-extractant "
          f"{both.transfer_mae_other_extractants.mean():.3f} "
          f"(gain {both.transfer_mae_other_extractants.mean()-both.transfer_mae.mean():+.3f})")

    print("\n[macro] per extractant (>= 3 cells):")
    print(both[both.n_cells >= 3][["extractant_name", "n_cells", "n_publications",
                                   "transfer_mae", "transfer_mae_diffpub",
                                   "transfer_mae_other_extractants"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
