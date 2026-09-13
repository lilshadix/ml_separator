"""Is the ligand effect on the amplitude separable from the laboratory effect at all?

The question this answers
-------------------------
``d6_step3_nested_anova.csv`` reports a *sequential* sum-of-squares split of the cell amplitude in
which "extractant identity" takes 59.7 % and ``publication_id`` takes **0.08 % if the condition
blocks are entered first and 5.4 % if provenance is entered first**.  That 70-fold order dependence
has been read as a nuisance.  It is not: it is the whole of the ligand/laboratory confounding, and
before any decomposition -- sequential, Shapley/LMG, or REML -- is quoted, the corpus has to be
asked whether the two factors are *separable in principle*.

Three things are computed, all cheap and all exact.

1.  **The crossing table.**  How many publications measure each extractant, and vice versa.  If an
    extractant appears in one publication only, its ligand effect and that laboratory's effect are
    perfectly aliased and *no* estimator can tell them apart -- the split between them is decided
    entirely by the prior or the entry order, which is exactly what the ANOVA's order dependence
    was showing.

2.  **The R2 Venn.**  ``R2(y ~ extractant)``, ``R2(y ~ publication)``, ``R2(y ~ both)`` and hence
    the *unique* share of each and the aliased share they must divide between them.  A Shapley/LMG
    split (Gromping, The American Statistician 61(2):139-147, 2007) halves the aliased part by
    construction; reporting the aliased part itself is more honest than reporting any halving of
    it.  Rank deficiency of the joint design is reported next to it, because that integer is the
    number of degrees of freedom that are aliased outright.

3.  **The only unconfounded evidence in the corpus.**  For the extractants that *are* measured in
    more than one publication, the between-publication spread of the amplitude is a direct,
    ligand-free estimate of the laboratory effect, and the within-publication spread is the
    replicate noise.  Their ratio is the honest version of "how much of the amplitude is the
    ligand".  A leave-one-publication-out check asks the operational form of the same question:
    does a ligand's amplitude measured in one laboratory predict the same ligand in another?

Nothing here is a model.  It is a description of the design, and it bounds what every variance
decomposition and every hold-out design in the programme can possibly claim.

Usage:  python generations/gen16_protocol/scripts/g16_identifiability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen16_protocol"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen14.dirbench import MIN_METALS, load          # noqa: E402
from gen16.variance import indicator                 # noqa: E402

OUT = ROOT / "generations" / "gen16_protocol" / "results"
OUT.mkdir(parents=True, exist_ok=True)
REPLICATE_SD = 0.237        # corpus replicate sd of the amplitude, quoted throughout gen13/gen14


def r2(y: np.ndarray, *blocks: np.ndarray) -> tuple[float, int]:
    """In-sample R2 of ``y`` on an intercept plus the given blocks, and the design's rank."""
    n = len(y)
    A = np.c_[np.ones(n), *blocks] if blocks else np.ones((n, 1))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - float(res @ res) / sst), int(np.linalg.matrix_rank(A))


def main() -> None:
    bench = load()
    f = bench.frame.reset_index(drop=True)
    rich = f.n_metals.to_numpy() >= MIN_METALS
    d = f[rich].copy()
    d["amp"] = bench.coef[:, 0][rich]
    y = d.amp.to_numpy()

    # ---------------------------------------------------------------- 1. crossing table
    pubs_per_ext = d.groupby("extractant").publication_id.nunique()
    ext_per_pub = d.groupby("publication_id").extractant.nunique()
    multi = sorted(pubs_per_ext.index[pubs_per_ext >= 2])
    cross = pd.DataFrame([{
        "n_cells": len(d), "n_extractants": d.extractant.nunique(),
        "n_publications": d.publication_id.nunique(), "n_chemotypes": d.chemotype.nunique(),
        "extractants_in_one_publication_only": int((pubs_per_ext == 1).sum()),
        "extractants_in_two_or_more": int((pubs_per_ext >= 2).sum()),
        "share_extractants_aliased": float((pubs_per_ext == 1).mean()),
        "cells_on_aliased_extractants": int(d.extractant.isin(
            pubs_per_ext.index[pubs_per_ext == 1]).sum()),
        "publications_with_one_extractant": int((ext_per_pub == 1).sum()),
        "publications_touching_more_than_one_chemotype":
            int((d.groupby("publication_id").chemotype.nunique() > 1).sum()),
    }])
    print("=== 1. crossing table (well-determined cells) ===")
    print(cross.T.to_string(header=False))
    cross.to_csv(OUT / "g16_identifiability_crossing.csv", index=False)

    # ---------------------------------------------------------------- 2. the R2 Venn
    Ze, Zp, Zc = (indicator(d.extractant), indicator(d.publication_id), indicator(d.chemotype))
    r2_e, rk_e = r2(y, Ze)
    r2_p, rk_p = r2(y, Zp)
    r2_ep, rk_ep = r2(y, Ze, Zp)
    r2_c, rk_c = r2(y, Zc)
    r2_cp, rk_cp = r2(y, Zc, Zp)
    venn = pd.DataFrame([
        {"pair": "extractant vs publication",
         "r2_A_alone": r2_e, "r2_B_alone": r2_p, "r2_joint": r2_ep,
         "unique_A": r2_ep - r2_p, "unique_B": r2_ep - r2_e,
         "aliased_shared": r2_e + r2_p - r2_ep,
         "rank_A": rk_e, "rank_B": rk_p, "rank_joint": rk_ep,
         "aliased_dof": rk_e + rk_p - rk_ep - 1},
        {"pair": "chemotype vs publication",
         "r2_A_alone": r2_c, "r2_B_alone": r2_p, "r2_joint": r2_cp,
         "unique_A": r2_cp - r2_p, "unique_B": r2_cp - r2_c,
         "aliased_shared": r2_c + r2_p - r2_cp,
         "rank_A": rk_c, "rank_B": rk_p, "rank_joint": rk_cp,
         "aliased_dof": rk_c + rk_p - rk_cp - 1},
    ])
    print("\n=== 2. R2 Venn (in-sample, saturated indicators -- read the SHARES, not the levels) ===")
    print(venn.round(4).to_string(index=False))
    venn.to_csv(OUT / "g16_identifiability_venn.csv", index=False)

    # ---------------------------------------------------------------- 3. unconfounded evidence
    rows = []
    for e in multi:
        g = d[d.extractant == e]
        pm = g.groupby("publication_id").amp.agg(["mean", "std", "size"])
        within = np.nan
        if (pm["size"] > 1).any():
            num = float(np.nansum((pm["size"] - 1) * pm["std"].fillna(0.0) ** 2))
            den = float((pm["size"] - 1).sum())
            within = float(np.sqrt(num / den)) if den > 0 else np.nan
        rows.append({"extractant": e, "chemotype": g.chemotype.iloc[0],
                     "n_publications": int(len(pm)), "n_cells": int(len(g)),
                     "mean_amp": float(g.amp.mean()),
                     "between_publication_sd": float(pm["mean"].std(ddof=1)),
                     "within_publication_sd": within,
                     "range_of_publication_means": float(pm["mean"].max() - pm["mean"].min()),
                     "direction_flips_between_labs": bool(pm["mean"].min() < 0 < pm["mean"].max())})
    T = pd.DataFrame(rows)
    print("\n=== 3. the 7 ligands measured in more than one publication ===")
    print(T.assign(extractant=T.extractant.str.slice(0, 34) + "...").round(3).to_string(index=False))
    T.to_csv(OUT / "g16_identifiability_multipub.csv", index=False)

    # leave-one-publication-out: does the same ligand in another laboratory predict this one?
    lopo = []
    for e in multi:
        g = d[d.extractant == e]
        for p, gp in g.groupby("publication_id"):
            other = g[g.publication_id != p]
            if other.empty:
                continue
            lopo.append({"extractant": e, "held_out_publication": p,
                         "n_cells": len(gp), "observed": float(gp.amp.mean()),
                         "predicted_from_other_labs": float(other.amp.mean())})
    L = pd.DataFrame(lopo)
    obs, pred = L.observed.to_numpy(), L.predicted_from_other_labs.to_numpy()
    sst = float(((obs - obs.mean()) ** 2).sum())
    sse = float(((obs - pred) ** 2).sum())
    summary = pd.DataFrame([{
        "n_ligand_publication_units": len(L), "n_ligands": len(multi),
        "chemotypes_represented": int(T.chemotype.nunique()),
        "lopo_r2_same_ligand_other_lab": 1.0 - sse / sst,
        "lopo_mae": float(np.abs(obs - pred).mean()),
        "lopo_sign_agreement": float((np.sign(obs) == np.sign(pred)).mean()),
        "median_between_publication_sd": float(T.between_publication_sd.median()),
        "median_within_publication_sd": float(T.within_publication_sd.median(skipna=True)),
        "corpus_sd_amplitude": float(y.std()),
        "replicate_sd_quoted_in_reports": REPLICATE_SD,
        "ligands_flipping_direction_between_labs": int(T.direction_flips_between_labs.sum()),
    }])
    print("\n=== leave-one-publication-out, same ligand ===")
    print(summary.T.round(4).to_string(header=False))
    summary.to_csv(OUT / "g16_identifiability_lopo.csv", index=False)
    L.to_csv(OUT / "g16_identifiability_lopo_units.csv", index=False)

    print("\nHow to read this: the between-publication sd is a laboratory effect measured with the "
          "ligand held fixed.\nCompare it with the corpus sd of the amplitude before quoting any "
          "'extractant identity explains X %' number.")


if __name__ == "__main__":
    main()
