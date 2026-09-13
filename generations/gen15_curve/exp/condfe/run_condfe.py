"""Score the within-publication condition arms under all five designs.

Usage:  python run_condfe.py [main|guard|all]

``main``   the full arm board under B, BR, BQ, A, BP with the chemotype-blocked paired bootstrap.
``guard``  the confound guard: the same board on a bench with the largest publication removed, and
           again with the largest chemotype removed.

Everything is written to this directory.  Nothing outside ``gen15_curve/exp/condfe/`` is touched.
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (str(ROOT / "generations" / "gen15_curve"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gen15 import valuebench as V, arms as A  # noqa: E402
import conds as CD  # noqa: E402
import fearms as F  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

DESIGNS = ["B", "BR", "BQ", "A", "BP"]
LOGS: dict[str, dict] = {}
#: arms whose fitted within coefficients are worth printing column by column
COLS: dict[str, list] = {}


def build_arms() -> dict:
    """Reference arms, the four estimator families the task asks for, and the two oracles."""
    for k in ("FE_PHYS4", "FE_ACID", "FE_C64", "META_PHYS4", "FE_NUM14"):
        LOGS[k] = {}
    COLS.update({"FE_PHYS4": list(CD.PHYS4), "META_PHYS4": list(CD.PHYS4),
                 "FE_NUM14": list(CD.NUMERIC)})
    return {
        # --- references ------------------------------------------------------------------
        "FLAT": A.flat,
        "G14": A.g14,
        # --- the within-transformation (fixed effects) -------------------------------------
        "FE_ACID": F.fe_arm(["log_acid_M"], log=LOGS["FE_ACID"]),
        "FE_PHYS4": F.fe_arm(CD.PHYS4, log=LOGS["FE_PHYS4"]),
        "FE_NUM14": F.fe_arm(CD.NUMERIC, log=LOGS["FE_NUM14"]),
        "FE_C64": F.fe_arm(use64=True, log=LOGS["FE_C64"]),
        # --- the same quantity by the literal FWL two-step (identity check) ----------------
        "RESID_PHYS4": F.resid_arm(CD.PHYS4),
        # --- robust: one within slope per publication, median across publications ----------
        "META_ACID": F.meta_arm(["log_acid_M"]),
        "META_PHYS4": F.meta_arm(CD.PHYS4, log=LOGS["META_PHYS4"]),
        # --- publication random intercept ---------------------------------------------------
        "RI_PHYS4": F.ri_arm(CD.PHYS4),
        "RI_C64": F.ri_arm(use64=True),
        # --- the naive comparator with no publication handling at all ----------------------
        "POOL_C64": F.pooled_arm(use64=True),
        # --- oracles (never deployable) ----------------------------------------------------
        "O_WITHIN": F.o_within,
        "O_PUBMAG": F.o_pubmag,
    }


COMPS = {
    "FE_ACID_vs_G14": ("G14", "FE_ACID"),
    "FE_PHYS4_vs_G14": ("G14", "FE_PHYS4"),
    "FE_NUM14_vs_G14": ("G14", "FE_NUM14"),
    "FE_C64_vs_G14": ("G14", "FE_C64"),
    "META_PHYS4_vs_G14": ("G14", "META_PHYS4"),
    "RI_PHYS4_vs_G14": ("G14", "RI_PHYS4"),
    "RI_C64_vs_G14": ("G14", "RI_C64"),
    "POOL_C64_vs_G14": ("G14", "POOL_C64"),
    "O_WITHIN_vs_G14": ("G14", "O_WITHIN"),
    "O_PUBMAG_vs_G14": ("G14", "O_PUBMAG"),
    "FE_PHYS4_vs_FLAT": ("FLAT", "FE_PHYS4"),
    "G14_vs_FLAT": ("FLAT", "G14"),
}


def report(B: pd.DataFrame, C: pd.DataFrame, tag: str) -> None:
    print(f"\n########## {tag} ##########")
    print("\n--- extractant-macro MAE (lower is better) ---")
    print(V.wide(B).round(4).to_string())
    for col in ("macro_sign_acc_strong", "macro_pair_spearman"):
        if col in B.columns:
            print(f"\n--- {col} ---")
            print(V.wide(B, col).round(4).to_string())
    if len(C):
        print("\n--- chemotype-blocked paired bootstrap (positive point = candidate better) ---")
        keep = [c for c in ["design", "comparison", "point", "ci95_low", "ci95_high",
                            "p_two_sided", "loco_min", "loco_max", "loco_sign_stable",
                            "seeds_positive", "n_seeds", "passes_P1"] if c in C.columns]
        print(C[keep].round(4).to_string(index=False))


def subset_bench(bench, mask: np.ndarray):
    """A bench restricted to ``mask`` rows -- used only by the confound guard."""
    return replace(bench,
                   frame=bench.frame[mask].reset_index(drop=True),
                   Y=bench.Y[mask], coef=bench.coef[mask], n_obs=bench.n_obs[mask],
                   frames={k: v[mask].reset_index(drop=True) for k, v in bench.frames.items()},
                   groups=bench.groups[mask])


def dump_logs(tag: str) -> None:
    rows = []
    for arm, log in LOGS.items():
        if not log.get("shrink"):
            continue
        sh = np.asarray(log["shrink"], dtype=float)
        beta = np.vstack(log["beta"])
        beta_raw = np.vstack(log["beta_raw"])
        keep = np.vstack(log["keep"])
        rows.append({"arm": arm, "n_fits": len(sh), "shrink_mean": sh.mean(),
                     "shrink_zero_frac": float((sh == 0).mean()),
                     "cols_identified_mean": float(keep.sum(axis=1).mean()),
                     "beta_l1_mean": float(np.abs(beta).sum(axis=1).mean()),
                     "beta_raw_l1_mean": float(np.abs(beta_raw).sum(axis=1).mean())})
    if rows:
        T = pd.DataFrame(rows)
        T.to_csv(HERE / f"fit_log_{tag}.csv", index=False)
        print(f"\n--- what the within fits actually learned ({tag}) ---")
        print(T.round(4).to_string(index=False))
    # per-column mean coefficient for the interpretable blocks
    for arm, cols in COLS.items():
        log = LOGS.get(arm) or {}
        if not log.get("beta_raw"):
            continue
        Braw = np.vstack(log["beta_raw"])
        K = np.vstack(log["keep"])
        T = pd.DataFrame({"column": cols,
                          "beta_raw_mean": Braw.mean(axis=0),
                          "beta_raw_sd": Braw.std(axis=0),
                          "frac_positive": (Braw > 0).mean(axis=0),
                          "frac_identified": K.mean(axis=0)})
        T.to_csv(HERE / f"beta_{arm}_{tag}.csv", index=False)
        print(f"\n--- within coefficients on log|a|, {arm} ({tag}, mean over folds x designs) ---")
        print(T.round(4).to_string(index=False))


def run_main() -> None:
    bench = V.load()
    print(f"bench: {len(bench.frame)} cells, {bench.frame.publication_id.nunique()} publications, "
          f"{bench.frame.chemotype.nunique()} chemotypes")
    ctx_arms = build_arms()
    t0 = time.time()
    B, C, tables = V.score(bench, ctx_arms, DESIGNS, comps=COMPS)
    print(f"\ntotal {time.time() - t0:.0f}s")
    B.to_csv(HERE / "board_main.csv", index=False)
    C.to_csv(HERE / "contrasts_main.csv", index=False)
    for d, (_, pe) in tables.items():
        pe.to_csv(HERE / f"perext_{d}.csv", index=False)
    report(B, C, "MAIN: five designs, all arms")
    dump_logs("main")

    # the FWL identity, verified rather than assumed
    same = np.allclose(V.wide(B).loc["FE_PHYS4"].to_numpy(),
                       V.wide(B).loc["RESID_PHYS4"].to_numpy(), atol=1e-6)
    print(f"\nFWL identity (within == residual-on-residual) across all five designs: {same}")


def run_forced() -> None:
    """What if the within estimate were simply *trusted*, instead of shrunk by the inner CV?

    The main run's inner leave-one-publication-out shrinkage chooses zero in about nine folds in
    ten, so most of the within arms collapse onto gen14 by construction.  That is the honest
    answer, but it leaves open whether the shrinkage is merely too conservative.  Here the
    shrinkage is pinned at 1.0, and the robust publication-median estimator -- the one whose acid
    coefficient has the sign the diagnostic finds, and whose sign survives leave-one-chemotype-out
    -- is run alongside the plain pooled within estimator.  If a within condition effect can help
    at all, it has to show here.
    """
    bench = V.load()
    LOGS.clear(); COLS.clear()
    for k in ("FE_PHYS4_S1", "META_PHYS4_S1", "META_ACID_S1"):
        LOGS[k] = {}
    COLS.update({"FE_PHYS4_S1": list(CD.PHYS4), "META_PHYS4_S1": list(CD.PHYS4),
                 "META_ACID_S1": ["log_acid_M"]})
    arms = {
        "FLAT": A.flat,
        "G14": A.g14,
        "FE_ACID_S1": F.fe_arm(["log_acid_M"], shrink=1.0),
        "FE_PHYS4_S1": F.fe_arm(CD.PHYS4, shrink=1.0, log=LOGS["FE_PHYS4_S1"]),
        "FE_C64_S1": F.fe_arm(use64=True, shrink=1.0),
        "META_ACID_S1": F.meta_arm(["log_acid_M"], shrink=1.0, log=LOGS["META_ACID_S1"]),
        "META_PHYS4_S1": F.meta_arm(CD.PHYS4, shrink=1.0, log=LOGS["META_PHYS4_S1"]),
        "FE_ACID_MIN4_S1": F.fe_arm(["log_acid_M"], shrink=1.0, min_cells=4),
        "FE_CURV_PHYS4": F.fe_curv(CD.PHYS4, shrink=1.0),
        "O_WITHIN": F.o_within,
    }
    comps = {f"{k}_vs_G14": ("G14", k) for k in arms if k not in ("FLAT", "G14")}
    B, C, _ = V.score(bench, arms, DESIGNS, comps=comps)
    B.to_csv(HERE / "board_forced.csv", index=False)
    C.to_csv(HERE / "contrasts_forced.csv", index=False)
    report(B, C, "FORCED: shrinkage pinned at 1.0 (the within estimate simply trusted)")
    dump_logs("forced")


def run_guard() -> None:
    """Any gain must survive dropping the largest publication and the largest chemotype."""
    bench = V.load()
    f = bench.frame
    big_pub = f.publication_id.value_counts().idxmax()
    big_chem = f.chemotype.value_counts().idxmax()
    print(f"largest publication {big_pub} ({int((f.publication_id == big_pub).sum())} cells); "
          f"largest chemotype {big_chem} ({int((f.chemotype == big_chem).sum())} cells)")
    guard_arms = {k: v for k, v in build_arms().items()
                  if k in ("FLAT", "G14", "FE_ACID", "FE_PHYS4", "FE_C64", "META_PHYS4",
                           "RI_PHYS4", "O_WITHIN", "O_PUBMAG")}
    guard_comps = {k: v for k, v in COMPS.items()
                   if v[1] in guard_arms and v[0] in guard_arms}
    out = []
    for tag, mask in (("drop_pub", (f.publication_id != big_pub).to_numpy()),
                      ("drop_chem", (f.chemotype != big_chem).to_numpy())):
        F.reset_caches()
        sub = subset_bench(bench, mask)
        print(f"\n[{tag}] {len(sub.frame)} cells, {sub.frame.publication_id.nunique()} publications, "
              f"{sub.frame.chemotype.nunique()} chemotypes")
        B, C, _ = V.score(sub, guard_arms, DESIGNS, comps=guard_comps)
        B.insert(0, "guard", tag)
        C.insert(0, "guard", tag)
        out.append((B, C))
        report(B, C, f"GUARD {tag} ({big_pub if tag == 'drop_pub' else big_chem} removed)")
    pd.concat([b for b, _ in out]).to_csv(HERE / "board_guard.csv", index=False)
    pd.concat([c for _, c in out]).to_csv(HERE / "contrasts_guard.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    if what in ("main", "all"):
        run_main()
    if what in ("forced", "all"):
        F.reset_caches()
        run_forced()
    if what in ("guard", "all"):
        F.reset_caches()
        run_guard()
