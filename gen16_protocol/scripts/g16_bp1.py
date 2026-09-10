"""BP1 / BP1X: the conditional hold-out, reported next to BP.

BP answers "a new ligand in a new laboratory".  BP1 answers "this laboratory has already published
this system; predict a new condition set", and BP1X the strict anchor case "same laboratory, same
ligand, new conditions".  Reporting the pair is a more honest description of deployment than either
alone, and BP1X is the design under which one-measurement / anchor arms are well posed.

Every p-value here is a restricted wild cluster bootstrap-t over chemotypes (gen16.clusterboot),
not the percentile block bootstrap, because g16_size.py shows the percentile bootstrap rejects a
true null at 7-8 % on this corpus's own cluster membership.

Usage:  python gen16_protocol/scripts/g16_bp1.py [--full]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "gen13_separation", ROOT / "gen14_direction",
          ROOT / "gen15_curve", ROOT / "gen16_protocol"):
    sys.path.insert(0, str(p))

from gen13sep.metrics import per_extractant, summarise      # noqa: E402
from gen13sep.splits import all_folds                       # noqa: E402
from gen14.dirbench import load, feature_sets               # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS            # noqa: E402
from gen15 import arms as A                                 # noqa: E402
from gen16.clusterboot import wcr_test, percentile_block_p  # noqa: E402
from gen16.designs import bp1_coverage, bp1_folds, run_folds  # noqa: E402

FULL = "--full" in sys.argv
OUT = ROOT / "gen16_protocol" / "results"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = {"FLAT": A.flat, "MEAN_CURVE": A.mean_curve, "G14": A.g14, "O_AMP": A.o_amp}
if FULL:
    ARMS["G13_FULL"] = A.g13_full

bench = load()
frame = bench.frame
X = bench.matrix(LEAN_BLOCKS)
fs = feature_sets(bench)

# ---------------------------------------------------------------- coverage, before any fitting
cov = pd.DataFrame([bp1_coverage(frame), bp1_coverage(frame, require_extractant=True)])
print("=== what each conditional design can score ===")
print(cov.to_string(index=False))
cov.to_csv(OUT / "g16_bp1_coverage.csv", index=False)

# ---------------------------------------------------------------- score BP, BP1, BP1X
def scorable(require_extractant: bool) -> set:
    """The cell ids a conditional design can score, so BP can be matched to the same cohort."""
    rich = frame[frame.n_metals >= 5]
    g = rich.groupby("publication_id")["condition_key"].nunique()
    keep = rich[rich.publication_id.isin(set(g.index[g >= 2]))]
    if require_extractant:
        pe = rich.groupby(["publication_id", "extractant"])["condition_key"].nunique()
        ok = set(pe.index[pe >= 2])
        keep = keep[[(p, e) in ok for p, e in zip(keep.publication_id, keep.extractant)]]
    return set(keep.cell_id)


def restrict(folds, cell_ids: set):
    """BP folds whose *test* set is cut down to a cohort, so BP - BP1 is not a cohort difference."""
    from dataclasses import replace
    cid = frame.cell_id.to_numpy()
    out = []
    for f in folds:
        te = f.test_index[[cid[i] in cell_ids for i in f.test_index]]
        if len(te):
            out.append(replace(f, test_index=te))
    return out


bp = all_folds(frame, design="BP")
plans = {
    "BP": bp,
    "BP@S": restrict(bp, scorable(False)),      # BP, scored only where BP1 can be scored
    "BP1": bp1_folds(frame),
    "BP@SX": restrict(bp, scorable(True)),      # BP, scored only where BP1X can be scored
    "BP1X": bp1_folds(frame, require_extractant=True),
}
boards, pes = [], {}
for design, folds in plans.items():
    t0 = time.time()
    n_te = sum(len(f.test_index) for f in folds)
    tab = run_folds(bench, ARMS, folds, design=design, X=X, fs=fs)
    pe = per_extractant(tab, list(ARMS))
    bd = summarise(pe, tab, list(ARMS))
    bd.insert(0, "design", design)
    bd.insert(1, "n_folds", len(folds))
    bd.insert(2, "n_test_cells", n_te)
    boards.append(bd)
    pes[design] = pe
    print(f"  {design}: {len(folds)} folds, {n_te} scored cells, {time.time() - t0:.0f}s",
          flush=True)

B = pd.concat(boards, ignore_index=True)
B.to_csv(OUT / "g16_bp1_board.csv", index=False)
print("\n=== macro MAE of log SF, per extractant ===")
print(B.pivot(index="arm", columns="design", values="macro_mae_extractant")
       .reindex(list(ARMS))[list(plans)].round(4).to_string())
print("\nBP@S / BP@SX are BP scored on the conditional designs' own cohorts: the honest")
print("comparison for 'what does seeing the laboratory buy' is BP@S -> BP1, not BP -> BP1.")


# ---------------------------------------------------------------- cluster-robust contrasts
def delta(pe: pd.DataFrame, ref: str, cand: str, value: str = "mae_all"):
    """Per-extractant paired difference ref - cand (positive = candidate better), + its chemotype."""
    w = (pe.groupby(["extractant", "chemotype", "arm"])[value].mean()
           .unstack("arm").reset_index().dropna(subset=[ref, cand]))
    return (w[ref] - w[cand]).to_numpy(), w["chemotype"].to_numpy()


COMPS = [("MEAN_CURVE", "G14"), ("FLAT", "G14"), ("G14", "O_AMP")]
rows = []
for design, pe in pes.items():
    for ref, cand in COMPS:
        if ref not in pe.arm.unique() or cand not in pe.arm.unique():
            continue
        d, cl = delta(pe, ref, cand)
        r = wcr_test(d, cl, comparison=f"{ref} - {cand}", reps=9_999)
        rows.append({"design": design, "comparison": f"{ref} - {cand}", "gain": r.point,
                     "se_cv1": r.se_cv1, "se_cv3": r.se_cv3, "p_wcr": r.p_wcr,
                     "p_cv1_t": r.p_cv1_t,
                     "p_percentile_old": percentile_block_p(d, cl, reps=10_000),
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "G": r.G,
                     "G_star": r.G_star, "n_units": r.n_units})
# cross-design, same cohort, same arm: what does seeing the laboratory actually buy?
for arm in ARMS:
    for ref, cand in (("BP@S", "BP1"), ("BP@SX", "BP1X")):
        a = pes[ref].query("arm == @arm").groupby(["extractant", "chemotype"])["mae_all"].mean()
        b = pes[cand].query("arm == @arm").groupby(["extractant", "chemotype"])["mae_all"].mean()
        j = pd.concat([a.rename("r"), b.rename("c")], axis=1).dropna().reset_index()
        if len(j) < 8:
            continue
        r = wcr_test((j["r"] - j["c"]).to_numpy(), j["chemotype"].to_numpy(),
                     comparison=f"{arm}: {ref} - {cand}", reps=9_999)
        rows.append({"design": "cross", "comparison": f"{arm}: {ref} - {cand}", "gain": r.point,
                     "se_cv1": r.se_cv1, "se_cv3": r.se_cv3, "p_wcr": r.p_wcr,
                     "p_cv1_t": r.p_cv1_t, "p_percentile_old": np.nan,
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "G": r.G,
                     "G_star": r.G_star, "n_units": r.n_units})

C = pd.DataFrame(rows)
C["p_percentile_old"] = [v["p_percentile"] if isinstance(v, dict) else v
                         for v in C["p_percentile_old"]]
C.to_csv(OUT / "g16_bp1_contrasts.csv", index=False)
print("\n=== gains, wild cluster bootstrap-t over chemotypes ===")
print(C.round(4).to_string(index=False))
print("\nNoise floor: replicate sd of the amplitude 0.237; direction-label ceiling 0.891 macro acc.")
print("BP1/BP1X are CONDITIONAL designs -- the extractant is on both sides on purpose.")
