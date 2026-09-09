"""Three things the four main tables do not carry.

1. Q3 stratified by how far apart the target metals are: the design question is asked mostly about
   *adjacent* pairs, and the pooled number hides that the curve representation behaves completely
   differently there.
2. A finer risk-coverage curve, with the across-seed spread, so 'does the model know when it does
   not know' has an error bar.
3. The programme's own paired inference (gen13's chemotype-blocked bootstrap, reused unchanged) on
   every decision contrast, so each gain carries an interval and a p rather than a bare point.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from gen15 import valuebench as V              # noqa: E402
from gen13sep.metals import LANTHANIDES, ATOMIC_NUMBER  # noqa: E402
from gen13sep.inference import paired_contrasts        # noqa: E402
import decmetrics as M                          # noqa: E402
from run_metrics import DESIGNS, ARMS, REF, load_design  # noqa: E402

OUT = HERE / "results"
ARMSET = ARMS + REF + ["G14_TIED"]
DZ_BANDS3 = [(1, 1, "adjacent (dZ=1)"), (2, 4, "near (dZ 2-4)"),
             (5, 8, "far (dZ 5-8)"), (9, 13, "extreme (dZ 9-13)")]
COV = (1.0, 0.75, 0.5, 0.25, 0.10, 0.05)


def q3_by_dz(per_pair: pd.DataFrame) -> pd.DataFrame:
    z = {m: ATOMIC_NUMBER[m] for m in LANTHANIDES}
    per_pair = per_pair.assign(dZ=per_pair["B"].map(z) - per_pair["A"].map(z))
    rows = []
    for lo, hi, lab in DZ_BANDS3:
        blk = per_pair[(per_pair["dZ"] >= lo) & (per_pair["dZ"] <= hi)]
        if blk.empty:
            continue
        for arm, s in blk.groupby("arm"):
            per_seed = s.groupby("split_seed")[["spearman", "top1", "regret"]].mean()
            rows.append({"band": lab, "arm": arm,
                         "n_pair_tasks": int(len(s)),
                         "obs_spread": float(s["obs_spread"].mean()),
                         "spearman": float(per_seed["spearman"].mean()),
                         "top1": float(per_seed["top1"].mean()),
                         "regret": float(per_seed["regret"].mean()),
                         "regret_sd": float(per_seed["regret"].std(ddof=1))})
    out = pd.DataFrame(rows)
    # share of the achievable regret reduction the arm captures
    ref = out[out["arm"] == "_RANDOM"].set_index("band")["regret"]
    orc = out[out["arm"] == "O_BOTH"].set_index("band")["regret"]
    out["frac_of_oracle"] = [(ref[b] - r) / max(ref[b] - orc[b], 1e-9)
                             for b, r in zip(out["band"], out["regret"])]
    return out


def coverage_sd(tab: pd.DataFrame, arm: str, conf: np.ndarray, label: str) -> pd.DataFrame:
    """Risk-coverage with a per-seed spread; the honest baseline is the coverage=1 row."""
    rows = []
    for s, blk in tab.groupby("split_seed"):
        c = M.coverage(blk, arm, conf[tab["split_seed"].to_numpy() == s], coverages=COV)
        c["split_seed"] = s
        rows.append(c)
    per_seed = pd.concat(rows, ignore_index=True)
    g = per_seed.groupby("coverage").agg(mae=("mae", "mean"), mae_sd=("mae", "std"),
                                         sign_acc=("sign_acc", "mean"),
                                         sign_acc_sd=("sign_acc", "std"),
                                         n_pairs=("n_pairs", "sum")).reset_index()
    g.insert(0, "confidence", label)
    g.insert(0, "arm", arm)
    return g


def unit_frame(long: pd.DataFrame, value_col: str, target: str) -> pd.DataFrame:
    """Reshape a per-unit decision table into gen13's ``per_extractant`` contract."""
    keep = ["split_seed", "extractant", "chemotype", "arm"]
    out = long.groupby(keep, sort=False)[value_col].mean().reset_index()
    return out.rename(columns={value_col: target})


def main() -> None:
    bench = V.load()
    dz, cov, inf, pp_all = [], [], [], []
    for d in DESIGNS:
        tab = load_design(bench, d)
        units = M.sign_units(tab, ARMSET)
        _, cells = M.within_cell(tab, ARMSET)
        _, per_pair = M.cross_extractant(tab, ARMSET)
        per_pair.insert(0, "design", d)
        pp_all.append(per_pair)

        t = q3_by_dz(per_pair)
        t.insert(0, "design", d)
        dz.append(t)

        for arm in ARMSET:
            c = coverage_sd(tab, arm, tab[arm].abs().to_numpy(), "|prediction|")
            c.insert(0, "design", d)
            cov.append(c)
        c = coverage_sd(tab, "G14", tab["dir_conf"].to_numpy(), "gen14 |p-0.5|")
        c.insert(0, "design", d)
        cov.append(c)

        # ---- paired inference, gen13's chemotype-blocked bootstrap ----------------------
        q1u = unit_frame(units[units["band"] == "all"], "hit", "sign_acc_strong")
        cmp1 = {"G14_vs_HEAVIER": ("HEAVIER_ALWAYS", "G14"),
                "G13_vs_HEAVIER": ("HEAVIER_ALWAYS", "G13_FULL"),
                "G14_vs_G13": ("G13_FULL", "G14"),
                "G14_vs_FLAT": ("FLAT", "G14")}
        a = paired_contrasts(q1u, cmp1, value="sign_acc_strong", replicates=10_000)
        a.insert(0, "metric", "Q1 sign accuracy")

        q2s = unit_frame(cells, "spearman", "pair_spearman")
        b = paired_contrasts(q2s, {"G14_vs_HEAVIER": ("HEAVIER_ALWAYS", "G14"),
                                   "G14_vs_G13": ("G13_FULL", "G14")},
                             value="pair_spearman", replicates=10_000)
        b.insert(0, "metric", "Q2 within-cell Spearman")

        q2t = unit_frame(cells, "top1", "pair_spearman")
        c2 = paired_contrasts(q2t, {"G14_vs_HEAVIER": ("HEAVIER_ALWAYS", "G14"),
                                    "OBOTH_vs_HEAVIER": ("HEAVIER_ALWAYS", "O_BOTH")},
                              value="pair_spearman", replicates=10_000)
        c2.insert(0, "metric", "Q2 best-pair top-1")

        # Q3: the unit is the metal-pair task; blocks are the metal pairs themselves, which are
        # NOT independent (they share cells and metals), so this interval is the optimistic one.
        q3 = per_pair.assign(extractant=per_pair["A"] + "/" + per_pair["B"],
                             chemotype=per_pair["A"] + "/" + per_pair["B"])
        e = paired_contrasts(unit_frame(q3, "regret", "mae_all"),
                             {"G14_vs_RANDOM": ("_RANDOM", "G14"),
                              "G13_vs_RANDOM": ("_RANDOM", "G13_FULL"),
                              "G14TIED_vs_G14": ("G14", "G14_TIED"),
                              "OBOTH_vs_RANDOM": ("_RANDOM", "O_BOTH")},
                             value="mae_all", replicates=10_000)
        e.insert(0, "metric", "Q3 cross-extractant regret")
        f = paired_contrasts(unit_frame(q3, "spearman", "pair_spearman"),
                             {"G14_vs_RANDOM": ("_RANDOM", "G14"),
                              "G13_vs_RANDOM": ("_RANDOM", "G13_FULL")},
                             value="pair_spearman", replicates=10_000)
        f.insert(0, "metric", "Q3 cross-extractant Spearman")

        for part in (a, b, c2, e, f):
            part.insert(0, "design", d)
            inf.append(part)
        print(f"[{d}] extras done", flush=True)

    pd.concat(dz, ignore_index=True).to_csv(OUT / "q3_by_dz.csv", index=False)
    pd.concat(pp_all, ignore_index=True).to_csv(OUT / "q3_per_pair_long.csv", index=False)
    pd.concat(cov, ignore_index=True).to_csv(OUT / "q4_coverage_sd.csv", index=False)
    pd.concat(inf, ignore_index=True).to_csv(OUT / "paired_contrasts.csv", index=False)
    print("written")


if __name__ == "__main__":
    main()
