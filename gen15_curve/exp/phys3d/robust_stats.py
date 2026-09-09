"""Are the phys3d associations bigger than what 28 columns find by chance?

Two guards, both blocked on the chemotype, because the chemotype is the unit designs B and BP
hold out and the unit this corpus's effective sample size is counted in:

1. a chemotype-blocked bootstrap CI for every descriptor's Spearman with a, |a| and b
   (resample the 45 chemotypes with replacement, keep all of a chemotype's extractants);
2. a family-wise permutation null: at the chemotype level (45 units), permute the target across
   chemotypes and record the largest |rho| over all 28 descriptors.  The 95th percentile of that
   maximum is the bar a descriptor has to clear before "the strongest of 28 columns" means
   anything at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPS = 2000


def main() -> None:
    df = pd.read_parquet(HERE / "extractant_targets.parquet")
    cols = [c for c in df.columns if c.startswith("phys3d__")]
    chem = df["chemotype"].to_numpy()
    uch = np.unique(chem)
    idx_of = {c: np.flatnonzero(chem == c) for c in uch}
    rng = np.random.default_rng(20260909)

    # ---- 1. chemotype-blocked bootstrap CI ------------------------------------------------
    rows = []
    boots = [np.concatenate([idx_of[c] for c in rng.choice(uch, size=len(uch), replace=True)])
             for _ in range(REPS)]
    for tgt in ("a", "abs_a", "b"):
        y = df[tgt].to_numpy(dtype=float)
        for c in cols:
            x = df[c].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 20:
                continue
            point = stats.spearmanr(x[ok], y[ok]).statistic
            vals = []
            for b in boots:
                bb = b[np.isfinite(x[b]) & np.isfinite(y[b])]
                if len(bb) < 20 or len(np.unique(x[bb])) < 5:
                    continue
                r = stats.spearmanr(x[bb], y[bb]).statistic
                if np.isfinite(r):
                    vals.append(r)
            lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) > 100 else (np.nan, np.nan)
            rows.append({"target": tgt, "descriptor": c.replace("phys3d__", ""),
                         "rho": point, "ci_lo": lo, "ci_hi": hi,
                         "excludes_zero": bool(np.isfinite(lo) and lo * hi > 0)})
    boot = pd.DataFrame(rows)
    boot.to_csv(HERE / "robust_bootstrap.csv", index=False)

    # ---- 2. family-wise permutation null at the chemotype level ---------------------------
    cm = df.groupby("chemotype")[cols + ["a", "abs_a", "b"]].mean()
    M = cm[cols].to_numpy(dtype=float)
    print(f"chemotype-level units: {len(cm)}   descriptors: {len(cols)}\n")
    for tgt in ("a", "abs_a", "b"):
        y = cm[tgt].to_numpy(dtype=float)
        obs = []
        for j in range(M.shape[1]):
            ok = np.isfinite(M[:, j]) & np.isfinite(y)
            obs.append(abs(stats.spearmanr(M[ok, j], y[ok]).statistic) if ok.sum() >= 10 else np.nan)
        obs = np.array(obs, dtype=float)
        null = np.empty(REPS)
        for k in range(REPS):
            yp = rng.permutation(y)
            best = 0.0
            for j in range(M.shape[1]):
                ok = np.isfinite(M[:, j]) & np.isfinite(yp)
                if ok.sum() < 10:
                    continue
                r = abs(stats.spearmanr(M[ok, j], yp[ok]).statistic)
                if np.isfinite(r) and r > best:
                    best = r
            null[k] = best
        thr = float(np.percentile(null, 95))
        order = np.argsort(-np.nan_to_num(obs))
        top = [(cols[j].replace("phys3d__", ""), obs[j]) for j in order[:5]]
        p_fw = float((null >= np.nanmax(obs)).mean())
        print(f"=== {tgt} (chemotype means) ===")
        print(f"  family-wise 95th percentile of max|rho| under permutation: {thr:.3f}")
        print(f"  observed max|rho| = {np.nanmax(obs):.3f}  ({top[0][0]})  family-wise p = {p_fw:.3f}")
        print("  top 5: " + "  ".join(f"{n} {v:.3f}" for n, v in top))
        print(f"  descriptors clearing the family-wise bar: "
              f"{[cols[j].replace('phys3d__', '') for j in range(len(cols)) if np.nan_to_num(obs[j]) >= thr]}\n")

    print("chemotype-blocked bootstrap, descriptors whose 95 % CI excludes zero:")
    keep = boot[boot["excludes_zero"]].copy()
    keep["absrho"] = keep["rho"].astype(float).abs()
    keep = keep.sort_values(["target", "absrho"], ascending=[True, False]).drop(columns="absrho")
    print(keep.round(3).to_string(index=False) if len(keep) else "  none")


if __name__ == "__main__":
    main()
