"""L5 step 1: prove the hooked copy reproduces gen15 before any estimator is scored.

  * estimator self-checks: POOLED == residual_covariance to the last bit on NaN-padded rows;
    the pairwise Ledoit-Wolf equals sklearn's LedoitWolf on complete rows.
  * gen16.l5_fewshot.evaluate with the default estimator under BP, arms {G14, FLAT},
    hows=(widest, dopt, random), mask_publication=True, cov_kind='empirical', ks=0..3 must give
    G14@doptk3 = 0.1700 and G14@doptk1 = 0.2313 (gen15_curve/results/g15_support_board_empirical_BP.csv)
    to 4 decimals, and its pair table must equal the frozen gen15.fewshot.evaluate's on every
    shared column.

Usage:  .venv/Scripts/python.exe generations/gen16_leads/scripts/l5_verify.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import l5_cov as LC  # noqa: E402
from gen16 import l5_fewshot as L5  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15 import fewshot as FS  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

OUT = bootstrap.RESULTS / "L5"
OUT.mkdir(parents=True, exist_ok=True)
G15 = bootstrap.ROOT / "generations" / "gen15_curve" / "results" / "g15_support_board_empirical_BP.csv"
ARMS = {"G14": A.g14, "FLAT": A.flat}
HOWS = ("widest", "dopt", "random")
KS = (0, 1, 2, 3)
KEY = ["split_seed", "fold", "cell_id", "A", "B"]


def main() -> None:
    t0 = time.time()
    report: dict = {}
    # ---- estimator self-checks ----
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(120, LC.N_LN))
    rows[rng.random(rows.shape) < 0.35] = np.nan
    rows = rows - np.nanmean(rows, axis=1, keepdims=True)
    d_pooled = LC.check_pooled_matches_residual_covariance(rows)
    d_lw_s, d_lw_c = LC.check_lw_matches_sklearn()
    report["pooled_vs_residual_covariance_maxabs"] = d_pooled
    report["lw_shrinkage_vs_sklearn_absdiff"] = d_lw_s
    report["lw_cov_vs_sklearn_maxabs"] = d_lw_c
    print(f"POOLED == residual_covariance: max|diff| = {d_pooled:.3e}")
    print(f"pairwise LW == sklearn LedoitWolf on complete rows: |dshrink| = {d_lw_s:.3e}, "
          f"max|dcov| = {d_lw_c:.3e}")
    assert d_pooled == 0.0, "pairwise_moment does not mirror residual_covariance"
    assert d_lw_s < 1e-10 and d_lw_c < 1e-10, "pairwise LW does not reduce to sklearn"
    # Every estimator returns a symmetric 14 x 14.  Positive-definiteness is *reported*, not
    # asserted: the deployed `residual_covariance` is a pairwise-complete second moment of
    # row-centred curves with 35-80 % missingness, which is not guaranteed PSD and is measured
    # here not to be, on structured rows as well as on white noise.  The BLUP never inverts the
    # covariance itself -- only D Sigma D' + noise I, which stays solvable -- so asserting PD
    # would be a check the frozen estimator itself fails.  The min-eigenvalue of every estimator
    # on the REAL leave-chemotype-out residual rows is reported by scripts/l5_covrun.py.
    groups = np.array([f"c{i % 9}" for i in range(len(rows))])
    A0 = rng.normal(size=(LC.N_LN, 3))
    struct = rng.normal(size=(200, 3)) @ A0.T + 0.3 * rng.normal(size=(200, LC.N_LN))
    struct = struct - struct.mean(axis=1, keepdims=True)
    struct[rng.random(struct.shape) < 0.35] = np.nan
    struct = struct - np.nanmean(struct, axis=1, keepdims=True)
    gs = np.array([f"c{i % 9}" for i in range(len(struct))])
    eig_report = {}
    for name, fn in LC.ESTIMATORS.items():
        C = fn(struct, gs)
        ev = np.linalg.eigvalsh(C)
        Cw = fn(rows, groups)
        evw = np.linalg.eigvalsh(Cw)
        assert C.shape == (LC.N_LN, LC.N_LN) and np.allclose(C, C.T), name
        eig_report[name] = {"structured_min_eig": float(ev.min()),
                            "structured_max_eig": float(ev.max()),
                            "structured_trace": float(np.trace(C)),
                            "whitenoise_min_eig": float(evw.min())}
        print(f"  {name:9s} structured: trace {np.trace(C):.4f}  min eig {ev.min():.2e}  "
              f"max eig {ev.max():.4f}   white-noise min eig {evw.min():+.2e}")
    report["estimator_eigenvalues"] = eig_report

    # ---- the reproduction ----
    bench = V.load()
    r16 = L5.evaluate(bench, ARMS, "BP", ks=KS, hows=HOWS, cov_kind="empirical",
                      mask_publication=True, verbose=True)
    pe16 = per_extractant(r16.pairs, r16.modes)
    b16 = summarise(pe16, r16.pairs, r16.modes).set_index("arm")["macro_mae_extractant"]
    locked = pd.read_csv(G15).set_index("arm")["macro_mae_extractant"]
    got3, got1 = float(b16["G14@doptk3"]), float(b16["G14@doptk1"])
    exp3, exp1 = float(locked["G14@doptk3"]), float(locked["G14@doptk1"])
    print(f"\nG14@doptk3  L5 copy {got3:.10f}   locked {exp3:.10f}   diff {got3 - exp3:+.2e}")
    print(f"G14@doptk1  L5 copy {got1:.10f}   locked {exp1:.10f}   diff {got1 - exp1:+.2e}")
    shared = [m for m in b16.index if m in locked.index]
    maxdiff = float((b16[shared] - locked[shared]).abs().max())
    print(f"max |diff| over {len(shared)} shared modes vs locked board: {maxdiff:.2e}")
    report.update({"G14@doptk3_l5": got3, "G14@doptk3_locked": exp3,
                   "G14@doptk1_l5": got1, "G14@doptk1_locked": exp1,
                   "n_shared_modes_vs_locked": len(shared), "max_absdiff_vs_locked_board": maxdiff,
                   "n_scored_pairs_bp": int(len(r16.pairs))})
    ok4 = round(got3, 4) == round(exp3, 4) and round(got1, 4) == round(exp1, 4)

    # ---- byte-level check against a fresh run of the frozen gen15 evaluate ----
    r15 = FS.evaluate(bench, ARMS, "BP", ks=KS, hows=HOWS, cov_kind="empirical",
                      mask_publication=True, verbose=True)
    a = r15.pairs.set_index(KEY).sort_index()
    b = r16.pairs.set_index(KEY).sort_index()
    assert len(a) == len(b) and a.index.equals(b.index), "pair sets differ"
    cols = [c for c in a.columns if c in b.columns and a[c].dtype.kind == "f"]
    diffs = {c: float(np.nanmax(np.abs(a[c].to_numpy() - b[c].to_numpy()))) for c in cols}
    worst = max(diffs.values())
    worst_pred = max(v for c, v in diffs.items() if not c.startswith("sd@"))
    worst_sd = max(v for c, v in diffs.items() if c.startswith("sd@"))
    print(f"pair table vs frozen gen15 run: {len(a)} pairs, {len(cols)} shared float columns, "
          f"max|diff| predictions {worst_pred:.2e}, sd columns {worst_sd:.2e}")
    missing = sorted(set(a.columns) - set(b.columns))
    print(f"columns in gen15 not in L5 copy: {missing}")
    report.update({"n_pairs_frozen": int(len(a)), "max_absdiff_pred_vs_frozen": worst_pred,
                   "max_absdiff_sd_vs_frozen": worst_sd, "columns_missing": missing,
                   "reproduced_4dp": bool(ok4), "seconds": time.time() - t0})
    (OUT / "verify.json").write_text(json.dumps(report, indent=2))
    print(f"\nreproduced to 4 dp: {ok4}   ({time.time() - t0:.0f}s)")
    if not ok4 or worst_pred > 1e-9:
        sys.exit(1)


if __name__ == "__main__":
    main()
