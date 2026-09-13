"""Second stage of the lens-B refutation of CLAIM L3A.

The registered null permutes the model's sign calls *independently inside each of the 455 tasks*.
A real rule is not like that: it assigns one call per extractant per requested direction and uses
the same call in every one of the 91 pairs.  This file builds the coherent null -- a random
per-extractant score, width-matched to G14's own per-task deferral counts -- and re-runs the
headline against it, plus the remaining bookkeeping checks.

Usage (repo root):
  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe \
      generations/gen16_leads/scripts/refute_l3a_B_extra.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen16 import bootstrap                      # noqa: E402
from gen16 import l3_decision as L               # noqa: E402
from refute_l3a_B import Board, OUT, ext_meta, width_matched, bh   # noqa: E402

R_NULL = 2000


def coherent_null(bd: Board, reps: int = R_NULL, seed: int = 20260911) -> np.ndarray:
    """Seed-macro ``saved`` for ``reps`` random per-extractant scores, each used to defer exactly
    as many candidates as G14 defers in every task (width-matched, coherent across tasks)."""
    T = bd.T
    rng = np.random.default_rng(seed)
    S = rng.random((reps, bd.n_ext))
    out = np.zeros((reps, T.n))
    for t in range(T.n):
        idx, obs, call = T.idx[t], T.obs[t], bd.calls[t]
        n = len(idx)
        n_light, n_heavy = int((call == 1).sum()), int((call == -1).sum())
        st = S[:, idx]
        srt = np.sort(st, axis=1)
        top = (st >= srt[:, [n - n_light]]) if n_light else np.zeros_like(st, bool)
        bot = (st <= srt[:, [n_heavy - 1]]) if n_heavy else np.zeros_like(st, bool)
        sv = []
        for d in (-1, 1):
            succ = (np.sign(obs) == d) & (np.abs(obs) >= L.STRONG)
            K = int(succ.sum())
            kept = ~top if d == -1 else ~bot          # a heavy request defers the 'light' calls
            Nk = kept.sum(1)
            Kk = (kept & succ[None, :]).sum(1)
            sv.append(L.expected_draws(n, K) - L.e_model_from_counts(n, K, Nk, Kk))
        out[:, t] = 0.5 * (sv[0] + sv[1])
    return L.seed_macro(out, T.seeds)


def main() -> int:
    t0 = time.time()
    em = ext_meta()
    rows, notes = [], {}

    for d in L.DESIGNS:
        bd = Board(d, em)
        T = bd.T
        base = bd.point()

        # --- coherent, width-matched random ligand-level rule (the null the claim needs)
        cn = coherent_null(bd)
        rows.append(dict(design=d, check="coherent_random_null", point=base,
                         null_mean=float(np.mean(cn)), null_sd=float(np.std(cn)),
                         null_q95=float(np.quantile(cn, 0.95)),
                         null_q975=float(np.quantile(cn, 0.975)),
                         p_vs_null=float((cn >= base).mean()),
                         z=float((base - np.mean(cn)) / np.std(cn))))

        # --- covariate width-matched competitors, both signs (no sign cherry-picked)
        for lab, sc in (("dentate", bd.dent), ("nmetals", bd.nmet), ("absa", bd.absa)):
            for sg, tag in ((+1, "hi"), (-1, "lo")):
                v = bd.saved_for_calls(width_matched(bd, sc, sg))
                rows.append(dict(design=d, check=f"wm_{lab}_{tag}", point=v,
                                 p_vs_null=float((cn >= v).mean()),
                                 z=float((v - np.mean(cn)) / np.std(cn))))

        # --- constant rules must be exactly 0
        for lab, c in (("HEAVIER_ALWAYS", -1), ("LIGHTER_ALWAYS", +1), ("NO_CALL", 0)):
            calls = [np.full(len(o), c, int) for o in T.obs]
            rows.append(dict(design=d, check=f"constant_{lab}", point=bd.saved_for_calls(calls)))

        # --- how coherent is the model's own call?  (per seed, over the 91 pairs)
        cons = []
        for s in np.unique(T.seeds):
            ts = np.flatnonzero(T.seeds == s)
            per_ext = {}
            for t in ts:
                for e, c in zip(T.idx[t], bd.calls[t]):
                    per_ext.setdefault(e, []).append(int(c))
            cons.append(np.mean([len(set(v)) == 1 for v in per_ext.values()]))
        rows.append(dict(design=d, check="call_constant_across_pairs_frac",
                         point=float(np.mean(cons))))

        # --- fractions, not only absolute measurements, for the subset re-runs
        keep = bd.chem != "sc009"
        for lab, w in (("all", np.ones(bd.n_ext)), ("drop_sc009", keep.astype(float)),
                       ("nmet_5_8", ((bd.nmet >= 5) & (bd.nmet <= 8)).astype(float)),
                       ("nmet_9_14", ((bd.nmet >= 9) & (bd.nmet <= 14)).astype(float))):
            r = L.l3a_saved_weighted(w[None, :], bd.mats)
            sv = float(np.atleast_1d(L.seed_macro(r["saved"], T.seeds))[0])
            er = float(np.atleast_1d(L.seed_macro(r["e_random"], T.seeds))[0])
            em2 = float(np.atleast_1d(L.seed_macro(r["e_model"], T.seeds))[0])
            rows.append(dict(design=d, check=f"frac_{lab}", point=sv,
                             e_random=er, e_model=em2,
                             saved_frac=sv / er,
                             n_tasks=int(np.isfinite(r["saved"][0]).sum())))
        print(f"[{d}] extra done t={time.time()-t0:.0f}s", flush=True)

    # --- structural: is the observed side of a pair identical across the five seeds?
    bd = Board("BP", em)
    T = bd.T
    sig = {}
    for t in range(T.n):
        k = f"{T.A[t]}|{T.B[t]}"
        order = np.argsort(T.idx[t])
        sig.setdefault(k, []).append((tuple(np.asarray(T.idx[t])[order]),
                                      tuple(np.round(np.asarray(T.obs[t])[order], 12))))
    notes["observed_side_identical_across_seeds_sorted"] = bool(
        all(all(s == v[0] for s in v[1:]) for v in sig.values()))
    notes["n_distinct_pairs"] = len(sig)

    # --- Kish effective number of blocks over the 90 extractants
    for lab, blocks in (("chemotype", bd.chem),
                        ("publication", np.array([p[0] for p in bd.pubs]))):
        c = pd.Series(blocks).value_counts().to_numpy(float)
        notes[f"kish_neff_{lab}"] = float(c.sum() ** 2 / (c ** 2).sum())
        notes[f"n_blocks_{lab}"] = int(len(c))
        notes[f"largest_block_share_{lab}"] = float(c.max() / c.sum())

    # --- comparison family and BH
    fam = []
    for f in sorted((bootstrap.RESULTS).glob("*/*contrasts*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "family" in df.columns:
            df["__file"] = f.name
            fam.append(df)
    if fam:
        F = pd.concat(fam, ignore_index=True)
        notes["contrast_rows_total"] = int(len(F))
        notes["contrast_rows_by_family"] = {k: int(v) for k, v in F.family.value_counts().items()}
        notes["contrast_files"] = sorted(F.__file.unique().tolist())
        for famname in ("registered", "exploratory"):
            sub = F[F.family == famname].copy()
            if not len(sub):
                continue
            for pcol in ("p_registered", "p_two_sided"):
                if pcol in sub.columns:
                    sub[f"bh_{pcol}"] = bh(sub[pcol].to_numpy())
            notes[f"bh_{famname}_n"] = int(len(sub))
            l3a = sub[sub.comparison.astype(str).str.fullmatch("G14_saved_vs_0")]
            if len(l3a):
                notes[f"bh_{famname}_L3A"] = {
                    r.design: {"p_registered": float(r.p_registered),
                               "bh_p_registered": float(r.bh_p_registered),
                               "p_two_sided": float(r.p_two_sided),
                               "bh_p_two_sided": float(r.bh_p_two_sided)}
                    for r in l3a.itertuples()}

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "refute_l3a_B_extra.csv", index=False)
    (OUT / "refute_l3a_B_extra_notes.json").write_text(
        json.dumps(notes, indent=2, default=str), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 400):
        print(D.to_string())
    print(json.dumps(notes, indent=2, default=str))
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
