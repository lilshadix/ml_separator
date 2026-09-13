"""Audit the programme's inference: pairs cluster bootstrap vs wild cluster bootstrap-t.

Three things, in order:

1. **Calibration.**  A Monte Carlo under H0 with the corpus's *actual* chemotype-size profile
   (45 chemotypes over 90 extractants, the largest holding 23) measures the true size of every
   procedure the repository uses or could use, at nominal 5 %.  This turns "the pairs bootstrap
   is known to over-reject" into a number for this design.

2. **Re-inference.**  Every saved paired contrast is recomputed with the WCR bootstrap-t, the
   CR2 t-test on Bell-McCaffrey degrees of freedom, a Bayesian-bootstrap ROPE and a
   random-effects ROPE.  Output goes next to the original table as ``*_wildcluster.csv``.

3. **Detectability.**  The minimum detectable MAE difference at 80 % power under the honest SE,
   and the smallest between-chemotype R^2 a ligand covariate could have and still be found.

Usage
-----
    python gen13_separation/scripts/g13_inference_audit.py                 # calibration + gen13 BP
    python gen13_separation/scripts/g13_inference_audit.py <per_extractant.csv> [value]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen13sep import wildcluster as wc  # noqa: E402

HIGHER_IS_BETTER = frozenset({"sign_acc_strong", "pair_spearman", "curve_spearman", "hit",
                              "macro_accuracy"})
ROPE = 0.02          # the pre-registered practical margin, already in gen13sep.inference.MARGIN
REPS = 9999


# --------------------------------------------------------------------------------------
def calibration(sizes: np.ndarray, *, icc: float = 0.72, sims: int = 2000, reps: int = 999,
                seed: int = 20260909, tails: str = "normal") -> pd.DataFrame:
    """Rejection rate at nominal 5 % under H0, with the corpus's own cluster-size profile.

    Data generating process: ``d_i = a_{g(i)} + e_i`` with ``var(a)/(var(a)+var(e)) = icc``,
    mean zero.  ICC 0.72 is the measured chemotype-level ICC of the amplitude, so this is the
    dependence the real per-extractant differences carry.  ``tails='t3'`` draws both components
    from a scaled t(3) instead, because real MAE differences are heavy-tailed and skewed and the
    normal case flatters every procedure.
    """
    rng = np.random.default_rng(seed)
    G = len(sizes)
    n_g = sizes.astype(int)
    code = np.repeat(np.arange(G), n_g)
    N = len(code)
    sa, se = np.sqrt(icc), np.sqrt(1.0 - icc)
    from scipy import stats

    def noise(shape):
        if tails == "normal":
            return rng.standard_normal(shape)
        if tails == "t3":
            return rng.standard_t(3, size=shape) / np.sqrt(3.0)   # unit variance
        raise ValueError(tails)

    # the repository's pairs cluster bootstrap: the resample index matrix is fixed across sims,
    # so collapse it once into a (reps x N) multiplicity matrix and the whole bootstrap is a matmul
    picks = rng.integers(0, G, size=(reps, G))
    members = [np.flatnonzero(code == g) for g in range(G)]
    mult = np.zeros((reps, N))
    for b, row in enumerate(picks):
        idx = np.concatenate([members[j] for j in row])
        np.add.at(mult[b], idx, 1.0)
    mult /= mult.sum(axis=1, keepdims=True)

    keys = ("pairs_percentile_OLD", "iid_t", "cr1_normal", "cr1_tG1", "cr2_tG1", "cr2_bm",
            "cr3_bm", "wcr_cr2", "wcr_cr3")
    hits = {k: 0 for k in keys}
    for _ in range(sims):
        d = sa * noise(G)[code] + se * noise(N)
        draws = mult @ d
        p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
        hits["pairs_percentile_OLD"] += p < 0.05
        hits["iid_t"] += 2 * stats.t.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(N))), N - 1) < 0.05
        fit = wc.cluster_robust(d, code)
        t1, t2, t3 = fit.point / fit.se_cr1, fit.t_cr2, fit.point / fit.se_cr3
        hits["cr1_normal"] += 2 * stats.norm.sf(abs(t1)) < 0.05
        hits["cr1_tG1"] += 2 * stats.t.sf(abs(t1), G - 1) < 0.05
        hits["cr2_tG1"] += 2 * stats.t.sf(abs(t2), G - 1) < 0.05
        hits["cr2_bm"] += 2 * stats.t.sf(abs(t2), fit.dof_bm) < 0.05
        hits["cr3_bm"] += 2 * stats.t.sf(abs(t3), fit.dof_bm) < 0.05
        for kind, key in (("CR2", "wcr_cr2"), ("CR3", "wcr_cr3")):
            r = wc.wild_cluster_t(d, code, reps=reps, seed=int(rng.integers(1 << 30)),
                                  se_kind=kind, ci=False)
            hits[key] += r.p_wcr < 0.05
    mc_se = np.sqrt(0.05 * 0.95 / sims)
    out = pd.DataFrame([{"procedure": k, "rejection_rate_at_5pct": v / sims} for k, v in hits.items()])
    out["mc_se"] = round(float(mc_se), 4)
    out["sims"] = sims
    out["tails"] = tails
    out["icc"] = icc
    out["n_clusters"] = G
    out["kish_clusters"] = round(wc.kish(sizes), 2)
    return out


# --------------------------------------------------------------------------------------
def reinfer(per_ext: pd.DataFrame, comparisons: dict[str, tuple[str, str]], *,
            value: str = "mae_all", rope: float = ROPE) -> pd.DataFrame:
    """WCR bootstrap-t + ROPE posteriors for each (reference, candidate) pair."""
    sign = -1.0 if value in HIGHER_IS_BETTER else 1.0
    tab = (per_ext.assign(_v=sign * per_ext[value])
           .groupby(["arm", "extractant"])["_v"].mean().unstack("arm"))
    chem = (per_ext.drop_duplicates("extractant").set_index("extractant")["chemotype"]
            .astype(str).reindex(tab.index))
    rows = []
    for label, (ref, cand) in comparisons.items():
        if ref not in tab.columns or cand not in tab.columns:
            continue
        delta = (tab[ref] - tab[cand])
        ok = delta.notna()
        d, g = delta[ok].to_numpy(float), chem[ok].to_numpy()
        if len(d) < 5 or len(set(g)) < 3:
            continue
        fit = wc.cluster_robust(d, g)
        w = wc.wild_cluster_t(d, g, reps=REPS, se_kind="CR2")
        bbu = wc.bayes_bootstrap(d, g, rope=rope, estimand="unit_macro")
        bbc = wc.bayes_bootstrap(d, g, rope=rope, estimand="cluster_macro")
        re_ = wc.random_effects(d, g, rope=rope)
        # what the current code reports, recomputed on the same units for a like-for-like number
        rng = np.random.default_rng(8675309)
        names = sorted(set(g))
        members = [np.flatnonzero(g == b) for b in names]
        picks = rng.integers(0, len(names), size=(10000, len(names)))
        draws = np.array([d[np.concatenate([members[j] for j in row])].mean() for row in picks])
        p_old = float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))
        rows.append({
            "comparison": label, "value": value, "reference": ref, "candidate": cand,
            "point": fit.point, "n_units": fit.n, "n_chemotypes": fit.n_clusters,
            "kish_chemotypes": round(fit.kish_clusters, 2),
            "max_chemotype_share": round(fit.max_cluster_share, 3),
            "dof_bell_mccaffrey": round(fit.dof_bm, 2),
            "se_pairs_bootstrap": float(draws.std(ddof=1)),
            "se_cr1": fit.se_cr1, "se_cr2": fit.se_cr2, "se_cr3": fit.se_cr3,
            "p_pairs_percentile_OLD": p_old,
            "p_cr2_t_bm": w.p_t_dof, "p_wcr": w.p_wcr, "p_wcu": w.p_wcu,
            "wcr_ci_low": w.ci_low, "wcr_ci_high": w.ci_high,
            "mde80_OLD_2.80xse": 2.80 * float(draws.std(ddof=1)),
            "mde80_cr2_t": wc.detectable(fit.se_cr2, fit.dof_bm),
            "rope": rope,
            "P_better_unit": bbu.p_better, "P_rope_unit": bbu.p_rope, "P_worse_unit": bbu.p_worse,
            "P_better_chemo": bbc.p_better, "P_rope_chemo": bbc.p_rope, "P_worse_chemo": bbc.p_worse,
            "P_better_re": re_.p_better, "P_rope_re": re_.p_rope, "P_worse_re": re_.p_worse,
            "re_post_mean": re_.post_mean, "re_ci_low": re_.post_low, "re_ci_high": re_.post_high,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(pos[0]) if pos else \
        ROOT / "gen13_separation" / "metrics" / "BP_all" / "per_extractant.csv"
    value = pos[1] if len(pos) > 1 else "mae_all"
    pe = pd.read_csv(path)
    sizes = (pe.drop_duplicates("extractant").groupby("chemotype").size().to_numpy().astype(float))
    print(f"design file : {path}")
    print(f"units       : {pe.extractant.nunique()} extractants in {len(sizes)} chemotypes")
    print(f"Kish G*     : {wc.kish(sizes):.2f}   largest chemotype holds "
          f"{sizes.max():.0f}/{sizes.sum():.0f} = {sizes.max()/sizes.sum():.1%}")
    fit0 = wc.cluster_robust(np.arange(int(sizes.sum())) * 0.0 + 1.0,
                             np.repeat(np.arange(len(sizes)), sizes.astype(int)))
    print(f"Bell-McCaffrey dof for this cluster profile: {fit0.dof_bm:.1f} "
          f"(would be {len(sizes)-1} if balanced)\n")

    if "--no-calib" not in sys.argv:
        print("=== 1. calibration: rejection rate at nominal 5 %, H0 true, ICC 0.72 ===")
        cal = calibration(sizes, sims=int(next((a.split("=")[1] for a in sys.argv
                                                if a.startswith("--sims=")), 1000)))
        print(cal.to_string(index=False), "\n")
        cal.to_csv(path.parent / "inference_calibration.csv", index=False)

    arms = sorted(pe.arm.unique())
    best = "M_SELECTED" if "M_SELECTED" in arms else arms[0]
    comps = {f"{best}_vs_{a}": (a, best) for a in arms if a != best}
    print("=== 2. re-inference of the saved contrasts ===")
    out = reinfer(pe, comps, value=value)
    cols = ["comparison", "point", "p_pairs_percentile_OLD", "p_wcr", "p_cr2_t_bm",
            "se_pairs_bootstrap", "se_cr2", "dof_bell_mccaffrey",
            "mde80_OLD_2.80xse", "mde80_cr2_t",
            "P_better_chemo", "P_rope_chemo", "P_worse_chemo"]
    print(out[cols].round(4).to_string(index=False), "\n")
    dest = path.parent / f"{path.stem}_wildcluster.csv"
    out.to_csv(dest, index=False)
    print(f"written: {dest}\n")

    print("=== 3. detectability of a between-chemotype ligand covariate ===")
    for n_eff, tag in ((float(len(sizes)), "nominal chemotypes"),
                       (wc.kish(sizes), "Kish effective chemotypes")):
        for k in (1, 3):
            r2 = wc.detectable_r2(n_eff, n_predictors=k)
            print(f"  n = {n_eff:5.1f} ({tag:28s}) {k} predictor(s): "
                  f"smallest detectable between-chemotype R^2 = {r2:.3f}")


if __name__ == "__main__":
    main()
