"""Re-infer the two gen14 headline claims with the wild cluster bootstrap-t, and expose the
*effective* number of clusters that actually carries each statistic.

Why this script exists
----------------------
Both headline claims are paired means over extractants, resampled in chemotype blocks by a
*percentile* cluster bootstrap (``gen14.dirbench.Blocked`` / ``gen13sep.inference.paired_contrasts``):
the null is never imposed and the statistic is never studentised.

    claim 1   G14_DIR_HARD beats G13_FULL_MODEL (209-col regression) under BP, +0.0522 MAE, p=0.058
    claim 2   LOGIT_TOPO39 beats G13_ET_TOPO39 under BP, +0.0522 macro accuracy,   p=0.006

``gen13_separation/metrics/BP_all/inference_calibration.csv`` already measures the size of that
percentile bootstrap for this corpus's cluster profile: 7.1 % at nominal 5 % (ICC 0.72, Gaussian),
7.7 % with t(3) tails.  That is a real but *mild* distortion -- not the 15-25 % the econometrics
literature reports for difference-in-differences with few treated clusters, because here the design
is intercept-only and every chemotype loads on the same parameter.

The thing that is NOT mild, and that no size calibration on a *dense* statistic can see, is claim 2.
Its per-extractant differences are non-zero on only 15 of 82 units (12 better, 3 worse): macro
accuracy per extractant is a coarse proportion, and most extractants are simply tied.  A restricted
wild cluster bootstrap with Rademacher weights is exactly a chemotype-level sign-flip test, so its
attainable p-value is bounded below by the number of chemotypes that hold a non-zero unit:

    p_min(two-sided, symmetric) = 2 / 2 ** G_active     (plus the +1 correction)

If the 15 non-zero units live in, say, 6 chemotypes, then no valid chemotype-level test can return
p below ~0.03, and the reported p = 0.006 is an artefact of pretending 45 clusters contribute.
That is a *falsifiable, cheap* check and it is what this script computes.

Outputs ``results/g16_dir_signflip.csv`` and prints the table.  Runtime is dominated by the bench
load; the two models are ~6 s each under one design.

References
----------
Cameron, Gelbach & Miller, Rev. Econ. Stat. 90(3):414-427, 2008, doi:10.1162/rest.90.3.414
MacKinnon & Webb, Econometrics Journal 21(2):114-135, 2018, doi:10.1111/ectj.12107
MacKinnon, Nielsen & Webb, J. Econometrics 232(2):272-299, 2023, doi:10.1016/j.jeconom.2022.04.001
Benavoli, Corani, Demsar & Zaffalon, JMLR 18(77):1-36, 2017, arXiv:1606.04316
Ash et al., J. Chem. Inf. Model. 65(18):9398-9411, 2025, doi:10.1021/acs.jcim.5c01609
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep import wildcluster as wc          # noqa: E402
from gen14 import dirbench as db                # noqa: E402
from gen14 import models as M                   # noqa: E402

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "BP"
ROPE = 0.02          # gen13sep.inference.MARGIN, the pre-registered practical margin
REPS = 9999
RESULTS = ROOT / "gen16_protocol" / "results"


def active_clusters(d: np.ndarray, g: np.ndarray, tol: float = 1e-12) -> dict:
    """How much of the statistic is carried by how few chemotypes.

    Under a chemotype sign-flip (= Rademacher WCR) only chemotypes with a non-zero residual sum
    can change the bootstrap statistic.  ``G_active`` counts them; ``p_floor`` is the smallest
    two-sided p that test can return; ``top_share`` is the share of |sum d| held by the single
    largest-contributing chemotype -- MacKinnon & Webb's failure mode is one cluster dominating.
    """
    names = np.unique(g)
    sums = np.array([d[g == c].sum() for c in names])
    active = np.abs(sums) > tol
    G_act = int(active.sum())
    tot = np.abs(sums).sum()
    order = np.argsort(-np.abs(sums))
    return {
        "G_nominal": len(names),
        "G_active": G_act,
        "n_units": int(len(d)),
        "n_units_nonzero": int((np.abs(d) > tol).sum()),
        "p_floor_signflip": float(2.0 / 2 ** G_act) if G_act < 60 else 0.0,
        "top1_share_of_abs_sum": float(np.abs(sums[order[0]]) / tot) if tot > 0 else np.nan,
        "top3_share_of_abs_sum": float(np.abs(sums[order[:3]]).sum() / tot) if tot > 0 else np.nan,
        "largest_active_chemotype": str(names[order[0]]),
    }


def reinfer(d: np.ndarray, g: np.ndarray, label: str, p_old: float) -> dict:
    """WCR bootstrap-t + CR2 t + ROPE posteriors for one paired contrast."""
    fit = wc.cluster_robust(d, g)
    w = wc.wild_cluster_t(d, g, reps=REPS, se_kind="CR2", ci=True)
    w3 = wc.wild_cluster_t(d, g, reps=REPS, se_kind="CR3", ci=False)
    wwebb = wc.wild_cluster_t(d, g, reps=REPS, se_kind="CR2", weights="webb", ci=False)
    bbc = wc.bayes_bootstrap(d, g, rope=ROPE, estimand="cluster_macro")
    bbu = wc.bayes_bootstrap(d, g, rope=ROPE, estimand="unit_macro")
    re_ = wc.random_effects(d, g, rope=ROPE)
    row = {"comparison": label, "point": fit.point, "p_percentile_OLD": p_old,
           "p_wcr_cr2": w.p_wcr, "p_wcr_cr3": w3.p_wcr, "p_wcr_webb": wwebb.p_wcr,
           "p_wcu_cr2": w.p_wcu, "p_cr2_t_bm": w.p_t_dof, "p_cr2_normal": w.p_normal,
           "se_cr1": fit.se_cr1, "se_cr2": fit.se_cr2, "se_cr3": fit.se_cr3,
           "se_iid": fit.se_iid, "dof_bell_mccaffrey": fit.dof_bm,
           "kish_chemotypes": fit.kish_clusters,
           "max_chemotype_share": fit.max_cluster_share,
           "wcr_ci_low": w.ci_low, "wcr_ci_high": w.ci_high,
           "mde80_cr2_t": wc.detectable(fit.se_cr2, fit.dof_bm),
           "rope": ROPE,
           "P_better_chemo": bbc.p_better, "P_rope_chemo": bbc.p_rope, "P_worse_chemo": bbc.p_worse,
           "P_better_unit": bbu.p_better, "P_rope_unit": bbu.p_rope, "P_worse_unit": bbu.p_worse,
           "P_better_re": re_.p_better, "P_rope_re": re_.p_rope, "P_worse_re": re_.p_worse}
    row.update(active_clusters(d, g))
    return row


# --------------------------------------------------------------------------------------
def claim2_direction(bench) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-extractant macro-accuracy difference, LOGIT_TOPO39 minus G13_ET_TOPO39, under DESIGN."""
    FS = db.feature_sets(bench)
    a = db.run(bench, "G13_ET_TOPO39", M.candidate(M.dir_extratrees()),
               features=FS["TOPO39"], design=DESIGN)
    b = db.run(bench, "LOGIT_TOPO39", M.candidate(M.dir_logistic()),
               features=FS["TOPO39"], design=DESIGN)
    ua = db.unit_hits(a).set_index(["extractant", "chemotype"])
    ub = db.unit_hits(b).set_index(["extractant", "chemotype"])
    j = (ub["hit"] - ua["hit"]).dropna().reset_index()
    gain = db.paired_gain(a, b)
    return j["hit"].to_numpy(float), j["chemotype"].astype(str).to_numpy(), gain["p_two_sided"]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    bench = db.load()
    rows = []

    d, g, p_old = claim2_direction(bench)
    rows.append(reinfer(d, g, f"{DESIGN}: LOGIT_TOPO39 - G13_ET_TOPO39 (macro accuracy)", p_old))

    # claim 1 needs the per-extractant MAE table that g14_value.py currently discards; if a
    # previous run saved it, re-infer that too.
    pe_path = ROOT / "gen14_direction" / "results" / f"g14_value_per_extractant_{DESIGN}.csv"
    if pe_path.exists():
        pe = pd.read_csv(pe_path)
        tab = pe.groupby(["arm", "extractant"])["mae_all"].mean().unstack("arm")
        chem = (pe.drop_duplicates("extractant").set_index("extractant")["chemotype"]
                .astype(str).reindex(tab.index))
        for ref, cand, lab in (("G13_FULL_MODEL", "G14_DIR_HARD", "G14hard_vs_FULL"),
                               ("MEAN_CURVE", "G14_DIR_HARD", "G14hard_vs_MEANCURVE"),
                               ("G13_DIR_HARD", "G14_DIR_HARD", "G14hard_vs_G13dir")):
            if ref in tab.columns and cand in tab.columns:
                delta = (tab[ref] - tab[cand]).dropna()
                rows.append(reinfer(delta.to_numpy(float),
                                    chem.reindex(delta.index).to_numpy(),
                                    f"{DESIGN}: {lab} (mae_all)", np.nan))
    else:
        print(f"[note] {pe_path.name} not found -- add "
              f"`pe.to_csv(db.RESULTS / f'g14_value_per_extractant_{{design}}.csv', index=False)` "
              f"after line 101 of g14_value.py and re-run it to cover claim 1.\n")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"g16_dir_signflip_{DESIGN}.csv", index=False)

    show = ["comparison", "point", "n_units", "n_units_nonzero", "G_nominal", "G_active",
            "p_floor_signflip", "top1_share_of_abs_sum", "p_percentile_OLD", "p_wcr_cr2",
            "p_wcr_cr3", "p_cr2_t_bm", "dof_bell_mccaffrey", "mde80_cr2_t",
            "P_better_chemo", "P_rope_chemo", "P_worse_chemo"]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(out[show].round(4).to_string(index=False))
    print(f"\nwritten: {RESULTS / f'g16_dir_signflip_{DESIGN}.csv'}")


if __name__ == "__main__":
    main()
