"""What the corpus could have detected: a measured bound to replace "we tried seven priors".

The magnitude result is a negative one -- seven priors for |a| all land within 0.007 of a plain
constant.  A negative result is only informative next to a power statement, and the programme has
none.  This script produces the two numbers that turn it into a bound:

1. **Minimum detectable MAE difference.**  ``(t_{.975,dof} + t_{.80,dof}) * se_CR2`` on the
   Bell-McCaffrey degrees of freedom, not ``2.80 * se`` on a pairs-bootstrap se.  Reported next to
   each contrast's own point estimate: a "significant" result whose effect is *below* its own MDE
   is in the regime where the estimate that clears the threshold is necessarily an exaggeration
   of the truth (Gelman & Carlin's Type M error).

2. **Smallest between-chemotype R^2 a ligand covariate could have and still be found.**

   The effective sample size for that question is *not* Kish's effective cluster count.  Kish's
   ``(sum n)^2 / sum n^2 = 11.7`` describes how unequally the 90 extractants sit in 45 chemotypes,
   which is what inflates the variance of a *unit-weighted mean*.  The magnitude question is
   different: it asks whether a covariate predicts a chemotype's *mean* amplitude, so each
   chemotype is one observation and the only penalty is that a chemotype holding one extractant
   estimates its own mean noisily.  With chemotype-level ICC ``rho`` the reliability of chemotype
   g's observed mean is

       lambda_g = tau^2 / (tau^2 + sigma_w^2 / n_g) = n_g / (n_g + (1 - rho)/rho)

   and the effective number of chemotypes is ``sum_g lambda_g`` -- the classical
   reliability/attenuation correction (Spearman 1904; Hunter & Schmidt, *Methods of Meta-Analysis*,
   3rd ed., Sage 2015, ch. 3).  For this corpus that is **34.3**, not 11.7, because 34 of the 45
   chemotypes hold exactly one extractant and a single extractant still estimates its own
   chemotype mean with reliability 0.72.

   Quoting 11.7 here would understate the corpus by a factor of three and make the negative result
   look far weaker than it is.  Quoting 45 would overstate it.  34.3 is the defensible number and
   the bracket [45, 11.7] is what it replaces.

Writes ``results/g16_detectability.csv``.  numpy/scipy only, < 30 s.

References
----------
Gelman & Carlin, "Beyond Power Calculations: Assessing Type S (Sign) and Type M (Magnitude)
    Errors", Perspectives on Psychological Science 9(6):641-651, 2014,
    doi:10.1177/1745691614551642
Bell & McCaffrey, "Bias reduction in standard errors for linear regression with multi-stage
    samples", Survey Methodology 28(2):169-181, 2002
Landrum & Riniker, "Combining IC50 or Ki Values from Different Sources Is a Source of Significant
    Noise", J. Chem. Inf. Model. 64(5):1560-1567, 2024, doi:10.1021/acs.jcim.4c00049
Ash et al., "Practically Significant Method Comparison Protocols for Machine Learning in Small
    Molecule Drug Discovery", J. Chem. Inf. Model. 65(18):9398-9411, 2025,
    doi:10.1021/acs.jcim.5c01609
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen13sep import wildcluster as wc  # noqa: E402

RESULTS = ROOT / "gen16_protocol" / "results"
ICC_AMPLITUDE = 0.72          # measured chemotype-level ICC of the cell amplitude
REPLICATE_SD = 0.237          # corpus replicate sd of the amplitude -- the irreducible floor


def reliability_neff(n_g: np.ndarray, icc: float) -> tuple[float, np.ndarray]:
    """Effective number of chemotypes for a *between-chemotype* regression on chemotype means."""
    r = (1.0 - icc) / icc                     # sigma_w^2 / tau^2
    lam = n_g / (n_g + r)
    return float(lam.sum()), lam


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pe = pd.read_csv(ROOT / "gen13_separation" / "metrics" / "BP_all" / "per_extractant.csv")
    n_g = pe.drop_duplicates("extractant").groupby("chemotype").size().to_numpy().astype(float)
    G, N = len(n_g), n_g.sum()

    print(f"chemotypes G = {G}   extractants N = {N:.0f}   largest chemotype = {n_g.max():.0f}"
          f" ({n_g.max()/N:.1%})")
    print(f"size profile (count of chemotypes by size): "
          f"{dict(zip(*np.unique(n_g.astype(int), return_counts=True)))}")
    print(f"Kish effective clusters on units      = {wc.kish(n_g):.2f}   "
          f"[right for a unit-weighted mean, WRONG for the covariate question]")

    rows = []
    for icc in (0.72, 0.60, 0.50):
        neff, lam = reliability_neff(n_g, icc)
        for k in (1, 3, 10):
            r2 = wc.detectable_r2(neff, n_predictors=k)
            rows.append({"quantity": "between_chemotype_R2", "icc": icc,
                         "n_effective": round(neff, 2), "n_effective_kind": "reliability-weighted",
                         "n_predictors": k, "power": 0.80, "alpha": 0.05,
                         "min_detectable": r2})
        print(f"ICC {icc}: reliability-weighted effective chemotypes = {neff:.1f} "
              f"(lambda in [{lam.min():.3f}, {lam.max():.3f}])")

    # the two brackets the reliability number replaces, kept so the choice is auditable
    for neff, kind in ((float(G), "nominal (no measurement error)"),
                       (wc.kish(n_g), "Kish on units (wrong estimand)")):
        for k in (1, 3):
            rows.append({"quantity": "between_chemotype_R2", "icc": np.nan,
                         "n_effective": round(neff, 2), "n_effective_kind": kind,
                         "n_predictors": k, "power": 0.80, "alpha": 0.05,
                         "min_detectable": wc.detectable_r2(neff, n_predictors=k)})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "g16_detectability.csv", index=False)
    print("\n=== smallest detectable between-chemotype R^2 at 80 % power, alpha 0.05 ===")
    print(out.round(4).to_string(index=False))

    neff72, _ = reliability_neff(n_g, ICC_AMPLITUDE)
    r2_1 = wc.detectable_r2(neff72, n_predictors=1)
    r2_3 = wc.detectable_r2(neff72, n_predictors=3)
    print(f"""
--- the sentence this licenses -------------------------------------------------------------
With {G} chemotypes, chemotype-level amplitude ICC {ICC_AMPLITUDE} and hence
{neff72:.1f} effective chemotypes, this corpus has 80 % power to detect a single ligand
covariate only if it explains at least {r2_1:.1%} of BETWEEN-chemotype amplitude variance
({r2_3:.1%} for a three-covariate block).  Seven priors were tried and none beat a constant,
so: no ligand covariate available here explains more than ~{r2_1:.0%} of between-chemotype
amplitude variance.  That is a measured bound, not an absence of effort.
Anything smaller is invisible to a corpus of this size and would stay invisible until the
number of chemically independent chemotypes roughly doubles.
--------------------------------------------------------------------------------------------
DO NOT derive the ROPE from the replicate sd.  The amplitude replicate sd is {REPLICATE_SD};
propagated through the curve basis (mean |r_i - r_j| = 1.227 over the 91 metal pairs) that is
{REPLICATE_SD * 1.2265:.3f} in MAE-of-log-SF units -- larger than the whole 0.122 gap between the
best model and the corpus mean curve, and larger than the 0.178 gap to the oracle.  A ROPE
that wide declares every result in the programme practically equivalent, including the ones
that are real.  The reason it is wrong: label noise is COMMON MODE in a paired comparison --
both arms are scored against the same noisy cells, so it cancels in the difference, which is
why the paired CR2 se is 0.017-0.037 and not 0.29.
The defensible practical floor for a PAIRED contrast is that contrast's own minimum
detectable effect (``mde80_cr2_t`` in g16_dir_signflip_*.csv), which is 0.05-0.11 here.
Landrum & Riniker (JCIM 2024, 0.27-0.50 log units of inter-source disagreement) set the
precedent for reading a floor off inter-source spread, but that floor bounds ABSOLUTE
accuracy, not the resolvable difference between two arms scored on the same cells.
written: {RESULTS / 'g16_detectability.csv'}""")


if __name__ == "__main__":
    main()
