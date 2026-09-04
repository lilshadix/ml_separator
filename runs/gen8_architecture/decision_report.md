# gen8 decision report

*What gen8 measured, what it changes about how the system should be used, and what
gen9 should and should not do.  Every number here has a paired interval in
`finalists/primary_bootstrap.csv` or the linked artefact.*

## 1. The three decisions gen8 actually settles

**Decision 1 — the deployment unit is not a prediction, it is a prediction plus one
requested measurement.** Zero-shot is 1.061 on the common cohort. One measurement takes
it to 0.793 blind, 0.696 chosen, 0.667 with the best calibration architecture; five take
it to 0.474. Nothing in six generations of modelling has moved the number that far, and
the gap to the 1-shot oracle (0.533) says the lever is not exhausted. The
system should ship as *"tell me your candidate conditions, I will tell you which one to
run first"*, not as a regressor.

**Decision 2 — choose the measurement centrally, then spread.** `MEDOID`/`CENTRAL` at
k = 1 (**+0.101 / +0.096** over random, CI excluding zero, 5/5 seeds, 77 and 75 of 99
ligands); a spread policy with the shrunk response-coefficient update from k = 2 onward
(`MAX_PREDICTIVE_VARIANCE` +0.055 over random at k = 2, `D_OPTIMAL` best at k = 5). Both halves are needed and
they point in opposite directions on purpose: the first measurement estimates a level
and wants a *typical* point, the second onwards estimate slopes and want *leverage*.
The farthest-point rule is significantly **worse** than random at k = 1 (−0.14 to −0.15,
0/5 seeds) and better than random at k = 2.

**Decision 3 — stop using model uncertainty to pick experiments.** No uncertainty-driven
rule clears the pre-registered bar; max-ensemble-sd against random is +0.025, CI
[−0.018, +0.052], and its point estimate flips sign between split draws. The reason is
structural, not a calibration failure. Under offset calibration the best candidate is
exactly `argmin |r_i − median(r)|`, which is verified to attain the oracle score to
6.7 × 10⁻¹⁶ over 715 (ligand, seed) blocks — while `argmin |r_i|`, which a perfectly
calibrated uncertainty would approximate, scores **0.780 against random's 0.709**, i.e.
worse than choosing blindly. The only uncertainty signal whose minimum beats random is
the RBF-GP **design** variance over the ligand's own condition axes (+0.070, 112/143,
5/5) — a geometric centrality measure wearing an uncertainty costume.

## 2. What must be said about the limits

* **Blind measurement harms a third of the population.** 57 of 143 ligands are worse
  after one *random* measurement than before it. Choosing centrally reduces this to 49.
  A deployment that measures at an arbitrary condition is not safe by default.
* **A ligand does not have one level; it has a level per series.** The transfer matrix
  is strongly positive on the diagonal and mostly negative off it. One acid-titration
  point does not calibrate the same ligand's metal selectivity in a different diluent.
  **Measure in the system you intend to predict.**
* **29 of 143 ligands cannot be fixed by any measurement.** Their shape floor is 73 % of
  their zero-shot error and the oracle one-shot arm lands exactly on it. They are *not*
  distinguished by donor chemistry (36 mechanistic descriptors, smallest corrected
  q = 1.000), by chemical novelty, or by how much data they have — they are the
  **wide-ranging ligands the model flattens hardest** (measured span 4.37 vs 1.57 decades,
  AUC 0.949; the model captures 29 % of their true range against 51 %).
* **The zero-shot level is still not predictable from structure**, and gen8's compact
  mechanistic representation — built precisely to test the "wrong neighbourhood"
  hypothesis — is significantly *worse* as a neighbourhood than plain ECFP Tanimoto.

## 3. gen9

**Do (in priority order):**

1. **Fix the flattening.** The model predicts extractant titrations at 1/20th of their
   true slope and acid titrations at 1/5th, and on the extractant axis its slope
   prediction is three times worse than guessing the corpus median. gen8's per-curve
   prior repair is worth +0.088 at k = 1 on the 26 ligands where the physics is
   identifiable (CI [+0.055, +0.144], 22/26, 5/5) and is the only change in gen8 that
   reduces *shape* error. Make it intrinsic: train the global model with a
   slope-consistency objective on the reconstructed curves rather than repairing its
   output afterwards.
2. **Close the acquisition gap.** Deployable 0.667 against an oracle 0.533 at k = 1. The
   oracle's rule is known exactly (`argmin |r_i − median(r)|`); the open problem is
   predicting that deviation from features. gen8 shows centrality captures about a third
   of it and uncertainty captures none.
3. **Buy the 29 residual-shape ligands.** They are a shape problem, so they need either
   a better response surface or — following the cross-series result — a *per-series*
   calibration rather than a per-ligand one.
4. **Verify TWE-24 against its primary document.** It is the worst ligand in the cohort
   by 0.6 log units, its six raw `D` values sit ~3 decades above five siblings measured
   on the identical grid in the same campaign, and removing it changes its nearest
   neighbour's level error from 4.42 to 0.53. The corpus has one confirmed precedent for
   exactly this failure mode (DMDPhPDA).

**Do not:**

* another learner sweep, another representation sweep, or another attempt to predict the
  zero-shot level from structure — three independent routes now agree it is not there;
* an absolute-value Neural Process or an absolute-value physics law: both were built and
  both are *significantly worse* than plain offset correction (0/5 seeds). The physics law
  works only as a correction to the frozen model, never as a replacement for it;
* a more sophisticated ligand -> parameter map. Swapping ridge for gradient boosting inside
  the winning physics-latent model is worth +0.003 (3/5 seeds), and an RBF lanthanide
  response over 14 categories +0.001 (3/5). The functional form carries the arm; the
  chemistry-to-parameter map does not;
* adding more ligand description to a calibrated model. Both architecture families
  answered the brief's question the same way — **molecular features do not add value
  after calibration**, and in the CNP they actively hurt.

## 4. Falsification scorecard (brief §32)

| gen8 would have failed if… | outcome |
|---|---|
| oracle selection were barely better than random | **survived** — 0.533 vs 0.793, gap 0.260, CI [+0.186, +0.310] |
| architecture could not beat plain offset correction | **survived, narrowly** — the shrunk response-coefficient update beats it by +0.042 at k=2 and +0.114 at k=5 |
| series-aware modelling could not reduce shape MAE | **survived, on a subset** — slope restoration cuts shape MAE by 0.076 on the 26 ligands with an extractant titration; on the acid and lanthanide axes it does nothing or harms |
| active acquisition could not beat random | **survived** — +0.101, CI [+0.037, +0.146], 5/5 seeds |
| calibration helped only rows near the measured condition | **FAILED, partially** — one measurement transfers within a series type and largely not across them |
| one-shot gains disappeared on a common ligand cohort | **survived** — 0.268 on the common cohort against 0.272 on maximal coverage |
| the result depended on leaking series identity or a hidden target | **survived** — the adapter interface passes only `truth[selected]`; asserted for every adapter in `tests/test_gen8_harness.py` |

One condition failed and it is reported as a headline, not a footnote: **calibration is
series-local.**
