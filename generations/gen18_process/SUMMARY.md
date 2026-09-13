## 0. What this generation answers, and what it does not

`docs/kshot_next_steps_20260913.md` sets the programme's goal as **a cleaner product at comparable
recovery, or the same product with fewer stages and less reagent** — explicitly *not* a lower ML
error — and asks the software (§8–§9) for one thing: the chain

> extraction-system composition → D of each metal **at the given loading** → cascade (extraction +
> scrub + strip + recycle) → purity, recovery, reagent consumption,

plus a systems database that never reconstructs a missing measurement as a fact, local recipe
optimisation that flags proposals outside the studied region, and a first applied case: **PC88A
versus Cyanex 272 for Pr/Nd with separate scrub and strip**.

**That chain now exists and runs locally.** What follows is what it is worth.

### The three things worth knowing

1. **The cascade arithmetic is trustworthy; it is the chemistry parameters that are thin.** The
   solver reproduces hand-computed Kremser and single-stage results to ≤ 2.5 × 10⁻⁹, purity,
   recovery and enrichment to the last digit, and mass balances to ~10⁻¹³. Nothing in this
   generation's uncertainty comes from the numerics.
2. **The corpus cannot predict D at new conditions better than looking up the nearest measurement**
   (R1, a pre-registered **null**: a pooled mass-action fit scores 1.090 against 1.064 for a
   nearest-condition 1-NN, macro over 14 systems, leave-one-publication-out). So the chain's D
   source is an interpolation, and its reach is the range of conditions somebody has already
   measured. This is the binding limit on every recipe the program proposes.
3. **The loading correction is real and it matters** (R2, pre-registered and **supported**): the
   ideal ligand-depletion term cuts per-series error from 0.514 to 0.305 across 10 loading series
   (95 % CI of the gain [0.030, 0.474]), and on the 7 series where loading actually moves D it
   goes 0.729 → 0.424. The middle step the task document asks for — *D at the given loading*, not
   D from a dilute sample — is the part that is earned by evidence.

Point 3 is the practically useful one: §1 of the task document warns that a D measured in a dilute
sample may not be dropped into a cascade, and this generation now has a pre-registered, corpus-
validated correction for exactly that, plus the flags (`HIGH_LOADING`, `THIRD_PHASE_RISK`,
`PHASE_BEHAVIOUR_UNKNOWN`) that stop the optimiser proposing a regime past a phase limit somebody
actually measured.

### What the program can be asked, today

Give it a feed and a product specification and it returns ranked regimes with purity, recovery,
stage counts, reagent consumption per kg of oxide, applicability flags and a parameter-status
column — for any system in the database with the metals of interest. On the 10 nitrate systems
carrying both Pr and Nd it finds, for example, a 21-stage TODGA regime at **purity 0.986 and
recovery 0.984** (`IN_DOMAIN_WITH_CAVEATS`; flagged `HIGH_LOADING` and `PHASE_BEHAVIOUR_UNKNOWN`,
so it is a candidate to test, not a recommendation).

### What it cannot be asked

- **Anything about PC88A, Cyanex 272 or D2EHPA from data.** The corpus contains none of them
  (§7 below): the case study runs on literature placeholders with declared ranges, and its output
  is an interval conditioned on those placeholders, not a measurement. Transcribing the real
  parameters from Banda 2014, Thakur 1993 and a Cyanex 272 source (open item **U4**) is the single
  change that would most improve this generation.
- **To rank ligands.** The gen15 direction model returns only two distinct magnitudes across 71
  candidates: it is a sign call, as gen16 concluded, and it is wired in as a pre-screen that
  validator rule V6 forbids from ever supplying a D.
- **Cost in currency.** `config/prices.json` holds unsourced placeholders (**U5**), so consumption
  per kg of oxide is the primary economic number and the cost proxy is NaN wherever a price is
  missing.

### The Pr/Nd case: what it settled, and what it did not

The case ran to specification otherwise (64 parameter draws × 2 systems × 250 LHS designs, 146
min). Its headline outputs need reading carefully:

- **The PC88A versus Cyanex 272 comparison is undecided, not lost.** No spec cell — not even the
  loosest, purity 0.95 at recovery 0.80 — was reached by either system in any of the 64 draws, so
  every comparison column is empty. **This is a statement about the search, not about PC88A.** The
  Fenske minimum at total reflux for that loosest cell at SF 1.4 is about **10 theoretical
  stages**, far inside the 40 + 40 the search allowed, so a feasible region must exist; a
  250-point Latin hypercube over 17 design variables did not find it. A structured grid over the
  same chemistry is reported in `results/case_prnd/targeted/`.
- **What a structured search found instead** (`results/case_prnd/targeted/`, exploratory). A
  cation-exchange extractant releases 3 H⁺ per Ln³⁺, so an unsaponified 0.1 M feed self-acidifies
  and extraction stalls — the grid shows exactly that, with maximum recovery **0.667** at zero
  saponification rising to **1.000** at 0.20–0.50, and purity collapsing again at 0.65 when the
  organic takes everything. The chain reproduces from first principles why industrial PC88A
  circuits are run saponified, and it is the variable that decides feasibility.
  **But the cell is still not met**: recovery ≈ 1 is reachable and purity ≈ 0.999 is reachable,
  never together — the best purity anywhere at recovery ≥ 0.80 is **0.814**. So the empty result
  is *not* explained by a bad `log_k` window (both ends of the trade-off work with it), nor purely
  by the random sampler. What remains untested is the circuit *shape*: the Fenske ~10 stages is a
  **total-reflux** bound, and at finite reflux this separation needs the internal reflux set by
  feed-stage, scrub-return and per-section O/A — which both searches held at defaults. That, or
  the placeholders are simply too far from real PC88A. Unresolved, and it needs U4.
- **Consistency check (b) is supported by the analytic bound, not by the stage ladder the script
  ran.** The verdict "the stage count is of the order of 70 + 70, not 7 + 7" is correct: 14 stages
  is *below* the 24-stage total-reflux minimum for (0.99, 0.99) at SF 1.4, hence impossible at any
  reflux, while the patent's 144 is ≈ 6 × N_min, a normal practical multiple (I reproduced
  N_min = 24.0 independently). The ladder table beside it is uninformative: its recovery collapses
  from 4 × 10⁻⁷ to 5 × 10⁻⁸⁵ as stages grow, the signature of a scrub washing the product back, so
  its 8 rows per rung sampled only degenerate designs. Read the Fenske line, not the ladder.
- **Check (a)** (Thakur's 97 % purity at > 85 % recovery) is therefore **not tested** by this run,
  and the report should not be read as casting doubt on it.
- **§13.5, the DGA + aqueous hold-back ligand**, re-run with the corrected D source: at the
  reference TODGA regime the β contrast buys almost nothing (purity 0.8806 → 0.8810 as
  Δlog β goes 0 → 1) because that regime already sits at high loading — and the regime is
  `INADMISSIBLE` anyway, because the sourced TODGA third-phase limit flags it. The mechanism
  itself is correct (in isolation a hold-back ligand with β(Pr) > β(Nd) lifts SF(Nd/Pr) from 2.00
  to 2.84); it is this operating point that cannot use it.

### Two findings from the integration pass

- **A defect in the deployed D source, found and fixed.** Because R1 was a null, every corpus
  system draws its D from the nearest-condition lookup — which was applying the loading correction
  to records that had themselves been measured under load. 28.7 % of corpus records with a metal
  concentration sit above loading fraction 0.1 (TBDGA: 81 %). Each record is now lifted to its own
  tracer limit first, using the same law R2 validated. The Pareto leaders barely moved, but 276 of
  298 candidates changed (median purity change 0.088). R1 and R2 are untouched: neither applies the
  term that way. Details in `addenda/INTEGRATION.md` §4.
- **A hypothesis of mine that failed.** After the null, the obvious repair is to use the
  mass-action fit only where its fitted exponent is reliable. It does not work: the correlation
  between slope reliability and M1's advantage is −0.117 (p = 0.69), and the two largest M1 wins
  are on systems whose slopes are *not* reliable. Reported rather than dropped, because it closes
  off a plausible-looking route (`results/eval/RELIABILITY_PROBE.md`).

### Where the honest uncertainty sits

Chemical coverage, again — the same constraint gen16 identified. Of 14 systems with enough
multi-publication data to fit at all, only 6 have an interpretable exponent; 4 have their exponent
pinned at the prior because the corpus never varied the ligand concentration for them. The nitrate
DGA family is well covered and the acidic organophosphorus family, which industry actually uses
for Pr/Nd, is absent. No modelling choice repairs that; measurements do.
