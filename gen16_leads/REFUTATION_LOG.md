# Gen16 — refutation log

*Every refuter's attempt and its outcome, including the ones that failed to break a claim, and
every temptation a lead recorded and did not act on.  Six refuters ran in Phase 4, two per claim,
each spawned without the other's output and each given one instruction: find the control, confound
or bookkeeping error that would make this claim wrong.  Their code is under
`results/refutation/<claim>/<lens>/` and `scripts/refute_*.py`.*

---

## 1. Summary

| claim | lens A (data, leakage, bookkeeping) | lens B (statistics, confounds) | outcome |
|---|---|---|---|
| **L3A** measurements saved | dented, **material** | dented, **material** | survives as a *pooled* statistic, refuted as a *laboratory-level* saving |
| **L4BP** acquisition order | dented, **fatal** | dented, **material** | **killed** — a featureless order reproduces it |
| **L1NULL** the xTB null | dented, **material** | dented, **material** | verdict **corrected**: "closed" → **undecided** (no power either way) |

Both refuters reproduced every headline digit for digit before attacking it, one of them by
rebuilding the pipeline from the sealed pre-registration text rather than from the lead's code.
**No claim survived both refuters untouched.**  That is the finding of this phase.

---

## 2. L3A — the direction call saves measurements

### What held

| check | result |
|---|---|
| independent re-derivation from the pre-registration text (own task builder, own arithmetic) | identical at 6 dp in all five designs; seal verified independently |
| **diglycolamides removed** (sc009, 23 of 90 extractants) | +1.459 under BP, down 27 % from +1.988, still ≥ +1.46 and positive in **all five** designs.  sc009-only candidate sets save just +0.386, so the effect is discrimination *between* chemotypes, not a diglycolamide artefact |
| **n_metals confound** | Spearman(call, n_metals) = −0.015; n_metals-stratified permutation null gives real − shuffled = +1.886; n_metals explains ~5 % |
| n_metals as a width-matched competitor | scores **negative** saving in every design |
| publication mask | BP is quoted and is the **smallest** of the five savings — the ordering expected if there were no publication leak |
| permutation null re-run at three fresh seeds | p = 0.0000 everywhere; null mean +0.001 |
| leave-one-publication-out (all 58) | BP range [+1.506, +2.428], **zero sign flips** |
| unit matching | candidate-set fingerprint and observed-`log SF` sum byte-identical across designs and arms |
| `HEAVIER_ALWAYS` = 0 | confirmed **algebraically**: a constant call cannot partition a candidate set |
| a real competitor (13-column gen6 donor census, used the same way) | saves **−3.14** measurements — it costs measurements; G14's increment over it is +5.13 |

### What dented it

1. **The scope test (both lenses, independently).**  Restricting each task's candidate set to
   extractants from **one publication** — the lead's own limitation 1, which it correctly declined
   to run unregistered — collapses the saving to **+0.0046 measurements (0.07 %)** under BP, CI
   [−0.016, +0.022], with the interval containing zero in **all five** designs.  On those tasks
   69.6 % give every candidate the same call (0.0 % pooled), and against a perfect-sign-call
   ceiling of +0.414 the model captures 1.1 % of the headroom, against 47.3 % pooled.
   *Caveat that keeps it from being fatal:* the within-publication test rests on 4 of 58
   publications, 31 extractants, 11 chemotypes, 18 of the 31 being sc009.
2. **Within one chemotype** (lens B): +0.334, registered rule fails in all five designs.
3. **Fold purity** (lens B): the task pools the held-out predictions of all five folds of a seed,
   so candidate *X* was predicted by a model whose training set contained candidate *Y*.  Rebuilt
   one fold at a time the saving under BP is **+0.906 of an E_random of 5.52 (16.4 %)** against
   +1.988 of 6.906 (28.8 %) — **46 % of the quoted magnitude**.  The effect survives: as a
   fraction of the perfect-sign ceiling the fold-pure run captures **49.8 %**, slightly more than
   the pooled run's 47.3 %.  This dents the *quoted magnitude*, not the existence of the effect,
   and both figures now travel with the claim.
4. **One overstatement in the lead's prose** about the decision rule, corrected in §4 of the
   lead's own report.

**Verdict.**  The arithmetic is right and it replicated on the withheld seeds.  The *claim as
worded* — a saving available to a chemist choosing among candidates — is refuted.  What survives
is a between-laboratory statement.  Carried everywhere the number appears.

---

## 3. L4BP — the acquisition order

### What held

| check | result |
|---|---|
| headline recomputation from the raw per-extractant parquet | +0.0500653184 against the lead's +0.0500653184 |
| diglycolamides removed | the claim gets *stronger* (+0.038, passes P1 in all five) — it is not a diglycolamide effect |
| n_metals | partial Spearman given \|a\| = −0.190; does not explain it |
| RANDOM re-drawn from a disjoint RNG stream | +0.063 (BP) — not a lucky draw set |
| scored units | byte-identical across every order and budget |
| decision rule, BH, LOCO, per-seed signs | all recomputed and confirmed |

### What killed it

**A chemotype order built from zero chemistry reproduces the whole effect.**  Sorting chemotypes
by their count of well-determined training cells — no features, no descriptors, no model —
gives `SIZE_vs_RANDOM` = +0.045 under BP and is positive in **all five designs** (+0.028 B,
+0.019 BR, +0.026 BQ, +0.023 A), a sign consistency the registered A-optimal order never
achieved.  What the 39 donor-topology columns add on top is

  `AOPT_vs_SIZE` = **+0.005**, p = 0.36, below the 0.02 margin, and **negative under B and BQ**,

carrying the same −/+/−/+/+ sign pattern as the claim itself.  The mechanism is direct:
Spearman(A-optimal pick position, chemotype cell count) = **−0.63** under BP against ≈ 0 for
random, and at a budget of 6 chemotypes the A-optimal order already holds **72 %** of the fold's
training cells against random's 18 %.  The reference was never matched on training-set size, and
matching it removes the claim.

**Verdict.**  Killed.  The transferable content is corpus **volume**, not chemistry choice.  The
claim's causal wording is falsified.

**Note, and it is deliberately not promoted.**  `SIZE` is itself a real, design-consistent,
featureless result, and it was found by a refuter.  A refuter check can kill a claim, never
resurrect one, so it is **not** a gen16 claim.  It is recommended for pre-registration in gen17
(`DECISION_REPORT.md` §12), where it would need its own matched null and its own five designs.

---

## 4. L1NULL — the cycle-corrected xTB null

Here the refuters' job was the mirror image: find the way a real signal could have been destroyed.
Both found it, by different routes, and they agree.

### What held

Reproduction (an independent rebuild including a different xyz parser and formula decomposition);
the null is not a diglycolamide artefact; partial correlations given `n_metals` move nothing
(−0.084 → −0.081); it is not an artefact of the S8 subset, of Spearman rather than Pearson or
Kendall, or of the 39-vs-62 set change; the family-wise permutation bar reproduces at three seeds.

### What dented it

1. **The corrected slope has no reliability.**  Split-half over metals within a series (odd vs
   even) gives Spearman **−0.272** for the SPECIES slope; the jackknife puts the between-extractant
   slope sd at 0.203 eV against a within-series standard error of 0.329 eV, i.e. **reliability
   0.000**.  The maximum attainable correlation against *any* target is then ≈ 0, so the
   registered \|ρ\| ≥ 0.40 bar was **unreachable in principle**.  The five models form a
   reliability ladder collinear with the lead's "dose–response in bookkeeping exactness"
   (split-half: NAIVE +0.930, ELEM +0.809, SPECIES_NFILLCOL +0.489, SPECIES_CONST +0.460,
   SPECIES −0.272; Spearman(\|ρ_obs\|, reliability) = +0.90).
2. **Signal injection: the correction destroys a real trend.**  Adding a genuine
   composition-independent trend to the raw energies and re-running, the retained fraction is
   NAIVE 0.998, SPECIES_CONST 0.897, ELEM 0.835, **SPECIES 0.416** — the registered correction
   removes ~58 % of a real signal (independently, the second refuter measures 64 %, matching the
   lead's own 73 % attenuation estimate).  SPECIES needs an injected trend of ≈ 0.45–1.1 eV per
   standardised-radius unit before it clears its own 0.40 bar, while the entire observed SPECIES
   slope spread is 0.25 eV.
3. **A correction that cannot absorb the trend regains signal.**  Fixed additive reference
   energies with no free per-series parameter give ρ = **+0.29** (lens A, `CYCLE_ADD`) and
   **+0.34 / +0.45** (lens B, `FIXCYC_CONSTNLIGS`, S8 / S14) — **above the registered 0.25
   "closed" threshold**.
4. **n = 19 cannot separate +0.167 from +0.5.**  The Fisher interval at the registered
   `SPECIES_CONST` row is [−0.337, +0.597], width 0.93; power to reject ρ = 0 at truth +0.4 is
   0.36.  The secondary registered row could never have decided anything.
5. **A deviation from the sealed text** (lens B): the sealed §3 wrote SPECIES as
   `+ n_fill × γ_species`; the lead implemented per-species molecule counts (n_NO₃, n_H₂O)
   because `n_fill` counts donor *sites*.  The lead disclosed this, ran the literal variant as
   well, and both give the same verdict — but it is a deviation and it is recorded here.

**Verdict corrected to UNDECIDED — and the correction cuts both ways.**  The registered verdict
"**CLOSED**" is estimator-specific and overstated: the registered estimator is too noisy to detect
anything, so the experiment had no power, and the honest sentence is not "the cycle-corrected slope
does not correlate with the amplitude" but **"the cycle-corrected slope, as constructed, does not
correlate with itself"**.

But the positive side fails just as clearly, and the audit of this log caught the first draft
treating it as encouraging.  The attenuation-free estimators are unregistered post-hoc arms and
they do not survive the diglycolamide control the brief makes mandatory: `CYCLE_ADD` falls from
**+0.291 to −0.026** (n = 43, p = 0.87) when sc009 is removed, and it scores **+0.049** on the 19
constant-composition series against **+0.476** on the 43 whose composition varies — the positive
lives on the composition step itself, which is the artefact the lead exists to remove.  Nor does
anything in L1's registered family clear its own registered permutation bar (largest 0.602 against
a 95th-percentile bar of 0.609, family-wise p = 0.057), gen15's +0.644 included; and the free 2D
competitor `frac_donor_pairs_within_3` (−0.42 to −0.53) is larger than every xTB estimator on the
table.

So Stage 2 remains the only construction that is exact *and* attenuation-free, and it is worth its
6.7 CPU-hours because it is cheap and decisive — **not** because gen16 found anything promising.
`DECISION_REPORT.md` §4 and §12 state the expected-null prior that goes with it.

### A refuter finding that is not a claim, and is striking

Attacking the null from the chemistry side, both refuters asked whether *where along the series the
builder's inner sphere reorganises* is itself predictive.  It is, and more strongly than any
descriptor in the programme's history: pure composition-position descriptors containing **no energy
at all** reach Spearman **+0.83** (`steppos_n_H2O`, S14, n = 27), **+0.78** (S8, n = 29),
**+0.76** (`slope_n_H2O_vs_r`, n = 39) and **−0.50** (`nligs_range`, S8) against the signed
amplitude, where the best 2D descriptor in four generations is −0.53.
**Treat this with suspicion, not excitement.**  These are properties of a *geometry builder's
recipe*, not measurements; they were found post hoc by refuters over many tried columns; the n is
small; and the family-wise bar for the max \|ρ\| over the registered 12 columns was already ≈ 0.60.
Whether it is real physics (the radius at which a ligand's coordination sphere reorganises is a
size-match observable) or an artefact of the builder's heuristics is exactly the question gen17
should pre-register.  It is written up in `DECISION_REPORT.md` §12 as lead 1 for the next
generation, and it is **not** a gen16 result.

---

## 5. Temptations recorded by the leads and not acted on

Reproduced from the lead reports, because a temptation resisted is evidence about the process.

**L2.**  The BCa lower bound under BP is +5.7 × 10⁻⁶; reading "the CI" as the BCa interval alone
would have opened a closed gate on a six-millionths margin.  The percentile interval was fixed
before the run and p = 0.061 fails regardless.  Also: design A is the only design whose interval
excludes zero, and A never selects.

**L1.**  Relaxing the frozen construction to obtain a friendlier reproduction set; both readings of
`n_fill` were run instead and both reported.

**L3.**  Reading "the CI" as the more favourable of two bootstraps (every row instead carries both
and names which one the headline used); reporting the permutation p as (1 + k)/(R + 1) when it came
out 0; dropping the ≥ 5-candidate rule or lowering the 0.3 success threshold; reporting only the
requested direction where the model wins; promoting an exploratory row; presenting
`HEAVIER_ALWAYS` = 0 as a win without explaining that it is zero by construction; **running the
within-publication version once the pooled number came out large** (left to the refuters, who ran
it and dented the claim with it); adding a second, narrower interval after seeing the first.

**L4.**  Stopping at BP, the best-looking design; dropping `UNCERT` after a discouraging dry run;
promoting the cell-matched maxmin result, which is positive in all five designs; **promoting the
no-diglycolamide result, which passes P1 in all five designs, on the ground that a refuter check
can kill a claim but never resurrect one**; filtering an unattractive prospective list; choosing
the averaging set after looking; re-tuning a fallback guard that fires 2 100 times for one order
and never for another.

**L5.**  Repairing Ledoit–Wolf after it returned MAE 34 (a repaired arm is a sixth arm chosen after
seeing a number); arguing the 0.02 margin down for the measured mode, where the whole error is
0.167 so the margin is 12 % of it — *the argument is correct and is still a post-hoc change to a
registered rule*; quoting `LOWRANK2` at its best rung only; dropping the coverage-band half of the
calibration rule, under which an arm would have qualified; fixing `NOISE_VAR`, which would move
coverage from 98.8 % to nominal in one line and change every measured-mode MAE.

## 6. Process failures by the orchestrator, recorded

1. A commit chained with `;` rather than `&&` after a failing audit, which swept partially written
   files from four running agents into commit `b7eae82`.  Disclosed in the following commit
   message; the finished files landed in the Phase 2 commit.
2. The first seed audit's call filter included the `seeds=` token itself, making the filter
   vacuous: it produced a false positive on a dataclass field and would have missed a real
   violation split across lines.  Rewritten as an AST walk and verified against a planted
   violation (`40a40bc`).
3. `scripts/g16_confirm.py` initially read a column `loco_stable` that does not exist
   (`loco_sign_stable` does).  Found by inspecting L2's output before the confirmation run, not by
   the run failing.
