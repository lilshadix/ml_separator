# Round-5 ideas, second round: ranked addendum to IDEAS_RANKED.md

Written 2026-10-23. This file ranks the 15 ideas of the second idea round (themes SHIFT, LAB, RECIPE and CONFIRM; the
theme notes are `shift.md`, `lab.md`, `recipe.md` and `confirm.md` in this folder) after two skeptic verdicts per idea:
- **Skeptic A**: independent recompute, leakage, near-copies, C5-BTBP and publication dependence;
- **Skeptic B**: chemistry, feasibility, power, overlap with round-1 ideas, practical value.

It is an addendum to `../ideas/IDEAS_RANKED.md` (round-1 ideas M01–M39) and uses its ids, tiers and closure ledger.

> **Status: EXPLORATORY throughout.** Every in-corpus number below comes from the 229 C3 cells, the 521-cell V1 bench
> and the 5 dev seeds. Rounds 1–4, the round-5 probes P1–P5 and both skeptic rounds have already seen all of them.
> Every threshold, stratum, K and window used here was chosen after looking. Nothing is sealed or confirmatory, and no
> number may carry decision language. E-CT, E-LAB, E-REC, E-DEC and the chord statistics are separate estimands and are
> never quoted next to the E-A reference (B1 = 0.452).

---

## Bottom line

1. **Nothing in round 2 opens a route to 0.50 on E-A.** No idea proposed a model change for E-A that survived; the one
   E-A idea (lab-3) is a closure line.
2. **No idea earns its own round-5 slot.** Seven ideas survived at least one skeptic; none survived both. Each survivor
   becomes sealed text, an annex row or a ledger line inside a round-1 host (M01/M05, M33, M34, M35). Eight ideas were
   rejected by both skeptics; most leave one descriptive row or one closure line behind.
3. **The most valuable output is a repair of the external condition-transfer test (confirm-4).** As written, M05/M01
   kill (iv) would use Holm inside the claim at G = 6, where an exact sign-flip cannot reject (smallest p 1/64 > 0.0125).
   Its per-target log ratio is undefined on corpus data (−inf). Its family-level T0 rule lets pH near-copy siblings into
   the primary class. Its power is quoted for a DGA world the new supply does not contain. The fixes cost hours and must
   be sealed before any new value is read.
4. **The supply does not match the evidence.** The corpus evidence behind every planning number is DGA (E-CT: 94 of 99
   PRIMARY cells) or aromatic N-donor (the heavy-selective Gd stratum: 10 of 12 units). The new literature supply (P4)
   is mostly acidic organophosphorus, quaternary-onium ionic liquids and chelators. Every power figure in round 2 is
   therefore conditional on chemistry that will not arrive, and has to be restated in world-specific form (DGA world,
   halved effect, cation-exchanger world).
5. **Two exploratory leads deserve sealed homes, not slots.**
   - **C-GD** (confirm-2): the Gd break of cation-exchange extractants is probably deeper than B1's amplitude. The
     chemistry is textbook for HDEHP-type systems; what is new is that B1's single, DGA-dominated amplitude would
     under-correct it. This note's reconciliation check (§ "Reconciliation check") shows that the skeptics' disagreement
     on deployed rows comes mostly from one component-identity error (pub_84113c17ad, P5 finding F6). Without it and
     without C5-BTBP, a stratum amplitude reduces SSE on eligible heavy-selective rows by about 4 %, ordered by chord
     depth (7 of 12 publications better). That is about +0.002 R² pooled, far below any MDE.
   - **E-LAB** (lab-1): a new extractant in a campaign that also reports other extractants' full series. A level-matched
     shape from those series gives +9 % to +11 % relative SSE against B1 on the same reads, and a count- and
     chemotype-matched placebo from other campaigns does not (−1.4 % to −2.7 %). But most of the edge is level-driven
     weighting rather than the specific donors, and two publications carry it.
6. **Data quality.** The skeptics found more than ten duplicate or near-copy links that the round-3 V2 lists miss, and two
   publications (pub_19783c039b and pub_b0b3296ec5) carry several of the condition-transfer leads (shift-1, shift-2,
   recipe-1, recipe-4, and 41 % of P1's E-CT gain). Both go to M25 as provenance checks (§ "Duplicate and provenance
   candidates").
7. **Eight closure lines** are added to the round-5 ledger (§ "Closure ledger additions").

---

## How to read this addendum

**Ranking rule.** Survivors are ranked by (decision value of the corrected version for round 5) × (probability that it
holds as corrected) ÷ (cost). No judge panel was run for round 2, so there are no J1–J3 scores. Because no idea earns a
slot, IDEAS_RANKED's tiers are replaced by an **action** column:

| action | meaning |
|---|---|
| SEAL | sealed text inside the host, written before any new value is read; the decisive leg is data-gated through the host |
| ANNEX | a descriptive row run with the host's software; never a decision slot |
| LEDGER | one closure line, adopted now; already computed by both skeptics |
| RIDER | a quantity recorded on every new source that arrives for other reasons |

**Survival.** "Survives" means at least one skeptic let the idea stand, with fixes. Ideas both skeptics rejected are in
the Rejected section with their fatal flaw and the fragment worth keeping.

**Prior.** Each skeptic's probability that the hypothesis holds and passes its own kill criterion. Where the skeptics
priced different things (substance against "passes its kill with the expected supply"), the consolidated line says so.

**Reproduction.** Every one of the 30 verdicts reproduced its idea's quick check (`reproduced_quick_check = true`), most
with independent code. Corrections to quoted numbers are given idea by idea.

---

## Summary table

| rank | id | action → host | short title | survives A / B | prior A / B | consolidated |
|---|---|---|---|---|---|---|
| 1 | confirm-4 | SEAL → M05 / M01 kill (iv) | C-ECT: condition-transfer confirmation on new sibling families | yes / no | 0.50 / 0.15 | ≈ 0.75 that own-series beats B1@4 in new non-T0 families; ≈ 0.15–0.25 for a decisive pass within round 5's supply |
| 2 | confirm-1 | SEAL → M34, M35, M33 | CP-13: per-claim minimum G, exclusion list, supply-audit kill | yes / no | 0.35 / 0.20 | ≈ 0.3 that at least one claim is decided; ≈ 0.9 that the sealing protects round 5 |
| 3 | confirm-2 | SEAL → M33 (stratum of its Gd-dip test) | C-GD: cation-exchanger Gd break deeper than B1's amplitude | yes / no | 0.55 / 0.25 | ≈ 0.85 true if acidic-OP labs arrive; ≈ 0.3 decisive pass |
| 4 | confirm-3 | SEAL, merged into confirm-2 | C-B1: one chord interval read against 0, β/2, β | yes / no | 0.40 / 0.25 | ≈ 0.25; its kill rule as written is wrong |
| 5 | lab-1 | ANNEX → M01 (new POPX0 stratum) | E-LAB: calibrate a new ligand from the same campaign's other series | yes / no | 0.30 / 0.15 | ≈ 0.5 the in-campaign amide channel is real; ≈ 0.15 a decisive external pass |
| 6 | lab-3 | LEDGER → M35 closure ledger | E-A: per-metal priors from other publications do not beat B1 | yes / no | 0.80 / 0.85 | ≈ 0.85 for the scoped line; ≈ 0.3 as written |
| 7 | shift-3 | RIDER → M05 / M22 / M33 ledger | Ce/Pr acid sign in DGA-class extractants (sign only) | yes / no | 0.35 / 0.15 | ≈ 0.6 the DGA sign holds in new data; level-beyond-fanning undecided |
| R | shift-1 | → M01 descriptive row; ledger | acid-distance rule for zero-read copying | no / no | 0.15 / 0.20 | |
| R | recipe-1 | → M01 read-ladder table | reads-equivalent credit of an own prior series | no / no | 0.15 / 0.12 | |
| R | recipe-3 | → M14 descriptive row | sign-first triage from an own prior series | no / no | 0.25 / 0.10 | |
| R | recipe-2 | → M18 descriptive row | fixed training-free metal ladder | no / no | 0.20 / 0.15 | |
| R | recipe-4 | → M03 arm (inside M01); ledger | ICP-MS condition skipping with a shrunk acid trend | no / no | 0.20 / 0.10 | |
| R | lab-4 | → one line in M34 | campaign, not laboratory, as the confirmation unit | no / no | 0.15 / 0.08 | |
| R | lab-2 | → M11 / M12 arm | in-run TODGA reference protocol | no / no | 0.15 / 0.08 | |
| R | shift-2 | → M03 arm; ledger | bracketed per-metal slope model | no / no | 0.10 / 0.08 | |

Counts: 7 survive one skeptic (all of them survive A only), 8 are rejected by both, 0 survive both.

---

## Relation to the round-1 hosts (M01, M09, M14, M18, M25, M33, M34, M35)

"—" means no material relation. The prose under each idea gives the detail.

| id | M01 (E-CT) | M09 (window) | M14 (E-DEC) | M18 (metal ladder) | M25 (source recovery) | M33 (supply gate) | M34 (inference) | M35 (decision) |
|---|---|---|---|---|---|---|---|---|
| confirm-4 | **is** M01 kill (iv) / M05, repaired | T0 data flag uses in-window SF | — | — | pub_b0b3296ec5 and pub_8415538291 provenance; bit-identical pairs into CLEAN | audit of non-T0 families before extraction | lab unit; fixed sequence; G floor 7 | inconclusive zone; never beside 0.452 |
| confirm-1 | C-ECT = M01 kill (iv) | window row flag excluded | not in package | not in package | β_frozen is learned mainly from unlocated cells (P5) | supply-audit kill; Eu-Gd-Tb eligibility; mechanism tag | min-G table; two-stage nulls; lab unit | exclusion list; per-claim α; no global claim |
| confirm-2 | — | DL-flagged values excluded (cf. M24) | — | read Gd for cation exchangers | β_frozen anchor; C5-BTBP shift copies; pub_84113c17ad | **host**: stratum of M33's Gd-dip test | lab unit; analytic power; two-stage nulls | three-way decision |
| confirm-3 | — | V2-5 floor censoring of the chord | B0 vs B1 is M14's comparator too (no calls here) | — | B1's Gd anchor depends on M25 | the chord is M33's test vehicle | A3 = M34's positive control, run prospectively | status quo kept on non-significance |
| lab-1 | **annex** arm for POPX0; pairs with OFF_XEXT | sharing peaks at log D < −2 | — | "worth one read" unresolved | V2-missed duplicates | count of multi-ligand campaigns | lab unit; kill-rule operating characteristics | separate estimand E-LAB |
| lab-3 | untouched: same-campaign information is what transfers | — | — | — | the Gd term's anchor stays M25's job | — | reopen bar set by 13-cluster detectability | **closure ledger** line |
| shift-3 | TILT_CE at most a k = 0 diagnostic row | [−2, 2.5] eligibility; floor controls | — | — | flags in pub_0fd6ab4e40, pub_19783c039b | ledger rider; acidic-OP sign-reversal control | lab unit; exact sign test | no slot; consistent with the M28 closure |
| shift-1 | descriptive E-CT row (distance × range position) | below-range targets sit near the floor | — | — | duplicate pairs in pub_19783c039b, pub_c8195b4ca7 | — | lab units; ≥ 4–5 levels per family | closure line |
| recipe-1 | E-REC table in M01's read ladder | — | — | comparator = best no-prior policy (M18 ladder) | pub_b0b3296ec5 gate; bit-identity check | ~4 families in supply | family-level kill miscalibrated | — |
| recipe-3 | uses M01's COPY/SH arms | — | **host**: own-prior E-DEC row | — | quantised pub_a49146c8fd | — | absolute-LB kill passes 9–22 % | closure: literature direction |
| recipe-2 | own-prior clause belongs in M01's ladder | — | — | **host**: nested static row | — | chemistry guard needs acidic-OP data | G80 ≈ 200 units | closure: "spread first" is definitional |
| recipe-4 | arm in M03 (inside M01) | — | — | — | duplicate rule for cross-acid copies | — | ≈ 50 campaigns for 2/3 power | closure: no unshrunk extrapolation |
| lab-4 | chemist message moves to M11 | — | — | — | cross-publication JAEA copies | frozen laboratory key at arrival | **host**: lab units for Gd contrasts | tolerance looser than sealed rule |
| lab-2 | any second own-campaign ligand (MX) equals the reference | low-D arm (ε(1 + 1/D)) | — | — | three near-copies | — | lab-unit kill operating characteristics | host is M11 / M12 |
| shift-2 | AINT arm via M03; "bracket the acidity" advice | — | — | — | pub_19783c039b table-vs-figure check | — | dz 0.41 → ≈ 37 families | closure lines |

---

## Cross-cutting lessons from round 2 (rules for round 5)

These add to IDEAS_RANKED's eight cross-cutting rules.

9. **Every kill rule is stated with its operating characteristics**: P(kill | the corpus effect is true) and P(pass |
   null), at the realistic G. Several round-2 kills fired on non-significance or could not fire at all:
   - confirm-3's A1 kill withdraws an exactly calibrated Gd term in 56 % of G = 8 replicates;
   - recipe-3's absolute-lower-bound rule passes a true ≈ 0.92 rule only 9–22 % of the time with 11 families;
   - lab-1's external kill falsely kills a true effect 41–52 % of the time at 8 laboratory units;
   - recipe-2's "HULL − random" kill cannot fail.

   Every decision has an explicit inconclusive zone, and **non-rejection never switches the deployed model.**
10. **No Holm inside a claim at small G.** With exact sign-flip tests, G = 6 cannot reach 0.0125. Use fixed-sequence
    gatekeeping and declare the G floor in advance (confirm-4, confirm-1).
11. **State power in worlds, not one number.** Corpus-resampled power assumes new units are exchangeable with corpus
    units. For round 5's supply, report at least a corpus-DGA world, a halved-effect world and a cation-exchanger or
    null-like world (confirm-1, -2, -4).
12. **The laboratory is the unit for any Gd-amplitude claim.** The Gd chord is shared across a laboratory's papers
    (+0.37 to +0.47 on publication means; +0.73 cell-level for the same extractant) even though the pooled per-metal
    fingerprint is not (lab-4, Skeptic B). Several papers from one group on one ligand family are one unit.
13. **The cheapest same-information comparators that changed conclusions in round 2:**

    | idea | comparator that changed the conclusion |
    |---|---|
    | shift-1 | SH with 2 reads, not B1 with 4 reads |
    | shift-2, recipe-4 | piecewise-linear bracketing interpolation (AINT) or the bracketing-pair mean (PAIRMEAN) |
    | recipe-1 | the best label-free no-prior policy (fixed ladder or DOPT_Q-nested), not HULL |
    | recipe-3 | every arm on identical rows at matched coverage |
    | lab-1 | a count- and chemotype-matched, level-matched placebo from other campaigns, not an all-publication pool |
    | lab-2 | any second ligand of the same campaign (MX, OFFREF) |
    | recipe-2 | HULL |

14. **Weights in placebos must match the arm's weights.** A pooled-donor placebo with w ≈ 1 loses for reasons unrelated to
    the claim (lab-1's MG_LM10_all; recipe-1's literature arm).
15. **Duplicate scan before any sibling or donor pool is built**: bit-identity, constant-offset copies and quantised
    values, within and across publications (see § "Duplicate and provenance candidates"). The V2 lists are not enough.
16. **The ICP-MS cost frame removes most "save a measurement" claims.** recipe-2, recipe-3 and lab-2 lose their practical
    value once all 14 Ln come from one run. State the cost scenario (ICP-MS, radiotracer, literature mining) in every
    practical claim.

---

## Surviving ideas, ranked

### 1. confirm-4: C-ECT, condition-transfer confirmation on new sibling families (SEAL → M05 and M01 kill (iv))

- **Hypothesis (corrected).** In a new laboratory × extractant family with at least 3 full-series conditions, for a target
  that differs from its siblings by a change that alters the SF pattern (T1 aqueous medium, T2 organic phase, T3 mixed),
  the shrunk own-series template SH (K = 1.78) with 4 reads beats B1 with 4 reads. Secondary: SH with 3 reads against B1
  with 4. Not claimed: SH with 2 reads; and "extractant transfer" as distinct from "campaign fingerprint", unless the
  data separate them.
- **Mechanism.**
  - Short-range roughness is shared within a campaign (r = 0.36) and not across laboratories, so averaging one's own
    series removes it.
  - For ideal cation exchange (D ∝ [HA]³/[H⁺]³ for every Ln³⁺), log SF does not depend on pH or [HA], so pH or [HA] scans
    ("T0") are near-copies by construction and serve only as a data check.
  - The invariance fails in buffered or complexant media (acetate, lactate, DTPA: TALSPEAK-type systems are strongly
    pH-dependent), at high loading and at high acidity.
- **Quick-check result (EXPLORATORY).**
  - PRIMARY stratum, 14 publication × chemotype units, BPG:

    | contrast against B1 with 4 reads | wins / losses | mean log SSE ratio |
    |---|---|---|
    | SH, 4 reads | 13 / 1 | +0.66 |
    | SH, 3 reads | 12 / 2 | +0.42 |
    | SH, 2 reads | 8 / 6 | +0.07 |
    | COPY, 0 reads | 4 / 10 | −0.39 |

  - Exact sign-flip power, one-sided 0.025, SH with 4 reads: 0.63–0.67 at G = 6, 0.89–0.93 at G = 7, 0.95–0.99 at G = 8.
  - Acid-concentration-only families: 5 of 5 win. SWAP-null size ≤ 0.033.
- **Reproduction.**
  - Skeptic A reproduced everything with independent code: B1 to 8.9e-16, SH and COPY to 2e-15, 13 / 1 at +0.661, power
    0.643 / 0.915 / 0.984.
  - The result holds under every robustness check A ran:

    | check | result |
    |---|---|
    | leave-one-publication-out | never below 12 / 1 |
    | with C5-BTBP | 14 / 1 |
    | without pub_b0b3296ec5 | 12 / 1, +0.636 |
    | constant-offset near-copy siblings removed (6.2 % of slots) | 13 / 1, +0.625 |
    | V2-5 floor readings dropped | 11 / 1 |

  - Skeptic B reproduced it as well (13 / 1, +0.661; power 0.645 / 0.921 / 0.987).
- **What the skeptics found.**
  1. **Holm inside the claim cannot reject at the declared minimum G.** At G = 6 the first Holm step (0.0125) is below
     the smallest attainable exact p (1/64 = 0.0156). At G = 7, Holm cuts primary power to 0.60–0.69.
  2. **The design's unit statistic is undefined on corpus data.** The design says "mean of per-target log ratios";
     target 63d11bbb7d28eb4c (pub_c8195b4ca7) has SSE_B1 = 0, so its log ratio is −inf. Per-target ratios range from
     −8.2 to +6.1. The power was computed with the summed-SSE family ratio.
  3. **Family-level T0 classification leaks near-copies.** A pH × [HDEHP] × 2-diluent family (Chiba) would count as
     "mixed", yet each target's template would be dominated by T0 near-copy siblings.
  4. **The power describes a DGA world.**
     - 13 of the 14 units and 94 of the 99 PRIMARY cells are DGA. The one non-DGA unit (DMDPhPDA, pub_8415538291) is
       provenance-pending. The bench holds no cation-exchanger family with 3 or more cells and ≥ 10 metals.
     - Non-DGA CLEAN units without C5 give 4 wins / 1 loss, mean +0.14, mostly with one sibling.
     - Power falls to 0.31 / 0.45 / 0.50 at G = 6 / 7 / 8 if 25 % of new families are null-like, and to
       0.14 / 0.19 / 0.22 at 50 %.
  5. **SH is not distinguishable from cheaper own-series rules.** Against SIB it is 9 W / 5 L (p 0.34); against the
     nearest-condition sibling, 7 / 7. The claim is "own-series template + 3–4 reads", not "K = 1.78".
  6. **Laboratories cluster families.** With families in laboratories of 2 at ICC 0.3–0.5, a family-level sign-flip has
     size 0.026–0.054 at nominal 0.025.
  7. **The T0 "void" rule is uncalibrated.** No T0 family exists in any repository data. In DGA acid-only families, COPY
     beats B1 with 4 reads in only 2 of 5, because B1 is already very precise on smooth series.
  8. **Supply.** P4 verifies about 4 laboratories with ≥ 3 full-series conditions (KU Leuven, Chiba, Turanov, Atanassova).
     After pair-level T0 classification, at most 2–3 non-T0 verified families remain, against a minimum of 7.
  9. **Overlap.** Skeptic B: this is M05 plus M01 kill (iv) re-parameterised. The new content is the T0–T3 taxonomy, the
     SH-with-3-reads secondary and the power table.
- **Corrected design (both skeptics' fixes merged).**
  1. **Host.** The pre-registration annex of M05 and M01 kill (iv). No separate slot.
  2. **Unit.** The laboratory, from P4's frozen lab_id. Families are nested within laboratories, and each laboratory has
     weight 1. The family-level p is reported next to the laboratory-level p. With 5 or fewer laboratories an exact test
     cannot reach 0.025, so the result is a descriptive case series.
  3. **Statistic.** Per family, log(Σ SSE_B1@4 / Σ SSE_SH) over leave-one-condition-out targets. Exclude targets whose
     scored pairs are all fixed by the reads. Flooring each target's SSE at a stated noise variance is a sensitivity.
  4. **T0 at pair level.** In the primary analysis, SH uses only siblings that differ from the target in a T1–T3 field. A
     target is admitted only if at least one such sibling remains. The T0-inclusive version is a sensitivity.
     - T0 means: pH, [HA] or phase-ratio changes only; a cation exchanger; no pH-dependent aqueous complexant or buffer;
       trace loading; below the acidity where solvation extraction starts.
     - The mechanism class comes from a frozen map applied blind to values. The map must say where bifunctional ionic
       liquids, quaternary-ammonium nitrate and HTTA + TOPO synergists go.
  5. **Multiplicity.** Fixed sequence: SH@4 vs B1@4 at one-sided 0.025; then SH@3 vs B1@4 at 0.025 only if the first
     passes. Replay: primary 0.92 / 0.90 at G = 7; secondary 0.71 / 0.62 at G = 7 and 0.87 / 0.74 at G = 8 (τ 0 / 0.065).
     Minimum G = 7 laboratories (8–9 if the package's α/3 applies); G = 6 would need 6 wins of 6.
  6. **Comparators on identical rows.** B1@4, COPY, the nearest-condition sibling, SIB. The mechanism contrast is
     SH − OFF_XEXT (or PB_PUB), registered before data where an other-extractant pool exists. It replaces M01 kill (i),
     which P1 showed to be an identity on aligned reads.
     - Default wording: "same-campaign information (extractant transfer and campaign fingerprint not separated)".
     - Rewording to "campaign fingerprint" needs at least 4 families with a pool, and a threshold set before data, away
       from the seen 43–47 %.
  7. **T0 control as a data-integrity flag, not a void switch.** Flag a family if the across-condition RMS difference of
     in-window adjacent log SF exceeds 0.10 log. HDEHP/P507-class series span about 5–6 decades from La to Lu, so a
     single-condition "full series" carries floor or saturation values; that is a measurement issue, not a refutation.
  8. **Power stated before freezing, in three worlds:** corpus DGA (0.90–0.92 at G = 7), halved effect (0.07–0.30 at
     G = 6–10), cation-exchanger or null-like (0.01–0.58 at G = 6–7, depending on the noise model). Plus the 25 % and
     50 % null-like mixtures.
  9. **Descriptive report by mechanism class**: neutral solvating, cation exchanger, ionic liquid.
- **Kill criterion (laboratory level).**
  - Confirm on exact one-sided p ≤ 0.025 with at least 7 laboratories.
  - Kill if the mean log ratio is ≤ 0 or the win share is < 0.5.
  - Anything else is inconclusive. A small real non-DGA gain (corpus +0.14 log, about 13 % SSE) is "below MDE", not
    "killed".
  - Fewer than 7 laboratories: descriptive case series, with no kill or promote language.
- **Practical value (what a chemist would do differently).**
  - If confirmed: a DGA-type laboratory that has at least 2 prior full series and works with radiotracers or subset feeds
    measures 3–4 metals at a new acid or diluent condition and predicts the rest from its own series. That saves about one
    tracer against B1 with 4 reads.
  - A curator imputes the missing metals of a partially reported condition from the same paper's full series.
  - An ICP-MS laboratory changes nothing in measurement. For pH scans the chemist copies the SFs (textbook mass action).
  - If killed: use B1 even when an own prior series exists.
- **Cost.** Hours to seal (the code exists in P1 and `confirm_out/p1`). The decisive leg is data-gated: days of
  extraction, often closed access (CIAC, Utkal and CSM are closed), the user's permission (U3) and the M22 digitisation
  gate for figure-only series.
- **Prior.** A 0.50 · B 0.15. Consolidated: about 0.75–0.8 that own-series information beats B1@4 in genuinely new non-T0
  families; about 0.3 that at least 7 independent such laboratories arrive; so about 0.15–0.25 for a decisive pass.
- **Relations.**
  - **M01:** this is M01's prospective claim, kill (iv). It inherits SH, K and the CLEAN rule, and replaces kill (i) with
    SH − OFF_XEXT.
  - **M09:** the T0 flag uses in-window adjacent SF.
  - **M14, M18:** none.
  - **M25:** pub_b0b3296ec5 (41 % of the corpus gain) and pub_8415538291 are provenance-pending. The bit-identical
    cross-acid pairs go into the CLEAN rule.
  - **M33:** the metadata-only supply audit of non-T0 families runs through the M22/M33 ledger before anything is
    extracted.
  - **M34:** laboratory unit, exact sign-flip G floor, fixed sequence, two-stage null worlds.
  - **M35:** one-sided 0.025 per test, explicit inconclusive zone; E-CT is never quoted beside 0.452.

### 2. confirm-1: CP-13, one frozen confirmation package sized to the real supply (SEAL → M34, M35, M33)

- **Hypothesis (corrected).** About 7–13 new laboratories (P4 gives 13 expected; the P4 skeptic's corrections give 7–11)
  can decide only a few questions. If round 5 seals only those, each with a minimum G and with every undecidable
  endpoint excluded in advance, the supply turns into decisions or into honestly labelled case series, not into
  "underpowered" reports. Excluded in advance: pooled or E-A claims, a TOST on β, the window as a row flag, "two reads plus
  your own series", the sc009 Nd dip, and any DGA-only Gd test.
- **Mechanism.** B1's gain sits about 95 % on the Eu/Gd and Gd/Tb pairs. A unit's model-free Gd chord (log D(Gd) minus the
  Eu–Tb radius interpolation) carries the Gd-transport question at lower G than SSE tests. But see confirm-3: the link is
  algebraic, not evidence.
- **Quick-check result (EXPLORATORY).**
  - Decidable:

    | claim | power |
    |---|---|
    | C-GD (heavy-selective non-DGA mean chord < β) | 0.62–0.74 at G = 6, 0.88–0.94 at G = 8 |
    | A1 (a dip exists) | 0.83 at G = 8 |
    | full B1 chain A1 → A2 → A3 | 0.36 / 0.23 at G = 8, 0.70 / 0.50 at G = 13 |
    | E-CT, SH@4 vs B1@4 | 0.89–0.93 at 7 families |

  - Not decidable: TOST ≤ 0.08; window flag ≤ 0.15; SH@2 ≤ 0.06.
  - The between-unit chord sd is 0.12–0.16, not about 0.1. Raw unit-mean t-tests are anti-conservative (0.043–0.050);
    winsorising to [−0.40, +0.20] restores size ≤ 0.027.
- **Reproduction.**
  - Skeptic A reproduced the strata, units and winsorised chords exactly:

    | stratum | units | mean chord | sd |
    |---|---|---|---|
    | heavy-selective non-DGA | 12 | −0.217 | 0.101 |
    | light-selective non-DGA | 13 | −0.073 | |
    | DGA | 19 | −0.051 | |

  - A also reproduced C-GD power 0.74 / 0.94 / 0.99, chord-vs-gain Spearman −0.957 (rule right in 92 % of 127 cells) and
    E-CT 13 W / 1 L with exact power 0.63 / 0.91 / 0.98.
  - Leave-one-publication, copy-group, cluster or chemotype-out keeps the heavy-selective mean in −0.20 to −0.25.
  - Skeptic B reproduced the stratum and the power by algebra.
- **What the skeptics found.**
  1. **Stratum mismatch (A, blocking).** The planning numbers come from the slope-defined heavy-selective stratum, but
     the package freezes a mechanism-metadata stratum. On the corpus the metadata class (aromatic N-donors plus acidic
     P) has 19 non-DGA units, mean −0.158, sd 0.125, and only 58 % of them are slope-heavy-selective. Planned power for
     the frozen rule is then 0.13 / 0.23 / 0.28 / 0.33 at G = 4 / 6 / 8 / 10.
  2. **Planning chemistry (A and B).** 10 of the 12 corpus heavy-selective units are aromatic N-donors. The only acidic
     organophosphorus unit is one cell of dihexyl hydrogen phosphate (chord −0.37). The supply is acidic OP and
     chelators. A search-snippet reading of the OECD-NEA HEH[EHP] entry (not verified; the page fetch returned 404)
     implies a P507 chord of about −0.105, that is, about β; for P507-like units C-GD power would equal its size.
  3. **Supply to G (A).** Verified heavy-selective new units with Eu, Gd and Tb: 2 (Chiba HDEHP, Osaka HDEHP), plus
     Sofia 8-HQ as probable. The realistic G_HS is 3–5, below the frozen minimum of 6.
  4. **E-CT planning (A and B).** 14 of the 15 PRIMARY families are DGA. Excluding T0 (pH and [HA] scans, the usual
     designs for acidic OP) probably leaves 4 or fewer primary-class families. For cation exchangers, T0 and T1 overlap
     as written.
  5. **Independence (A and B).**
     - 6 units from 3–4 laboratories at intra-laboratory ICC 0.3–0.5 give size 0.032–0.051 at nominal 0.025 (A).
     - Two-stage (chemotype → laboratory) nulls give 0.041–0.049 for balanced compositions at ICC 0.3, and 0.06–0.14 for
       unbalanced ones or higher ICC (B).
     - By the package's own rule (drop any test with size > 0.035), C-GD would be dropped unless the laboratory is the
       unit.
  6. **What can be decided is already known (B).** The Gd break of HDEHP-type extractants goes back to Peppard (1957).
     The open questions (A2 → A3, C-ECT) are the ones the supply cannot decide.
  7. **A1's decision rule (B).** "Fail to reject → withdraw B1's Gd term" withdraws a real effect about 17 % of the time
     at G = 8.
- **What survives.** The package as a frame: per-claim minimum G, an exclusion list, a metadata-only supply-audit kill and
  a size gate. Its planning values and its targeting do not survive as written.
- **Corrected design: sealed text, no slot.**
  1. **Into M34:**
     - a per-claim minimum-G table and a supply-scenario power table, recomputed lab by lab from P4 (G_HS, G_Gd, G_ECT),
       each with the probability of reaching its minimum;
     - a size gate with two-stage (chemotype → laboratory) null worlds at ICC 0.23–0.69, next to the i.i.d. worlds;
     - one value per laboratory for every Gd and E-CT claim (laboratory-bootstrap power at G = 6 is 0.742 against 0.736
       for units, so it costs nothing).
  2. **Into M35:** the exclusion list, one-sided 0.025 per claim, no global claim, and three-way decisions. A1
     non-rejection reads "inconclusive", never "withdraw".
  3. **Into M33 and M22:**
     - the supply-audit kill (metadata only): if every claim is below its minimum G, no confirmation runs and the round
       is reported as descriptive;
     - eligibility: the unit itself reports Eu, Gd and Tb; values below DL or LOQ excluded; aqueous-complexant media
       excluded (M29 rule);
     - a frozen mechanism-class tag (see confirm-2).
  4. **Re-aim the targeted search at E-CT-capable laboratories**: at least 3 non-T0 full-series conditions of one
     extractant, full La–Lu. That is the only open question here. Acidic-OP Gd triads are a handbook question.
  5. **Before sealing:**
     - compute and state β_frozen (the B1 refit on all 229 cells; fold betas range from −0.030 to −0.140), with C-GD power
       at that value. At β = −0.118, power falls to 0.48 at G = 6 and 0.71 at G = 8; at −0.072 it is 0.89 / 0.99;
     - fix per-family planning amplitudes from primary SF tables (Peppard for HDEHP; a handbook for P507), never from the
       corpus, and report power for an HDEHP-like and a P507-like centre separately.
  6. **Chord hygiene:** V2-5 floor censoring, and a resolution rule for sources reporting D to one decimal (about ±0.05 log
     of quantisation noise, comparable to β/2).
- **Kill criterion (package level).**
  - At the supply audit, a claim with G below its minimum is descriptive.
  - Any test whose size exceeds 0.035 (Clopper–Pearson lower bound) under i.i.d. or two-stage nulls is dropped before
    sealing.
  - Per-claim kills are in confirm-2/3 and confirm-4.
- **Practical value.**
  - It tells the user what to collect: full La–Lu series (at least Eu, Gd and Tb), at least 3 non-T0 conditions per
    extractant, from independent laboratories.
  - It tells a chemist in advance what each outcome would change: apply B1's Gd term, a deeper class amplitude, or none;
    and whether "one full series per extractant, then 3–4 metals per new condition" is supported.
- **Cost.** Hours to seal. Desk work (web reading) for the primary SF tables. About 2–3 days for a targeted search,
  data-gated on the user's access and permission.
- **Prior.** A 0.35 · B 0.20. Consolidated: about 0.3 that the corrected package decides at least one claim with round
  5's supply. About 0.9 that sealing it protects round 5 from undecidable or mis-sized tests.
- **Relations.**
  - **M01:** C-ECT is M01 kill (iv); confirm-4 holds the design.
  - **M09:** the window as a row flag is excluded (power ≤ 0.15).
  - **M14, M18:** not in the package; E-DEC and the k ladder stay descriptive.
  - **M25:** β_frozen is learned mainly from the 69 unlocated cells (P5), so its anchor depends on M25. Values are sealed
    by a separate agent.
  - **M33:** the supply audit, eligibility and tags; M33's frozen one-shot scoring is the vehicle for C-GD and C-B1.
  - **M34:** host of the minimum-G table, the two-stage nulls and the laboratory unit.
  - **M35:** host of the exclusion list and the three-way decision.

### 3. confirm-2: C-GD, the Gd break of cation-exchange extractants is deeper than B1's amplitude (SEAL → stratum of M33's Gd-dip test)

- **Hypothesis (narrowed).** In new laboratories using cation-exchange extractants, the laboratory-mean Gd chord is below
  B1's β_frozen (about −0.09). That is, B1's DGA-dominated amplitude under-corrects them.
  - The stratum covers acidic organophosphorus (HDEHP/D2EHPA, HEH[EHP]/P507, Cyanex 272), carboxylic acids, and chelating
    cation exchangers (β-diketones, acylpyrazolones, 8-hydroxyquinolines).
  - The stratum is defined by mechanism metadata, not by the sign of the series slope.
  - Soft N-donors (BTP, BTBP, PDA, phenanthroline diamides) are a separate descriptive family, because their selectivity
    direction varies (59 % heavy-selective in the corpus).
- **Mechanism.**
  - The gadolinium break and tetrad effect in acidic organophosphorus extraction are textbook results (Peppard, 1957 and
    1969). Gd/Eu is one of the lowest adjacent SFs with P507.
  - The planning value −0.26 (HDEHP: Gd/Eu ≈ 1.5, Tb/Gd ≈ 5) is not verified from a primary table.
  - In the corpus, the winsorised chord is −0.217 in the slope-defined heavy-selective non-DGA stratum, −0.073 in the
    light-selective one and −0.051 for DGA. B1's fold β averages −0.094.
- **Quick-check result (EXPLORATORY).**
  - One-sample t on winsorised unit chords against β, one-sided 0.025: power 0.62–0.74 at G = 6, 0.88–0.94 at G = 8,
    0.97–0.99 at G = 10. Null size ≤ 0.019.
  - The heavy- vs light-selective Welch contrast has only 0.33–0.72 power.
  - The sign of the slope from the 4 reads agrees with the full-series sign in 96.6 % of read sets.
- **Reproduction.**
  - Skeptic A reproduced it exactly with its own code (bootstrap power 0.73 / 0.94 / 0.98) and checked robustness:

    | check | heavy-selective mean chord |
    |---|---|
    | leave-one-publication-out | −0.202 to −0.233 |
    | leave-one-cluster-out | −0.202 to −0.247 |
    | without C5-BTBP | −0.209 |
    | without C5-BTBP and DMDPhPDA | −0.210 |
    | 3 most negative units dropped | −0.171 |
    | curvature-adjusted chord | −0.207 (light-selective stratum −0.050) |

  - A found that a heavy-selective amplitude fitted on other publications cuts held-out heavy-selective Gd-cell SSE
    against β by 59 % (51 % unit-weighted; 8 of 11 publications); by 37 % / 46 % (6 of 9) without C5-BTBP and DMDPhPDA;
    and by 29 % / 42 % (4 of 6 clusters) under leave-one-cluster-out. The light-selective amplitude does not transfer
    (−13 % to −15 % against β).
  - Skeptic B reproduced the split exactly.
- **What the skeptics found.**
  1. **The stratum is not operationally defined (A).** "Acidic OP + BTP/BTBP" gives 4 units at −0.289. Adding PDA and
     phenanthroline amides gives 10 units at −0.188. The slope stratum gives 12 at −0.217. The same chemotype (sc029) is
     heavy-selective in one publication and light-selective in others.
  2. **The corpus stratum is not the tested chemistry (A and B).** 11 of 12 units are N-donors or CMPO-type. The whole
     bench holds one acidic cation-exchange cell with Eu, Gd and Tb, and no chelating cell. Eight of the 12 units are
     single cells, whose chord noise (sd 0.127) matches the between-unit sd (0.10).
  3. **Power is overstated (A).** With the analytic non-central t at −0.217 and sd 0.101, power is 0.68 / 0.85 at
     G = 6 / 8. At sd 0.15 it is 0.38 / 0.53; at a shrunk mean of −0.175 it is 0.37 / 0.51.
  4. **The answer is already known for the target chemistry (B).** A pass would re-confirm 1960s chemistry. It would say
     nothing about the post hoc corpus lead, which was an N-donor lead.
  5. **"Heavy-selective" is not the mechanism (B).** 16 of 19 DGA units are heavy-selective and still shallow (−0.063).
     Inside non-DGA, the chord–slope relation is carried by the soft N-donors (Spearman −0.77); the other 8 non-DGA units
     show none (−0.19, p 0.65). In a model with the signed slope, the heavy-selective coefficient drops to −0.044 (se
     0.050) (A).
  6. **Deployed-row value (B).** In B's out-of-fold refit on the round-4 rows, the stratum amplitude (fold mean −0.224
     against pooled −0.117) gives R² 0.4586 against B1's 0.4521, but the gain looked like C5-BTBP alone (2.26 of the net
     1.79 SSE gain on heavy-selective rows). Without C5 rows in scoring: 0.4857 against 0.4863. **This note's
     reconciliation check reverses most of that reading** (§ "Reconciliation check"). The net loss outside C5 comes mainly
     from pub_84113c17ad, the P5 component-identity publication (TEHDGA + HDEHP with CDTA/HEDTA recorded as the
     extractant). That is an aqueous-complexant system, which confirm-2's own eligibility excludes, and it is
     light-selective by its full-series slope.
     - Excluding it and C5-BTBP, the stratum amplitude reduces SSE on eligible heavy-selective rows by 4.4 % (BPG) /
       4.0 % (BP): 7 of 12 publications better, +5.2 % / +5.6 % on Gd-hidden rows, about 0 on Gd-read rows.
     - The winners are the deep-chord triazinyl/BTBP units and the single acidic-phosphate cell. The losers are amide and
       CMPO units with shallow chords.
     - So the deployed signal outside C5 follows chord depth (and mechanism), not the heavy-selective label. It is worth
       about +0.002 R² pooled, far below any MDE, and the 7/5 split carries no inference.
  7. **Supply (B).** P4 verifies only Chiba and Osaka (both HDEHP) inside the stratum with Eu, Gd and Tb. The M29 rule
     removes HDEHP papers in lactate, DTPA or acetate media (TALSPEAK-type). Realistic G is 2–5, so the result is
     descriptive.
  8. **Kill rule (A and B).** "Point estimate ≥ β" kills with probability about 0.5 when the true mean equals β, and about
     0 for acidic-OP units. If it fired on acidic-OP units, it would point to data or convention problems (radiotracer
     methods, DL handling, complexant media), not to N-donor-specific chemistry.
  9. **Provenance (A).**
     - The Gd–Lu segments of 3 C5-BTBP cells are constant-shift copies (shift −5.37, sd 0.012–0.026) of DMDPDGA cell
       1b71b86063cd1faf (pub_74428aeab2).
     - The pub_723ceb6c2c C5-BTBP cell is a +1.95-shift copy of TODGA cell 354fe634113a078e.
     - pub_e9799b75f3 sc068 and sc070 copy each other on Gd–Lu.
     - None of these moves the stratum mean (−0.217 → −0.217 / −0.224).
- **Corrected design (merged).**
  1. **Host.** A pre-specified stratum of M33's one-shot unit-level Gd-dip test, together with confirm-3's chord interval.
     Not a separate α/3 claim.
  2. **Stratum by frozen mechanism metadata.**
     - Cation exchangers (acidic organophosphorus, carboxylic acids) and chelating cation exchangers.
     - A class table fixed before data says where PDA and phenanthroline amides, CMPO-type phosphoryl acetamides,
       malonamides and neutral organophosphorus compounds go (all outside the stratum).
     - Soft N-donors are a separate descriptive family. The reads-only slope sign, computed without Eu, Gd and Tb so that
       the stratum is not coupled to the chord, is a sensitivity.
  3. **Planning amplitudes per extractant family, frozen from primary SF tables before any new value is read.** Report
     power per centre:
     - HDEHP-like (about −0.26): 0.90–1.00 at G = 6;
     - P507-like (about β): power ≈ size;
     - mixed.
  4. **Unit: the laboratory.** Two HDEHP laboratories are replicates of one chemistry. "Stratum" language needs at least 3
     distinct extractant families; report units per extractant.
  5. **Statistics.**
     - The chord interval (winsorised and raw), read against β_frozen.
     - A model-level contrast: per-unit Gd-pair SSE of B1 with the frozen family amplitude against B1 with β_frozen,
       equal-weight exact sign-flip.
     - Secondary: the chord against the signed series slope measured outside Eu–Tb. In corpus non-DGA it gives Spearman
       −0.63, and steep heavy-selective units average −0.263 against −0.154 for flat ones. A chemist's prediction should
       scale with steepness.
  6. **Media and regime rules, frozen.** Exclude aqueous complexants (lactate, citrate, DTPA, HEDTA, CDTA; decide acetate
     in advance), the solvating regime of HDEHP at high acid, and DL-flagged values. Ionic-liquid diluents are a
     sensitivity, because the mechanism changes there.
  7. **Power from chemistry worlds** (analytic non-central t), not from the post hoc corpus bootstrap.
  8. **Provenance flags** for the C5-BTBP shift copies (to M25).
- **Kill criterion (three-way, on the laboratory-mean chord).**
  - Confirmed if the one-sided 97.5 % upper bound is below β_frozen, with at least 6 laboratories and at least 3 families.
  - Killed ("no under-correction") if the 97.5 % lower bound is above β_frozen. A lower bound above β/2 also raises
    confirm-3's B1-harm flag.
  - Otherwise inconclusive, reported as an interval.
  - The point-estimate kill of the original idea is dropped.
- **Practical value.**
  - For established P507/HDEHP circuits nothing changes: keep using tabulated SFs.
  - For a new acidic-OP or chelating extractant screened with few reads: include Gd in the read set, or apply a family
    amplitude (about −0.2 to −0.26) instead of −0.09. With ICP-MS, measure the Eu–Gd–Tb triad directly.
  - Without the fix, B1 predicts Gd/Eu ≈ 2.2 where about 1.5 is true, roughly twice the stages for that cut (B).
- **Cost.** Minutes of analysis once data exist. Desk work to fix the planning values from primary tables. Data-gated
  through M33/M22.
- **Prior.** A 0.55 · B 0.25. Consolidated: about 0.85 that acidic-OP laboratories show a chord deeper than β if they
  arrive; about 0.3–0.4 that at least 6 eligible laboratories from at least 3 families arrive; so about 0.25–0.35 for a
  decisive pass.
- **Relations.**
  - **M01:** none.
  - **M09:** DL-flagged values are excluded (as in M24).
  - **M14:** none.
  - **M18:** a read-policy note: for cation exchangers, read Gd. DOPT_Q already reads Gd in about 60 % of read sets.
  - **M25:** β_frozen is learned mainly from unlocated cells (P5). The C5-BTBP shift copies and the pub_84113c17ad
    component error are provenance items.
  - **M33:** host (a stratum of its primary Gd-dip transport test against 0 and −0.094).
  - **M34:** laboratory unit, analytic power, two-stage nulls.
  - **M35:** three-way decision.
  - Also: M31's salvage (a family-stratified dip with family values frozen now) is the same idea; M11's in-run TODGA arm
    stays the laboratory route.

### 4. confirm-3: C-B1, B1 transport decided by the Gd chord (SEAL, merged into confirm-2)

- **Hypothesis.** In new laboratories:
  - A1: a Gd dip exists on average (mean chord < 0);
  - A2: B1's frozen term lowers the average unit's Gd-pair error (mean chord < β/2);
  - A3: the deployed B1 pipeline beats B0 for the average laboratory (equal-weight log SSE ratio > 0, with a pooled guard).

  Tested as a fixed sequence.
- **Mechanism.** On Gd-pair rows where Gd is hidden and Eu and Tb are read, B1 helps a unit exactly when
  (c − β)² < c², that is, when c < β/2.
- **Quick-check result (EXPLORATORY).**
  - Unit chord: −0.092 over all units (sd 0.157), −0.142 non-DGA (sd 0.12), −0.027 DGA.
  - Chord vs gain: Spearman −0.957; the rule agrees with the unit's win in 95 % of units.
  - Power on non-DGA chord units:

    | test | G = 8 | G = 13 |
    |---|---|---|
    | A1 | 0.83 | 0.99 |
    | A1 → A2 | 0.48 | 0.76 |
    | full chain (τ 0 / 0.065) | 0.36 / 0.23 | 0.70 / 0.50 |

  - TOST ≤ 0.08 everywhere. DGA units: A2 ≤ 0.10 at any G.
- **Reproduction.**
  - Skeptic A reproduced it independently: unit chords −0.0924 / −0.142 / −0.027; chain 0.37 / 0.24 at G = 8 and
    0.70 / 0.50 at G = 13.
  - Leave-one-publication-out ranges: A1 at G = 8 0.76–0.94; A2 at G = 13 0.69–0.92. Within-publication ICC is about −0.09;
    one-unit-per-laboratory null sizes are ≤ 0.024.
  - Skeptic B reproduced it as well (A1 0.835; chain 0.347 / 0.228 and 0.698 / 0.508).
- **What the skeptics found.**
  1. **The A1 kill fires on non-significance.** "Point estimate ≥ 0 or 97.5 % upper bound > 0 → withdraw B1's Gd term" is
     just "A1 not rejected".
     - When the true mean equals β, where B1 is exactly calibrated, it withdraws the term in 56 % of replicates at G = 8
       and 27 % at G = 13 (A).
     - In the all-chord world it fires 34 % at G = 8; in the light-selective world 55 % (B).
     - At G = 8 the procedure picks B0 twice as often as B1 in a world where B1 is truly better (B).
  2. **Following the rule loses against the status quo (B).** Expected excess Gd-pair loss is worse than always-B1 in
     every corpus world except DGA (all-chord G = 8: 0.0268 against 0.0238).
  3. **In the likely supply, the chain endorses the wrong amplitude (B).** In the heavy-selective world the chain passes
     with power 0.98–1.00 at G = 6–8 and makes B1 the default, although B1 under-corrects by about 2×. An amplitude
     re-estimated from the new units cuts the excess loss by 58 % (0.0250 → 0.0106). That is confirm-2's question.
  4. **The quoted power is for another population.** It assumes A3 runs on chord units only. Run as designed (all units,
     including those without an Eu–Gd–Tb triad), the chain is 0.23 / 0.17 at G = 8 + 3 and 0.54 / 0.37 at G = 13 + 4 (A).
     In B's all-unit chord pool it is 0.17 / 0.11 at G = 8.
  5. **"The chord is a more powerful carrier" rests on an asymmetry (A).** τ noise was added to the SSE arms only. At
     τ = 0, A3 alone matches A2 (0.79 against 0.76 at G = 13).
  6. **The link is algebra, not evidence (A and B).** On Gd-hidden rows the gain equals 2[c² − (c − β)²] to 1.6e-15. DOPT_Q
     reads Gd in 46–60 % of read sets, and those carry 58 % of B1's gain; there the rule is only approximate (cell
     accuracy 0.86–0.88). "Eu and Tb read, Gd hidden", where the identity is exact, occurs in 0.9 % of read sets.
  7. **Post hoc winsorisation and censoring (A).** The [−0.40, +0.20] bounds clip exactly one corpus unit and break the
     linear chord–gain equivalence A2 relies on. V2-5 censoring moves the non-DGA mean from −0.142 to −0.147.
  8. **Overlap (B).** A3 is M34's B1 − B0 positive control run prospectively. A1/A2 restate M31 and H-GD-DONOR.
- **Corrected design: merged into confirm-2.**
  - **One sealed interval** of the laboratory-mean chord, read against 0, β/2 and β_frozen, reported with the per-class
    amplitude:
    - 97.5 % lower bound > β/2: B1 harms the average new laboratory, so use B0 for new chemotypes. This futility rule is
      real but weak: when B1 truly harms, it fires only 0.17 at G = 8 and 0.25 at G = 13; state that next to the rule.
    - 97.5 % upper bound < β/2: B1 helps the average new laboratory (A2).
    - Otherwise: keep B1 as the status quo and decide nothing.
  - **A1 is descriptive only.** Rejecting A2 implies A1.
  - **A3 runs alone and ungated** as M34's prospective B1 − B0 positive control, with the pooled guard. The gate costs
    0.06–0.10 of its power. Restrict it to Gd-eligible units, or quote the all-unit power.
  - The raw-mean A2 is co-primary with the winsorised one. Apply V2-5 censoring and the resolution rule. Report the
    chord-to-gain rule by read configuration, and freeze the DOPT_Q Gd-read rate as part of the claim.
- **Kill criterion.** As above. No kill fires on non-significance.
- **Practical value.** It decides, for a chemist screening a new extractant with 4 measured metals, whether to apply the
  Gd correction. For the acidic-OP supply the actionable answer comes from confirm-2 (a deeper amplitude, or read Gd).
- **Cost.** Minutes once data exist (about 100 lines of code on frozen inputs).
- **Prior.** A 0.40 · B 0.25 as written. About 0.25 that the merged interval leads to a decision with round 5's supply.
- **Relations.**
  - **M01:** none.
  - **M09:** V2-5 floor censoring of the chord.
  - **M14:** B0 vs B1 is also M14's comparator, but no direction calls are made here.
  - **M18:** none.
  - **M25:** B1's Gd anchor depends on M25 (P5).
  - **M33:** the chord is the vehicle of M33's Gd-dip test.
  - **M34:** A3 is its positive control, run on new data.
  - **M35:** status quo by default; nothing switches on non-significance.

### 5. lab-1: E-LAB, calibrate a new ligand from the same campaign's other series (ANNEX → M01, new POPX0 stratum)

- **Hypothesis (narrowed).** Consider a new extractant read at 4 metals (k = 3, DOPT_Q) in a campaign that also reports at
  least one other extractant series with at least 6 metals.
  - The rule: replace B1's shape u with v = (1 − w)u + w·c. Here c is the mean of the other series' quadratic-detrended
    deviations, using only donor metals whose log D lies within 1 log of the target's B1 level, with w = n/(n + 6.1).
  - Claim: this lowers strict-adjacent SSE against B1 on the same reads, and a count- and chemotype-matched,
    level-matched placebo from other campaigns does not.
  - Scope: amide O-donor solvating extractants (DGA, malonamide, DOODA-like) in nitrate media.
  - Separate estimand E-LAB, never quoted next to 0.452.
- **Mechanism (revised by the skeptics).**
  - Most of the edge is level-driven adaptive weighting: the rule falls back to B1 on the 16 % of unread metals with no
    donor within 1 log.
    - A random-donor control with the same per-metal donor counts already gives +7.3 %.
    - The specific level-matched donors add only +1.8 % [−0.5, +4.2] (8 up, 1 down).
    - Level matching still beats the best global shrinkage of the unmatched shape (K = 12: +4.5 %) by +4.7 % [+2.2, +7.2].
  - Sharing is highest at low D (r 0.53 at log D < −2). A shared feed or mass-balance error ε(1 + 1/D) predicts this as
    well as homologue chemistry does.
  - 97.7 % of the "other-chemotype" donor cells share the target's broad ligand class, so campaign and homologue chemistry
    cannot be separated in the corpus.
- **Quick-check result (EXPLORATORY; best of about 25 arms on seen rows).**

  | population | cells / pubs | B1 → MX_LM10 | rel. SSE | units up / down |
  |---|---|---|---|---|
  | POPX | 92 / 16 | 0.444 → 0.494 | +9.0 % [+0.3, +17.6] | 10 / 4, p 0.037 |
  | POPX0 (new extractants) | 60 / 13 | 0.401 → 0.467 | +11.1 % | 8 / 3 |

  - Against the BPG-legal universal version: +13.0 %.
  - The top 3 publications carry 93 % of the gain.
- **Reproduction.** Skeptic A reproduced it exactly (MX_LM10 maximum difference 0.0 over 7,870 rows; +8.96 % / +11.06 %;
  BP +9.20 % / +11.34 %). Skeptic B reproduced it too.
- **Robustness.**

  | check | result |
  |---|---|
  | against count- and chemotype-matched placebo E_PBC (A) | placebo −1.4 % vs B1; MX_LM10 − E_PBC +10.2 % [+1.6, +18.9]; 13 of 16 publications, p 0.011; 10 of 12 laboratories, p 0.006; POPX0 +11.9 %, 11 of 13 |
  | against count-matched placebo E_PCM (B) | placebo −2.7 %; MX_LM10 − E_PCM +11.4 % [+2.3, +20.4], 11 / 3, p 0.008; +5.4 % (p 0.033) without the two lead publications |
  | leave-two-publications-out (without pub_126bbe7976, pub_1bd76c48d9) | +2.6 % |
  | leave-three-out | +2.5 %, p 0.16 |
  | 4 source clusters | 3 up / 1 down, p 0.125 |
  | near-copy donors removed | +8.55 % / +10.57 % (A, sd < 0.05); +7.6 % [−1.1, +16.4], p 0.093 (B, label-free 0.15 rule) |
  | by class (B) | DGA +11.2 % (59 cells), malonamide +16.5 % (8), pyridine/phenanthroline amides 0.0 % (12), other −2.2 % (10) |
  | "worth about one read" | MX_LM10 at 3 reads vs B1 at 4: 6 up / 7 down, so not resolved |
  | same-series vs other-series donors (36 cells, 4 publications) | +3.8 % vs −3.6 % |

- **What the skeptics found.**
  - Duplicates the V2 lists miss (TDDGA ≡ TDdDGA bit-identical; DMDPDGA ≈ TODGA; DPiperidineDGA ≈ TEtOMeDGA; a
    cross-publication TODGA copy).
  - The "not campaign-specific" control MG_LM10_all pools hundreds of donors (w ≈ 1) and breaks on pub_5a68dc5665
    (−129 %), so it can hardly fail to lose.
  - The external leg cannot decide. At most about 3–6 laboratory units exist (P4-002/003/005/055; P4-044/063/086
    uncertain), 3 of them corpus-linked. P4-069 has only 4 Ln. P4-012 is probably pub_b9dfc468cd. 15 of the 16 new
    laboratories work on chemotypes where the in-corpus prior is null.
  - The kill rule falsely kills a true effect 41–52 % of the time at 8 units.
  - Practical value is mostly interpretive under ICP-MS.
- **Corrected design.**
  1. **Host.** An annex arm of M01 for the POPX0 stratum (a new extractant in a known campaign), which M01 does not cover.
     It pairs with P1's other-extractant placebo and OFF_XEXT. Together they make M01's mechanism contrast.
  2. **Primary analysis with near-copies excluded.**
     - A frozen label-free rule: a donor is a near-copy if it is in the same medium and its log D lies within 0.15 of the
       target's at all 4 read metals. Metal-concentration duplicates are dropped too.
     - The V2-missed duplicates are quarantined (§ "Duplicate and provenance candidates").
     - Reproduction targets are restated on the cleaned population.
  3. **Comparator for "not campaign-specific":** E_PBC / E_PCM (same n_m and w_m, same chemotype or broad class, other
     campaigns), not MG_LM10_all.
  4. **Mechanism controls:** E_RM (random own-campaign donors with matched counts), the global-K shrinkage control,
     same-series against other-series donors, and a low-D band split.
  5. **Scope.** Amide O-donor solvating extractants in nitrate media. Other classes are declared null-prior strata.
  6. **Units.** Laboratory (frozen author-overlap key), with the cluster level and leave-two-out reported.
  7. **External leg.** Count units from P4 metadata before any transcription. Fewer than 8 independent laboratories with at
     least 2 extractants × at least 6 Ln in one campaign: descriptive. Fold transcription into M05/M22 rather than spend
     separate days.
  8. Every in-corpus number is labelled best-of-about-25 arms; expect regression.
- **Kill criterion (8 or more laboratory units).**
  - Kill if the laboratory-median relative SSE (MX_LM10 − B1) is ≤ 0, or its one-sided 95 % lower bound is ≤ 0.
  - Narrow to "level-conditional weighting, not campaign" if MX_LM10 − E_PCM ≤ 0.
  - Harm flag if any class with at least 3 units loses more than 5 %.
  - State the operating characteristics: false kill 0.41–0.52 at 8 units, 0.25 at 11, 0.06–0.10 at 16.
  - The "bug kill" (reproduce +9.0 / +11.1 within 0.5 points) is replaced by a rerun on the cleaned population.
- **Practical value.**
  - Mostly interpretive: compare ligands within one run; treat a per-metal anomaly shared by every ligand in a paper as a
    campaign signature until another campaign reproduces it; curators keep campaign identity and every series of a
    multi-ligand paper.
  - Predictive use only for partial-series screening (radiotracers, literature mining) of a new DGA or malonamide homologue
    run next to full series: at most about +0.05 R².
- **Cost.** Minutes to rerun with frozen code. The external leg needs days of transcription and the user's permission.
- **Prior.** A 0.30 · B 0.15. About 0.5 that the in-campaign DGA/malonamide channel is substantively real; about 0.15 that
  it passes an external kill with the supply that exists.
- **Relations.**
  - **M01:** annex arm (POPX0), next to OFF_XEXT and PB_PUB; overlaps M06's salvage (same campaign against other
    publication).
  - **M09:** sharing peaks at log D < −2; level matching uses a ±1 log window.
  - **M14:** none.
  - **M18:** the read-equivalence claim is unresolved.
  - **M25:** duplicates for the V2 addendum.
  - **M33:** count multi-ligand campaigns in the supply ledger.
  - **M34:** laboratory unit and kill operating characteristics.
  - **M35:** separate estimand E-LAB.

### 6. lab-3: closure line for E-A, per-metal priors from other publications (LEDGER → M35 closure ledger)

- **Hypothesis (scoped by both skeptics).** Under E-A (BP/BPG, fold-legal training donors only), per-metal deviation priors
  learned from other publications do not beat B1. This covers unconditional priors (a donor mean, or the ridge-fitted B2)
  and priors level-matched to the target's predicted D at each metal.
- **Mechanism.**
  - Cross-publication sharing of per-metal deviations is indistinguishable from 0:

    | relation | r | publication-bootstrap 95 % |
    |---|---|---|
    | same extractant, other publication (XE) | 0.010 | [−0.23, +0.17] |
    | same chemotype, other publication (XC) | 0.011 | [−0.07, +0.11] |
    | other extractant, other publication (XX) | −0.033 | |
    | XE, level-matched | 0.024 | |

  - The corpus-mean pattern is about 4 % of the variance and mostly sits in B1's Gd term. A foreign shape adds its own
    variance and removes nothing.
- **Quick-check result (EXPLORATORY), as recomputed by both skeptics.**

  | arm (BPG, rel. SSE vs B1) | value | fold-legal? |
  |---|---|---|
  | MG (all other publications) | −0.8 % | no: about 60 % of donors are test-fold cells |
  | MCO (same chemotype) | +1.8 % (−0.2 % without C5) | no: E-KX, all donors co-held-out |
  | MG_LM10_all | −2.6 % | no |
  | E_MRP (one random publication) | −4.8 % | |
  | **MG_tr** | **−4.8 % [−8.5, −1.2]** | yes |
  | **MG_LM05 / 10 / 20_tr** | **−4.8 / −5.8 / −6.8 %** | yes |

  - MG_LM10_tr: publication jackknife [−10.1, −1.5], cluster jackknife [−10.8, −0.8]; 13 publications up, 26 down; without
    C5 −6.8 % [−11.8, −1.8].
  - The shrinkage sweep is monotone toward λ = 0 (λ 0.1 → −0.06 %, 0.25 → −0.33 %, 0.5 → −1.2 %).
  - BP agrees to within 0.003–0.004.
- **Reproduction.** Skeptic A reproduced everything with independent code (B1 to 8.9e-16 on 18,020 rows). Skeptic B
  reproduced it and ran the proposed replay in 25 s.
- **What the skeptics found.**
  - The same-chemotype leg is empty under E-A: 0 of 1,145 target × seed × fold instances have a same-chemotype training
    donor, because BP/BPG hold out chemotypes. So MCO_tr equals B1 exactly.
  - The kill criterion can never reopen the route. B1's own gain over B0 fails it: +5.9 % [−0.6, +12.3].
  - The line as written would wrongly close M07. The lab-3 estimator at Nd, with E-KX donors, reproduces M07's sc009 lead
    (Pr/Nd 0.26 → 0.54).
  - "All per-metal skill is local to the campaign" is false: the Gd break is universal (that is B1), and the Nd dip
    transfers within the TODGA class.
  - Mostly a duplicate of the existing B2 line. The new content is the level-conditional, non-parametric donor mean.
- **Ledger line (adopt now).** "E-A only: per-metal deviation priors from training-fold publications, unconditional (donor
  mean or ridge-fitted B2) or level-matched at 0.5 / 1 / 2 log, do not beat B1 under BP/BPG. They are worse by 1.5–7 %
  relative SSE (BPG MG_tr −4.8 % [−8.5, −1.2]; MG_LM05/10/20_tr −4.8 / −5.8 / −6.8 %), robust to tolerance, Gd handling,
  shrinkage and C5-BTBP. Level matching makes it worse. Same-chemotype priors cannot be tested under E-A and are not
  closed. The E-KX sc009 Nd dip (M07) is carved out. Grade FIRM on significant harm, not on the reopen bar."
- **Reopen bar.** Only a pre-declared fold-legal variant that beats B1 by at least the size 13 clusters can detect (about
  4–6 % relative SSE, about 0.02–0.03 R²), with a positive cluster-jackknife lower bound with and without C5-BTBP. State
  that bar instead of "cannot beat B1 by 1 %".
- **Practical value.** Do not "correct" adjacent SFs with whole per-metal (or level-matched) patterns from other
  laboratories' papers. The universal Gd break is already in B1. For TODGA-class DGAs the Nd/Pr dip is the one
  chemotype-level exception still under test (M07: borrow the dip, not the SF). The tool ships no universal per-metal
  correction.
- **Cost.** Zero; already computed (`lab3skepB_closure.py`, `lab3_recompute_skA.py`).
- **Prior.** A 0.80 · B 0.85 for the scoped line. About 0.3 for the hypothesis as written.
- **Relations.**
  - **M01:** untouched; same-campaign information is exactly what does transfer.
  - **M09, M14, M18, M33:** none.
  - **M25:** the closure does not cover the Gd term, whose anchor is M25's job.
  - **M34:** the reopen bar is the detectable size under 13 clusters.
  - **M35:** a closure-ledger line; cross-reference M07 and the B2 line.

### 7. shift-3: Ce/Pr acid sign in DGA-class extractants, sign only (RIDER → M05/M22 supply ledger, M33)

- **Hypothesis (narrowed by both skeptics).** In DGA-class neutral O-donor extractants in mineral acid (HNO₃, HCl), with Ce
  and Pr inside log D [−2, 2.5], raising acid concentration makes the Ce/Pr strict-adjacent log SF more heavy-favouring
  (negative slope per decade of acid). This is a claim about the sign only. There is no size claim, no La/Ce or Pr/Nd
  claim and no mechanism claim.
- **Mechanism: undetermined.**
  - The solvation-number argument as stated predicts the opposite sign. It gives the observed sign only if
    g = d log[L]_eff / d log[acid] > 0, for example through acid-driven DGA aggregation.
  - Level fanning explains part of the effect. In window and clean, the level model predicts −0.21 of the observed −0.34.
    The Ce/Pr residual is −0.13 with a linear level model (4 of 6 negative) and −0.02 with a quadratic one (3 of 6).
  - Against a pure Ce artefact: the Ce-free La/Pr slope is negative in 10 of 10 publications and 6 of 6 laboratories, and
    the Ce deviation slope has no sign (6 of 10, p 0.57) (A).
  - For a partly Ce-specific deficit: La/Ce is mixed inside DGA (pub_19783c039b +0.22, +0.26) (B).
  - Consistent with M28's closure, which concerned the between-cell Ce deviation, not this within-series pair slope.
  - recipe-4's skeptic B found the same cross-publication slope sign independently: Ce/Pr 0 positive / 6 negative.
- **Quick-check result (EXPLORATORY).**
  - The Ce/Pr acid slope is negative in 11 of 11 publications (p 0.001; mean −0.32 per decade, sd 0.38). HNO₃ only: 0 of
    9 positive.
  - The other pairs as quoted: La/Ce 8 of 10 negative (corrected below), Pr/Nd 8 of 11, Yb/Lu 9 of 10 positive.
  - The predictive tilt TILT_CE at 0 reads: +0.017 (hindsight; C5-BTBP carries it).
- **Reproduction.**
  - Skeptic A reproduced it exactly. La/Ce is actually 7 negative, 1 exact zero (La and Ce bit-identical) and 2 positive.
  - The Ce/Pr sign survives every variant A ran:

    | variant | result |
    |---|---|
    | without C5 | 10 / 10 |
    | at least 3 acid levels | 8 / 8 |
    | extractant as the unit | 8 / 8 |
    | log D windows [−2, 2.5] / [−1.5, 2] / [−1, 1.5] | 10/10, 7/7, 7/7 |
    | leave 3 publications out | worst one-sided p 0.004 |
    | within-group acid permutation placebo | 0 of 500 permutations give all publications negative |
    | level-model residual | negative in 10 of 11 |

  - Laboratory level: 6 of 6 negative (one-sided p 0.016); 5 of 5 without C5 (0.031); 4 laboratories without JAEA and C5
    (0.0625). It is one pair chosen post hoc out of 13.
  - Skeptic B, clean subset (no C5, no DTPA publication pub_1605e436d7, no organic-acid media): 7 of 7 (p 0.016); 6 of 6
    in window (p 0.031); median −0.18 per decade.
- **What the skeptics found.**
  - The light-end framing is dead. Three independent laboratories with La and Nd (but no Ce/Pr) show positive La/Nd acid
    slopes (pub_b9dfc468cd, pub_f8a621ede4, pub_392f966ef2). Pr/Nd is about 50 % in window.
  - The non-DGA support is floor (C5-BTBP) or near zero. Two units get the sign from aqueous complexation (DTPA, malonic).
  - Whether the effect is a light-end property beyond level fanning is undecided.
  - The prospective test is unreachable. P4 has about 5–7 new laboratories in the nitrate / neutral-solvating regime, 2
    of them DGA. At n ≈ 6 the sign test needs 6 of 6 (power 0.53 at P(negative) = 0.9; 0.22–0.50 at a realistic 0.66–0.79).
  - The predictive leg rests on C5-BTBP and the DTPA publication.
- **Corrected design: a rider, not a slot.**
  1. For every new DGA-class mineral-acid series collected for M05/M22 for other reasons, record:
     - the frozen Ce/Pr and La/Pr slope signs;
     - the level-conditioned residual (per-metal acid slope against per-metal mean log D, fitted on Nd–Lu; linear primary,
       quadratic sensitivity);
     - the laboratory.
  2. **Precision eligibility.** A unit counts only if the Ce/Pr slope SE is ≤ 0.2 per decade (at least 3 levels, or a span
     of at least 0.5 decade), with Ce and Pr in window and flat-floor cells dropped.
  3. **Controls.**
     - acid permutation within series;
     - a level-matched control: other pairs in the bottom within-group level tercile (69 % negative on the bench);
     - a sign-reversal control from acidic-OP pH series (cation exchange). There, a floor or compression artefact predicts
       the opposite sign, and the solvation claim predicts no tilt;
     - Yb/Lu as the artefact exemplar.
  4. Exclude complexant-driven series (DTPA/EDTA holdback, organic-acid media) by frozen metadata.
  5. **Decision at laboratory level** with a precision-weighted random-effects mean:
     - support if the one-sided 95 % upper bound is < 0 and at least 70 % of units are negative;
     - kill if the pooled mean is ≥ 0 or at most 50 % are negative;
     - otherwise inconclusive;
     - fewer than 8 eligible laboratories: a descriptive case series.
  6. Drop TILT_CE. If it is ever used, it is one diagnostic row in M01's k = 0 ladder, without C5 and the DTPA publication,
     with a laboratory-median prior (about −0.20 per decade) taken leave-own-laboratory-out.
  7. Data-quality flags: pub_0fd6ab4e40 (La ≡ Ce at 2 M and 3 M; Pr = Nd at 3 M) and the pub_19783c039b copy pair (sign
     unchanged without it).
- **Kill criterion.** As in step 5. No size claim, ever.
- **Practical value.** Small and DGA-specific:
  - light-end (Ce/Pr) SFs screened at low acid understate the heavy-selectivity at high acid, by about 0.1–0.15 log per
    threefold increase in acid at fixed [L];
  - check at the operating D, not at fixed [L]: raising acid while lowering [L] partly cancels the effect;
  - nothing changes for Pr/Nd (the didymium pair) or the heavy pairs;
  - Ce is usually removed by oxidation in real circuits, which limits the process value.
- **Cost.** Minutes per new series (the scripts exist). No slot.
- **Prior.** A 0.35 · B 0.15 (passes its kill as written). About 0.6 that the DGA-only Ce/Pr sign holds in new
  mineral-acid data.
- **Relations.**
  - **M01:** TILT_CE is dropped (at most a k = 0 diagnostic row). The acid-concentration-only E-CT class is where this
    slope lives; Ce pairs carry 32.8 % of P1's SH − B1 gain, which this sign helps explain.
  - **M09:** the [−2, 2.5] window gates eligibility; floor controls.
  - **M14, M18:** none.
  - **M25:** the two data-quality flags.
  - **M33:** a supply-ledger rider, plus the acidic-OP sign-reversal control tag.
  - **M34:** laboratory unit and an exact one-sided sign test. JAEA is 5 of the 11 publications.
  - **M35:** no decision slot. Consistent with the M28 Ce-acid closure. Related to M07, which has the same
    frozen-sign-plus-external-test structure.

---

## Rejected ideas (both skeptics), with the fragment worth keeping

Ordered by the value of what they leave behind.

### shift-1: acid-distance rule for zero-read copying (prior A 0.15 · B 0.20)

- **Claim.** Copy your own series at 0 reads only within about a factor of 2 in [acid]; beyond that, read and use SH.
- **Fatal flaws.**
  - **It depends on one publication.** Without pub_19783c039b the within-publication rank statistic falls from +0.259
    (p 0.04) to +0.151 (p 0.19). That publication holds a near-copy pair (44012938119033a5 at 4.0 M, 5169c377777d282b at
    5.0 M; 13 of 14 log D bit-identical) that the V2 lists miss.
  - **"Distance" is range position.** Permuting within publication × position (inside / above / below the siblings' acid
    range) leaves +0.066 (p 0.37). Interior targets alone give −0.079. The below-range flag alone gives +0.302
    (p 0.021). In-corpus the rule is "do not copy down to a lower acidity than measured".
  - **The near-side "skip the measurement" claim fails the cheapest alternative.** COPY@0 0.655 against SH@2 0.664;
    equal-weight relative SSE −0.71, with 2 of 8 publications positive.
  - **The external test is close to a tautology.** With 3-level families, rank distance is just "ends vs middle". Any
    monotone acid drift passes: kill (i) fires only 15 % of the time.
  - **Covered already.** M01 carries the COPY arm and a nearest-condition comparator. M03 carries the
    interpolation-vs-extrapolation stratum.
- **Salvage.**
  - One descriptive E-CT row in M01: ACOPY@0, COPY@0, AINT@0, SH@2, SH@4, B1@4 and OFF by |Δlog₁₀ acid| bin × interior /
    extrapolation × below / above; with and without pub_19783c039b and C5-BTBP; both distance metrics reported (the
    dx_near / dx_mean fork).
  - The better-founded reframing: in HNO₃, the value of the first 2 reads grows with distance (+0.512, p 0.008).
  - Closure line (below). The two duplicate pairs go to M25.
- **Relations.** M01 descriptive row (via M03's stratum); M25 duplicates; M34 external families need at least 4–5 levels and
  laboratory units; M09: below-range targets sit near the floor. M14, M18, M33, M35: none beyond the ledger line.

### recipe-1: reads-equivalent credit of an own prior series (prior A 0.15 · B 0.12)

- **Claim.** 2 spread reads plus the own template ≈ 4–5 no-prior reads; literature series give no credit.
- **Fatal flaws.**
  - **One publication.** Without pub_b0b3296ec5, SH_own@2 − D7@4 falls from +0.052 to −0.001, and COPY@0 − D7@3 from
    +0.027 to −0.086. Family level: 10 up / 11 down, median −0.009.
  - **The weakest comparator.** HULL is the weakest label-free no-prior policy. Against the fixed ladder the gain is
    +0.014; against DOPT_Q-nested, +0.031. COPY@0 loses to both at 3 reads. The credit is about 2 reads, not 2.5.
  - **"Literature gives no credit" is an artefact of w = n_pool/(n_pool + K) ≈ 1.** With a single-series weight the
    literature arm gives +0.028 (9 up / 4 down); the same-chemotype pool gives +0.030 [+0.002, +0.074].
  - **The practical claim contradicts the strata.** With one own series (n_own = 1) the credit is about 1 read.
    Diluent-only cells show none. Acid-type changes split 1 to 1.
  - **The kill rule cannot decide.** It falsely downgrades a true effect 48–65 % of the time at 11 families. Supply is
    about 4 families.
  - **Covered.** It is M02 / M01's read ladder reframed.
- **Salvage.**
  - One descriptive E-REC table in M01's read ladder, by stratum (n_own = 1, n_own ≥ 2, giant cluster, the rest), with a
    column for m = 1 / 2 random own siblings and with and without pub_b0b3296ec5.
  - The comparator is the best label-free no-prior policy, named and frozen.
  - The recipe by sibling count: one prior series + 2 end reads ≈ 3 no-prior reads; at least 2 dense same-campaign series
    + 2 reads ≈ 4 (DGA, mostly one study).
  - An analytical-artefact control: remove the template's per-element constant and report the share of credit that
    survives.
- **Relations.** M01 annex table; M18 supplies the comparator policy; M25 gets the pub_b0b3296ec5 provenance gate and
  bit-identity checks; M33/M05 supply about 4 families; M34 family-level operating characteristics. M09, M14, M35: none.

### recipe-3: sign-first triage from an own prior series (prior A 0.25 · B 0.10)

- **Claim.** Trust the old series' sign of a pair at a new condition when |old SF| ≥ 0.1; re-measure only for magnitude.
- **Fatal flaws.**
  - **Comparators on different rows.** On the identical covered rows, always-heavier scores 0.877 (not 0.814), D7@2 0.918
    and D7@4 0.934. At matched coverage COPY scores 0.922, D7@4 0.919, B1@4 0.933 and SH_own@2 0.949. So COPY at 0 reads
    is worth about 4 no-prior reads for selective direction: recipe-1's credit on another endpoint.
  - **The literature leg is dead.** 0.922 against always-heavier 0.934 on identical rows; 92 % of those rows are sc009.
  - **"≥ 90 % of all pairs" holds only pooled, in DGA, with many siblings.** Single sibling 0.872; outside the giant
    cluster 0.812; acid changes 0.785–0.879. 8 of 19 families are below 0.90.
  - **The kill fires in sample**, and with 11 families it would pass a true 0.92 rule only 9–22 % of the time.
  - **Practical value overstated.** ICP-MS saves nothing; radiotracers save about 33–43 %, not half. Only 7 % of cells are
    fully covered, and heavy pairs are essentially never covered.
  - **Covered by M14 + M01 + M05.**
- **Salvage.**
  - One descriptive "own prior" row in M14 (E-DEC): risk–coverage curves and AURC for COPY_own, SH_own@2 and SH_own@4,
    against chemotype × pair-type majority (leave-own-publication-out), G14, D7@2/4 and B1@4, at matched coverage.
  - The three-way rule (lighter / heavier / inseparable) is primary. Opposite-direction call rates: COPY 1.1 %, SH_own@2
    0.9 %, D7@2 2.3 %.
  - Stratify by condition-change type and DGA vs non-DGA (non-DGA: COPY 0.911 against always-heavier 0.589).
  - Exclude quantised sources (pub_a49146c8fd, whose log SFs are logs of small-integer D ratios).
  - Closure line (below).
- **Relations.** M14 host row; M01 provides COPY/SH; M25 gets the quantisation flag; M34 operating characteristics. M09,
  M18, M33, M35: none beyond the ledger line.

### recipe-2: fixed training-free metal ladder (prior A 0.20 · B 0.15)

- **Claim.** A fixed ladder (La,Ho → La,Nd,Ho → La,Nd,Er,Lu → La,Nd,Gd,Er,Lu) matches DOPT_Q at 3–5 reads and beats it at
  2.
- **Fatal flaws.**
  - **Claim (c) is false on the idea's own data.** With an own prior, DOPT_Q beats the ladder at 3 reads (+0.019
    [+0.002, +0.045]), and HULL beats it at 5 reads.
  - **Claim (a) holds only in sample.** Selected out of sample (LOPO/LOCO), the ladder scores 0.533–0.544 against 0.552 at
    5 reads. The 2-read margin is 87 % giant cluster.
  - **Its edge over HULL is two publications plus a continuation artefact.** Under linear extrapolation La,Ho loses to
    La,Lu (−0.026 at 2 reads). For acidic organophosphorus extractants (adjacent SF ≈ 2.5 through the heavy end), La,Ho
    would be wrong, and that is the supply.
  - **Short series fail.** On series of 8 metals or fewer (46 % of cells) the ladder is just the HULL fallback and loses to
    DOPT_Q by 0.038–0.076, which meets its own kill.
  - **It is not nested.** Reaching the 4-read rung costs 5 experiments.
  - **The kills cannot fire**, and G80 ≈ 200 units.
  - **The practical advice is wrong.** La-140 and Ho-166 are short-lived activation products; under ICP-MS every metal is
    read anyway.
- **Salvage.**
  - One descriptive row in M18 (host of M17): the nested static order La → Ho → Nd → Lu → Gd (0.343 / 0.432 / 0.501 / 0.552
    in sample), LOCO-selected static sets, HULL, DOPT_Q-nested and random at 2–5 reads, by series length.
  - Ladder − HULL as the primary contrast.
  - A chemistry-class guard: La,Lu (HULL) for monotone heavy-selective cation exchangers.
  - Closure line: "spread first, never one metal" is definitional (HULL − random +0.197 at 2 reads; one read = zero reads).
- **Relations.** M18 host row; the own-prior clause belongs in M01's ladder; M33 supplies acidic-OP series for the guard;
  M34 power; M35 closure line. M09, M14, M25: none.

### recipe-4: ICP-MS condition skipping with a shrunk acid trend (prior A 0.20 · B 0.10)

- **Claim.** Average the campaign's acid-only series; a half-shrunk, clamped log-acid line beats the plain average inside the
  range by at least +0.03.
- **Fatal flaws.**
  - **Near-copy leakage carries the headline.** Two cross-acid copy pairs (pub_19783c039b, pub_c8195b4ca7) make two
    targets "interior". De-duplicated, the gain is +0.024 [+0.004, +0.065], below the idea's own +0.03.
  - **λ = 0.5 was chosen after looking.** Nested choice gives +0.021 to +0.027; 4 of 8 publications improve;
    pub_19783c039b carries 73 % of the gain. RMSE moves only from 0.114 to 0.109 log.
  - **It ties the bracketing-pair mean** (PAIRMEAN 0.535 = LINHALF 0.535).
  - **Ce/Pr carries 52 % of the gain**, the closed Ce-acid lead.
  - **The 3-level recipe was never tested, and it fails.** LINHALF3 ≈ MEAN3; pooled R² 0.07–0.17; RMSE 0.15 log, larger
    than the within-campaign acid SD (0.116). Piecewise-linear interpolation through the 3 levels wins wherever there is a
    trend.
  - **Underpowered**: about 50 campaigns for 2/3 power. The named supply does not fit (KU Leuven has no acid variation;
    Chiba is pH-controlled).
  - **Covered** by M03, H-COND and M28.
- **Salvage.**
  - One 0-read acid-trend arm in M03 (LINHALF, LINCLIP), with the claim statistic LINHALF − max(MEAN, PAIRMEAN).
  - A cross-publication pair-type slope (POOLPAIR) as the more interesting hypothesis. Slope signs agree by pair: Ce/Pr 0
    positive / 6 negative, Yb/Lu 6 / 0, Ho/Er 1 / 6, La/Ce 1 / 5.
  - Exploratory: at the range edge, the clamped trend value beats the last measured level (+0.103, LOPO [+0.045, +0.139]).
  - Closure lines (below). A duplicate rule for the cross-acid copies goes to M25.
- **Relations.** M03 arm inside M01; M25 duplicate rule; M34 power; M35 closure lines. M09, M14, M18, M33: none.

### lab-4: campaign, not laboratory, as the confirmation unit (prior A 0.15 · B 0.08)

- **Claim.** Same-laboratory papers are nearly independent (per-metal sharing about 0.05), so publication × chemotype units
  hold type-I error.
- **Fatal flaws.**
  - **Wrong quantity.** Type-I error depends on contrast-level correlation, not on per-metal sharing. Same-laboratory rank
    correlations of per-publication log SSE ratios: S02 − B1 +0.57 (p 0.009), S05 − B1 +0.46, S06 − B1 +0.61. Size
    crosses 0.07 at a contrast-level correlation of about 0.10–0.15.
  - **The premise is false for the contrast that matters.** The Gd chord is shared across a laboratory's papers: +0.37 to
    +0.47 on publication means (true dip +0.47, p 0.033), and +0.73 cell-level for the same extractant.
  - **The steps cannot decide.** Step 1 is circular. Step 2 fires with probability 0.35–0.39 at a true correlation of 0.3,
    and flips with trivial unit choices (B1 − B0: −0.107 with publication units against +0.320 with publication ×
    chemotype units, no filter).
  - **It contradicts itself.** Campaign-level sharing implies that publication × chemotype units inside one publication
    are dependent (other-chemotype same-publication sharing +0.19 to +0.77).
  - **Counterexample.** JAEA republished series across publications (pub_af0c74798c, pub_d68ed8c8cf).
  - **Covered** by M34 / M36. Its tolerance (0.07) is looser than M34's sealed rule.
- **Salvage.**
  - One line in M34: for Gd-amplitude contrasts, the primary unit is the laboratory (aggregated across chemotypes), from
    P4's frozen lab_id; publication units are a sensitivity.
  - Repair the laboratory key: drop common-surname links (liu, zhang, zhou, xu), handle Family–Given order and links
    through a single principal investigator.
  - Copy detection at arrival.
  - Chemist message, two parts: (a) the mid-series campaign noise fingerprint does not persist, so any in-run reference is
    re-measured in each campaign; (b) a ligand family's Gd break does persist across a laboratory's papers, so several
    same-ligand papers from one group are one unit of evidence for a Gd claim.
- **Relations.** M34 host line; M33 frozen laboratory key; M25 cross-publication copies; M35 tolerance. M01: the noise
  message goes to M11. M09, M14, M18: none.

### lab-2: in-run TODGA reference protocol (prior A 0.15 · B 0.08)

- **Claim.** A full TODGA series in every campaign carries a campaign fingerprint f that improves the new ligand's 4-read
  prediction by at least 3 %; analytical arms tell artefact from chemistry.
- **Fatal flaws.**
  - **The protocol makes its own endpoint pointless.** An equimolar 14-Ln feed by ICP gives the new ligand's full series
    in the same run. The correction is tiny anyway: applied |w·f| has a median of 0.008–0.009 log D.
  - **The evidence is one TODGA series in one homologous DGA screen.** pub_126bbe7976 carries 79–86 % of the gain. TODGA
    ranks 1st of 15 candidate references there, and the other 14 average +0.9 %. All 46 targets sit in one source cluster.
  - **The own estimator fails its bar.** FP improves 3 of 5 publications. Without one copy target it gives +2.5 %, below
    the 3 % claimed.
  - **Any second ligand does as well.** MX (+5.9 %) and OFFREF (+5.8 %) match the own TODGA. Same-medium TODGA equals all
    own TODGA, and other-medium beats same-medium where both exist. The corpus says "campaign", not "same run".
  - **The kills are not real.** P(kill | null) is 0.50 at any G. The attribution rule sits below duplicate noise.
  - **f is confounded with TODGA's own condition dependence.**
  - **It needs at least 8 collaborating laboratories; the user has none.** Covered by M11 / U4 / M12.
- **Salvage (into M11 / U4).**
  - The reference-series arm: in-run against previous-campaign against medium-matched literature TODGA, and a day split
    (day-1 ligand calibrated with day-2 TODGA).
  - One fixed-condition TODGA tube per laboratory as a ring standard; the estimand is M12's V_x/V_w.
  - Noise-calibrated attribution (χ² against duplicate noise).
  - ICP-MS/MS O₂ mass-shift or ICP-OES as the second method for the monoisotopic LnO⁺ victims (Pr-141, Tb-159, Ho-165,
    Tm-169).
  - A low-D arm for the ε(1 + 1/D) signature; the reference also run in the new ligand's own diluent.
- **Relations.** M11/M12 host; M01: MX (any other own-campaign ligand) is the comparator; M09: low-D arm; M25: three
  near-copies; M34: laboratory-unit operating characteristics. M14, M18, M33, M35: none.

### shift-2: bracketed per-metal slope model (prior A 0.10 · B 0.08)

- **Claim.** A James–Stein-shrunk per-metal log D vs log[acid] (and log[L]) slope model, gated to bracketed targets, beats
  the plain template at 0–2 reads.
- **Fatal flaws.**
  - **The frozen arm does nothing.** AEBG@0 − ACOPY@0 is +0.010 R². It moves 9 of 57 cells, and 0.0088 of the gain is
    pub_19783c039b.
  - **It ties plain bracketing interpolation** (AINT 0.515 against 0.515). The label-free λ loses to a constant 0.25.
  - **It fails its own 4-read rule** (−4.1 % relative SSE against SH).
  - **The factorial-grid claim cannot come from the frozen rule.** Pyran-DGA has λ = 0. C5-BTBP reverses without the
    corner cell c950dc. The only other two-coordinate grid fails.
  - **Power.** dz 0.41 needs about 37 bracketed families.
  - **Covered** by M03, M02 and H-COND.
- **Salvage.**
  - AINT and "mean of the 2 acid-nearest siblings" as M03 arms.
  - Closure lines (below).
  - The bracketing advice (measure reference series that bracket the operating acidity, especially on the low side) goes
    into M01's practical reading.
  - The factorial grid becomes a descriptive case series under M05.
  - A provenance check of pub_19783c039b (a parallel series at non-round acid values looks like table plus figure; cell
    652810 at 3.99 M nearly copies 440129 at 4 M and sits in SH's pool) goes to M25.
- **Relations.** M03 arm inside M01; M25 provenance check; M34 power; M35 closure lines. M09, M14, M18, M33: none.

---

## What round 2 adds to each round-1 host

| host | additions from round 2 |
|---|---|
| **M01 (E-CT)** | confirm-4's repaired kill (iv): laboratory unit, fixed sequence, summed-SSE family statistic, pair-level T0, SH − OFF_XEXT replacing kill (i), world-specific power. lab-1's POPX0 annex arm with E_PBC/E_PCM placebos. Descriptive rows: shift-1 (distance × range position), recipe-1 (E-REC credit by sibling count against the best no-prior policy). M03 arms: AINT, PAIRMEAN, LINHALF/LINCLIP, POOLPAIR (shift-2, recipe-4). Practical reading: bracket the operating acidity from below; read ≥ 2 metals and use SH rather than copy |
| **M05 (external leg)** | confirm-4 is its pre-registration annex. Riders: shift-3's Ce/Pr sign, a lab-1 unit count, the factorial-grid case series (shift-2) |
| **M09 (window)** | no new claim. Notes: lab-1's low-D sharing (r 0.53 at log D < −2) and shift-1's below-range targets both sit at the lower edge, which supports reading the window as a gross-error flag tied to the floor |
| **M14 (E-DEC)** | recipe-3's "own prior" row: risk–coverage for COPY_own, SH_own@2, SH_own@4 at matched coverage; three-way rule primary; non-DGA stratum |
| **M18 (metal ladder)** | recipe-2's nested static row (La → Ho → Nd → Lu → Gd), LOCO-selected static sets, by series length; chemistry-class guard |
| **M25 (source recovery)** | the duplicate and provenance list below; provenance checks of pub_19783c039b and pub_b0b3296ec5; pub_84113c17ad (F6) matters for any Gd stratum |
| **M33 (supply gate)** | confirm-1's supply-audit kill and eligibility; confirm-2's mechanism-class stratum and per-family planning values from primary tables; confirm-3's chord interval; shift-3's rider and acidic-OP sign-reversal control |
| **M34 (inference)** | confirm-1's minimum-G table and two-stage (chemotype → laboratory) null worlds; laboratory units for all Gd contrasts (lab-4 salvage); fixed sequence instead of Holm at small G; operating characteristics for every kill |
| **M35 (decision)** | confirm-1's exclusion list; three-way decisions with inconclusive zones; non-rejection never switches the deployed model; the closure lines below |

---

## Closure ledger additions (one line each, no slot)

| line | closes | evidence (EXPLORATORY) |
|---|---|---|
| E-A per-metal deviation priors from other publications, unconditional or level-matched, are worse than B1 (scoped to E-A; same chemotype untestable; M07 carved out) | lab-3 | BPG MG_tr −4.8 % [−8.5, −1.2]; MG_LM10_tr −5.8 % [−10.1, −1.5]; shrinkage monotone toward 0 |
| Zero-read copying of your own series is never the recommendation: worse than SH with 2 reads at every acid distance, worst when copying below the measured acid range | shift-1 | near side COPY@0 0.655 vs SH@2 0.664 (2 of 8 publications); below-range flag +0.30 |
| Free per-metal acid-slope regression, linear extrapolation in log acid, and slope physics at 4 reads are retired | shift-2 | AREG@0 −0.279; AINTX@0 −0.32; AEBGSH@4 −4.1 % vs SH |
| Never extrapolate an unshrunk acid trend beyond the measured range | recipe-4 | −0.64 R², 6 of 8 publications lose (64 % of the loss from one 2-level publication) |
| A literature series of the same extractant gives no direction information beyond "the heavier member extracts more" | recipe-3 | 0.922 vs always-heavier 0.934 on identical covered rows |
| "Spread the reads, never one metal" is definitional, not a finding | recipe-2 | one read = zero reads by construction; HULL − random +0.197 at 2 reads |
| Publication units are not valid for Gd-amplitude contrasts | lab-4 | Gd chord shared across a laboratory's papers +0.37 to +0.47 |
| An in-run TODGA reference is not needed for prediction; any second own-campaign ligand carries the same signal | lab-2 | MX +5.9 % vs own TODGA +5.3 %; same-medium = all-medium own TODGA |

---

## Duplicate and provenance candidates for M25 (V2 addendum)

Found by the round-2 skeptics. Detection used label values (bit-identity, constant offsets), so under IDEAS_RANKED rule 7
each item is a **sensitivity and a provenance question**, not an edit to V1, until archive evidence or the source settles
it (as P5 did for the V2 flags).

| item | where | finding | found by |
|---|---|---|---|
| 1 | pub_19783c039b | 44012938119033a5 (4.0 M) / 5169c377777d282b (5.0 M): 13 of 14 log D bit-identical (only Tb differs); round-3 flag-only, not in V2 | shift-1 A, recipe-4 A, P1 check |
| 2 | pub_19783c039b | a parallel series at non-round acid values (0.4986 … 5.9942 M) with a different diluent label, likely table + figure of the same data; cell 652810 (3.99 M) nearly copies 440129 (4 M), sd 0.04, and sits in SH's pool | shift-2 A |
| 3 | pub_c8195b4ca7 | d4c1e23a091d0960 / e4ee526bda3920c2: 7 of 14 log D bit-identical at a different acid; heavily quantised values (5 metals at exactly 1.134 in one cell); acid-only siblings near-parallel on 6 metals (sd 0.006–0.02) | shift-1 A, recipe-1 A, recipe-4 A, shift-2 A |
| 4 | pub_b9dfc468cd | TDDGA 259e68bb986bb484 ≡ TDdDGA 26cbed1ed70886fe, bit-identical at 8 metals; both CLEAN C3 targets and each other's donor | lab-1 A, lab-2 A |
| 5 | pub_5a68dc5665 | DMDPDGA 37b0cdc532627e1d ≈ TODGA fadca46ff2b140c2 within 0.01 at 8 metals, same level; metal-concentration duplicates (0.0006 / 0.0007 mM) | lab-1 A and B, lab-2 A |
| 6 | pub_4cba27f5f3 | DPiperidineDGA 2c83afeaea6f67e0 ≈ TEtOMeDGA b1d9860b27f6967a within 0.04 at 14 metals | lab-1 A |
| 7 | pub_af0c74798c / pub_d68ed8c8cf (JAEA) | cross-publication copies: 4 series agree within 0.010 over 6–8 metals, two under different extractant labels ("TDdDGA" 4fcbabe856178b53 at 0.5 M ≡ TODGA); the 1 M TODGA is republished | lab-1 A, lab-2 A, lab-4 A |
| 8 | pub_6a0ef8528c (C5-BTBP), pub_74428aeab2, pub_723ceb6c2c | Gd–Lu segments of 3 C5-BTBP cells are constant-shift copies (−5.37, sd 0.012–0.026) of DMDPDGA 1b71b86063cd1faf; DMDPDGA also matches C5-BTBP 326289072f176f80 at +5.37 (sd 0.011); the pub_723ceb6c2c C5-BTBP cell is a +1.95-shift copy of TODGA 354fe634113a078e (sd 0.021) | confirm-2 A, lab-1 A |
| 9 | pub_e9799b75f3 | sc068 and sc070 are constant-shift copies on Gd–Lu; D reported to one decimal (identical chords of sc059 and sc070 are arithmetic coincidence, ±0.05 log quantisation) | confirm-2 A, confirm-1 A, confirm-3 A |
| 10 | pub_0fd6ab4e40 | La ≡ Ce at 2 M and 3 M; Pr = Nd at 3 M (possible digitising artefact) | shift-3 A |
| 11 | pub_a49146c8fd | log SFs are logs of small-integer D ratios (0.301, 0.176, 0.079, …) | recipe-3 B |
| 12 | pub_84113c17ad | component-identity error (P5 F6): CDTA/HEDTA recorded as the extractant, title names TEHDGA + HDEHP. It drives the deployed-row loss of the heavy-selective Gd amplitude outside C5-BTBP (this note's check) | P5; this note |

Two publications carry several condition-transfer leads and deserve a provenance check before any E-CT number is relied on:
**pub_19783c039b** (all of P1's SH − OFF; 73 % of recipe-4's gain; the favourable contrasts of shift-1 and shift-2) and
**pub_b0b3296ec5** (41 % of P1's SH − B1 gain; all of recipe-1's credit; the only acid-type publication).

---

## Reconciliation check done for this note (EXPLORATORY)

**Why.** For confirm-2 the skeptics disagreed on whether a heavy-selective (HS) Gd amplitude helps on deployed rows.
- Skeptic A: it transfers on Gd-chord cells (LOPO −59 % Gd-cell SSE against β).
- Skeptic B: in the deployed k = 3 pipeline it gains only through C5-BTBP.

**What was run.** `ideas2/scripts/addendum_gd_reconcile.py` (log `addendum_gd_reconcile.log` next to it). It is one
process of a few seconds, writes nothing but stdout, and uses only skeptic B's own out-of-fold rows
(`confirm2skB_out/O_rows_{BPG,BP}.parquet`: arms B1, POOL with the pooled refit, STRAT with the stratum amplitudes)
aligned row by row with `round4_20261022/predictions/REF_{BPG,BP}.parquet`. Alignment was asserted on seed, cell, y and B1.
The rows were split by:
- Gd read vs Gd hidden (from REF read_set, label-free);
- the eligibility confirm-2 freezes: the cell reports Eu, Gd and Tb (from the pair list), and is not the
  component-identity / aqueous-complexant publication pub_84113c17ad;
- with and without C5-BTBP.

**Results.** Relative SSE of STRAT against B1 on heavy-selective rows; BPG / BP; "pubs" counts publications where STRAT is
better / worse.

| HS rows defined by | subset | rows/seed | B1 → STRAT R² (BPG) | STRAT vs B1 | pubs + / − (BPG) |
|---|---|---|---|---|---|
| reads-only slope (B's stratum) | all | 360 | 0.356 → 0.381 | +3.9 % / +3.7 % | 8 / 6 |
| | without C5 | 212 | 0.496 → 0.498 | +0.4 % / +0.1 % | 7 / 6 |
| | without C5 and pub_84113c17ad (= all eligible) | 202 | 0.518 → 0.539 | **+4.4 % / +4.0 %** | 7 / 5 |
| | eligible, Gd hidden | 161–168 | 0.512 → 0.537 | +5.2 % / +5.6 % | 6 / 4 |
| | eligible, Gd read | 33–40 | 0.630 → 0.637 | +1.5 % / −3.6 % | 3 / 4 |
| | eligible, Eu/Gd + Gd/Tb pairs only | 35 | 0.617 → 0.697 | +21.9 % / +21.4 % | 7 / 5 |
| full-series slope | without C5 (pub_84113c17ad is light-selective here) | 233 | 0.543 → 0.565 | +4.7 % / +4.3 % | 7 / 3 |
| | without C5, Gd hidden | 190–197 | 0.515 → 0.534 | +4.3 % / +4.8 % | 6 / 3 |

Per publication (eligible, reads-defined, no C5-BTBP; SSE gain per seed, BPG):

| side | publications |
|---|---|
| gains | pub_1845fd6492 +0.137 (triazinyl, chord −0.39); pub_40c355ae86 +0.126 (dihexyl hydrogen phosphate, the single acidic-OP cell, −0.37); pub_8415538291 +0.062; pub_401a06b4b7 +0.052; pub_a49146c8fd +0.048; pub_723ceb6c2c +0.016; pub_af7a1cd919 +0.015 |
| losses | pub_3f3b6220e0 −0.104; pub_74428aeab2 −0.082; pub_b8175dc3a1 −0.062; pub_e9799b75f3 −0.041; pub_4fb6e0c041 −0.020 (amide and CMPO units with shallow chords) |

**Reading.**
- Skeptic B's "the gain is C5-BTBP alone" comes mostly from pub_84113c17ad. That publication is a misattributed
  TEHDGA + HDEHP system with aqueous complexants. confirm-2's own eligibility rule excludes it, and its full-series slope
  makes it light-selective.
- On eligible rows outside C5-BTBP, the stratum amplitude helps by about 4 % relative SSE on heavy-selective rows, mostly
  on Gd-hidden rows. That is about 0.21 SSE per seed, about 0.4 % of B1's total SSE, or about +0.002 R² pooled: far below
  any MDE.
- The 7 / 5 publication split carries no inference, and the gains follow chord depth and mechanism (triazinyl,
  acidic phosphate), not the heavy-selective label.
- This supports the fixes both skeptics converged on: a mechanism-defined stratum, and a depth- or family-scaled
  amplitude. It does not change confirm-2's fate: its decisive leg needs new laboratories.
- EXPLORATORY: all of this is on the 229 seen cells and 5 dev seeds.

---

## Process note

- **Files written** (all new; no existing file was modified):
  - this file;
  - `ideas2/scripts/addendum_gd_reconcile.py` and its log `ideas2/scripts/addendum_gd_reconcile.log`.
- **Inputs read:** `EVIDENCE_BASE.md`, `IDEAS_RANKED.md`, P1–P5 and `PROBE_CHECKS.md`, the confirm-2 skeptic B log and
  parquets, and the round-4 REF parquets. Python ran with `-B`, one process at a time, a few seconds each. No git, no
  downloads, no logins, no web access.
- **One slip against the read-scope rule.** One Bash output was too large and the harness persisted it to the
  `.claude\projects\...\tool-results` folder. I read that persisted copy once. It held only my own output: the first
  lines of `PROBE_CHECKS.md`.
- **Skeptic slips, as they reported them:**
  - shift-1 skeptic B wrote a patch helper in the session scratchpad, outside `ideas2/`;
  - lab-1 skeptic A and lab-2 skeptic A wrote log copies to the scratchpad;
  - the confirm-2 skeptic B WebFetch auto-cached a scanned PDF under the `.claude` project folder (not opened).

  None of these touched the repository's existing files.
- Every number above was computed on already-seen data and is **EXPLORATORY**.
