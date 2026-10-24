# Practical guidance for chemists (gen21 round 5)

Each line carries its evidence label:
- **ESTABLISHED:** round-4 development result, replicated.
- **EXPLORATORY:** seen on this corpus, not confirmed.
- **HYPOTHESIS:** not tested on this corpus; its registered check waits for new data.

Each line also carries its **cost scenario**. ICP-MS gives all 14 lanthanides from one experiment, so there the cost
unit is the extractant × condition, and "more metals" is free once the experiment is run. Metal-count economics apply
to radiotracer work, subset feeds and literature mining.

Numbers are R² of strict-adjacent log SF unless stated. They come from E-A's 229 cells in 39 publications, or from a
subset of them that depends on the item:

| items | rows |
|---|---|
| 1, 2, 5, 6, 7 | the E-A rows (229 cells, 39 publications) |
| 3 | E-CT PRIMARY (99 cells, 12 publications, 14 publication × chemotype units) |
| 4 | E-LAB PRIMARY (92 cells, 16 publications, 9 conservative laboratories) |
| 7 (own-prior row) | E-REC PRIMARY_OWN (112 cells, 16 publications) |
| 8 | the sc009 Pr/Nd rows (46 cells, 10 publications); its all-extractant comparison uses 214 cells in 38 publications, without C5-BTBP |
| 10 | the wider pairs (225 cells, 38 publications) |

Every in-corpus number is seen, not confirmed (`REPORT.md`).

## Predicting separation factors

1. **Four measured metals and no prior series of this extractant:** use radius interpolation plus the Gd-dip
   correction. That is B1 (R² 0.452), or the fixed amplitude -0.072, which scores the same within 0.001 (R² 0.451).
   - The fixed value is not fold-independent: -0.072 was estimated on 224 of the 229 cells it is scored on.
   - No tested chemistry, electronic, donor-class, encoder or An(III) model improves on B1 (round 4; round 5 registered
     no new model).
   - **ESTABLISHED.** Cost: any scenario.

2. **More metals help, but it is a different budget.** One more well-placed read adds about +0.045 R² on the same
   queries (paired; 9 of 13 data clusters gain, 2 lose, 2 tie).
   - With 5 metals, the like-for-like fixed cohort sits at about **0.48**. The mixed row over all 229 cells sits at about
     0.50, because it also includes the 51 cells that cannot take a legal 5th read. Those cells score 0.60 with 4 reads.
   - In the cohort that can take 7 legal reads (133 cells), R² rises from 0.43 at 4 reads to 0.48, 0.53 and 0.56 at 5,
     6 and 7.
   - Where you place the 5th read matters little: the envelope over sensible label-free policies is about 0.009, and no
     policy is better everywhere.
   - For Yb/Lu, read Lu or Tm.
   - **EXPLORATORY.** Cost: radiotracer and literature scenarios; with ICP-MS, measure everything.

3. **You have your own earlier series of this extractant** (same study or campaign; the laboratory key is still
   PROVISIONAL, U6 open):
   - **Do this:** shrink your own series toward B1 rather than copy it, and read 3-4 metals at the new condition. With
     4 reads this gave R² 0.70 against 0.53 for B1 on the same rows (+0.17, 14 of 14 publication × chemotype units).
   - **Averaging:** one shrunk series already beats B1 (+12 % relative SSE), and averaging several helps more.
   - **Do not:**
     - copy without reads (0.48, worse than B1);
     - use one raw series in a dense grid;
     - copy below the measured acid range.
   - **Acidity:** bracket the operating acidity, from below if possible.
   - **Other publications' curves** of the same extractant do not substitute.
   - **Weighting:** read-match weighting of the siblings looked better in-corpus (+0.02), but it is fragile, so the
     deployed rule stays the uniform shrunk mean.
   - **EXPLORATORY.** Confirmation needs at least 7 new laboratory chemotype units. Cost: one experiment per new
     condition (ICP-MS).

4. **A new extractant inside a campaign that has calibrated others:** borrowing the campaign's own level-matched donors
   gained about +8 % relative SSE over B1 (9 of 9 conservative laboratories; PROVISIONAL key). Most of that comes from
   using the campaign's donors at all: random own-campaign donors already give about +6 %.
   - **EXPLORATORY**, the best of about 25 arms on seen rows. Expect regression; it is also fragile (about +0.6 % when
     the two most helpful publications are left out).
   - Cost: campaign data in hand.

## Trusting a prediction

5. **Trust window:** when B1 predicts an adjacent-pair SF inside the level window, large errors are rarer.
   - **All adjacent pairs:** inside the window, about 1 pair in 90 is off by more than 0.5 log; outside, about 1 in 23
     (ratio 3.8, interval about 1.9-8.0).
   - **Light pairs are worse on both sides:** La/Ce about 1 in 45 inside vs about 1 in 6 outside (2.2 % / 15.9 %);
     Ce/Pr about 1 in 29 vs about 1 in 6 (3.5 % / 15.8 %).
   - The post hoc window "both predicted D between 0.01 and 100" behaves similarly (ratio 4.9). Re-measure, or move D
     into range, before trusting light-pair SFs outside it.
   - This is a tail-risk flag, not better accuracy. Inside R² is 0.52 against 0.50 for random abstention of the same
     size, and without the C5-BTBP publication on BP the flag's lower bound falls below 1.
   - **EXPLORATORY.** Cost: none (a reporting flag).

6. **Detection limits:** report "< DL" with the per-metal limit, and never mix clamped values with raw sub-limit reads
   in one pair. Tools should give bounds for those pairs.
   - The one-sided bound line covers about 89 % against a nominal 95 % on this corpus. That rests on 31 bounded rows per
     seed, 94 % of them at the C5-BTBP floor. Every miss is a C5-BTBP row; the 2 rows per seed without it are all
     covered. **EXPLORATORY (seen).**
   - **ESTABLISHED as practice** (the reporting rule above, not the coverage number). Cost: none (a reporting practice).

7. **Direction calls ("is the heavier or the lighter metal extracted more?"):** with 4 reads and *valid* calibration
   (Learn-then-Test at 10 % per-call risk), almost no adjacent pair can be signed on this corpus (0.5 %; 13 % without
   the one C5-BTBP publication).
   - **Other methods call more but bound less.** CRC controls only the per-pair (marginal) false-sign rate (realised
     0.046), and the plug-in has no guarantee. Both sign about 84 % of pairs, but neither bounds the error rate among
     the calls themselves.
   - **Limits:** no calls outside the read hull, and none for Tm/Yb or Yb/Lu without bracketing reads.
   - **With your own prior series** (shrunk toward B1, 4 reads), the wrong-direction rate among calls drops from about
     2.5 % (B1, 4 reads) to 0.3 %.
     - This is when each method calls its most confident half of the pairs (matched coverage 0.49): a ranking, not a
       calibrated call set.
     - It holds on the own-prior rows only (112 cells, 16 publications).
     - The gap comes from 4 of 18 publication × chemotype units.
   - **EXPLORATORY.** Cost: any scenario.

## Chemistry-specific cautions

8. **TODGA-class DGAs (sc009):** do not trust an interpolated Pr/Nd SF.
   - B1 implies about 1.7 where about 1.34 is observed. A borrowed Nd dip gives about 1.43, and lifts R² at this pair from
     0.26 to 0.51.
   - Borrow the dip, not another publication's SF. Over all extractants a universal Nd term does about as well as the
     chemotype one, so the chemotype-specific claim is not established.
   - **EXPLORATORY.** Cost: none (a prior).

9. **Cation exchangers (acidic organophosphorus, chelators):** include Gd in the read set, or measure the Eu-Gd-Tb triad.
   - A deeper family Gd amplitude (about -0.2 to -0.26 instead of -0.09) is an untested hypothesis. Its check (C-GD, a
     descriptive interval) was NOT RUN this round: the supply-audit kill fired, with 1 of the 6 laboratories it needs.
   - Without it, B1 may predict Gd/Eu ≈ 2.2 where the unverified HDEHP planning value implies about 1.5.
   - **HYPOTHESIS.** Cost: one metal choice.

10. **Wider pairs:** between non-neighbours (ΔZ ≥ 2), interpolation from 4 reads beats a straight line through the reads
    by a wide margin (R² 0.68 vs 0.45 at ΔZ = 2).
    - At Nd/Sm outside the largest data cluster, a straight line does about as well.
    - Heavy-lanthanide wide pairs (Ho-Lu) depend on whether Tm or Lu is read; they are not a claim.
    - **EXPLORATORY.** Cost: any scenario.

## Planning measurements

11. **Replicate or add a metal:** replicate a read only if its per-metal σ exceeds about 0.2; otherwise add a metal.
    Give extra reads to the extractants with the most unmeasured metals. **EXPLORATORY.** Cost: radiotracer and
    literature scenarios.

12. **Do not "correct" adjacent SFs with whole per-metal patterns from other publications.** **ESTABLISHED** (closure).
    Cost: none (a rule; any scenario).

13. **Another laboratory's light-end SF** may be trustworthy only to about 0.15 log, a one-pair lead. **EXPLORATORY.**
    Cost: literature scenario (reusing another laboratory's published values).
