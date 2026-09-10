# Gen16 — the confirmation run

*Executed once, 2026-09-10, on five seeds no discovery agent ever evaluated.  Whatever it
returned is the result; nothing was re-run and no sixth claim was added.*

## 1. The seeds, and how they were withheld

Generated before any lead ran, from the rule recorded in `PRE_REGISTRATION.md` §0:

```
seed_i = int(sha256("gen16-confirmation-seed-{i}").hexdigest()[:8], 16) % 900000 + 100000
i = 1, 2, …; skip collisions with the discovery seeds and with earlier seeds; first five; sorted
```

giving **`[137171, 506411, 721632, 937310, 939478]`**, whose commitment
`sha256(json.dumps(sorted(seeds)))` = `5a30455bc67d364aa3e65f6d3d28bd4b6fcf3301fe979ca4a1be4a9547de76a9`
was published in the sealed pre-registration and is verified by `scripts/g16_confirm.py` at run
time against both the commitment and the rule.  The seed file lived outside the repository until
this run; `scripts/g16_audit_seeds.py` checks by AST walk that no discovery file passes `seeds=`
to the frozen bench or contains a confirmation-seed literal, and `tests/test_protocol.py` checks
that the public rule hashes to the published commitment.

**The withholding is procedural, not cryptographic.**  The rule is public, so anyone can compute
the seeds; what is guaranteed is that no discovery number was computed on them.  The runner
refuses a second execution unless forced (`results/confirmation/CONFIRMATION_RUN_ONCE.json`).

## 2. Which claims qualified, and which did not

`PRE_REGISTRATION.md` §4 admits a claim only if it passed its registered decision rule **in all
five designs** on the discovery seeds and was not removed by its two refuters.  **Exactly one of
the five leads produced such a claim.**

| lead | candidate | why it did or did not go to confirmation |
|---|---|---|
| **L3a** | `G14_saved_vs_0` | **admitted.**  Positive with permutation p < 0.0005 and a chemotype-blocked CI excluding zero in all five designs.  Both refuters dented it *materially* but neither removed it: they left the pooled arithmetic exactly intact and restricted its **scope** (§4). |
| L1 | — | undecided (`DECISION_REPORT.md` §4): the registered estimator had no power and every post-hoc positive fails the diglycolamide control.  Neither a claim nor a null. |
| L2 | — | gate closed; no arm was run. |
| L3c | — | null, and negative on its second registered endpoint. |
| L4 | `AOPT_vs_RANDOM` | **not admitted**: the sign flips under B and BQ, so it never passed the five-design rule; and its lens-A refuter then killed it outright (a featureless "biggest chemotype first" order reproduces it). |
| L5 | — | no registered contrast reached the 0.02 margin in any design. |

Nothing was promoted from the exploratory family to fill the remaining four slots, and no claim
found by a refuter was promoted — a refuter check can kill a claim, never resurrect one.

## 3. The result

`saved` = expected measurements to the first useful candidate under a random order minus the same
under the model's order, in **measurements**; 455 tasks per design; permutation and
chemotype-blocked bootstrap as registered.  `p` is the chemotype-blocked bootstrap p (the
permutation p is 0.0000 at its 1/2000 resolution in every cell of both runs).

| design | discovery point | discovery 95 % CI | disc. p | **confirmation point** | confirmation 95 % CI | conf. p | seeds | LOCO | passes |
|---|---|---|---|---|---|---|---|---|---|
| B | +2.207 | [+0.521, +4.118] | 0.007 | **+2.086** | [+0.234, +4.170] | 0.016 | 5/5 | stable | ✅ |
| BR | +2.149 | [+0.455, +3.985] | 0.009 | **+2.170** | [+0.252, +4.195] | 0.022 | 5/5 | stable | ✅ |
| BQ | +2.179 | [+0.468, +4.070] | 0.010 | **+2.123** | [+0.248, +4.196] | 0.014 | 5/5 | stable | ✅ |
| A | +2.492 | [+0.771, +4.544] | 0.004 | **+2.285** | [+0.620, +4.366] | 0.005 | 5/5 | stable | ✅ |
| **BP** | **+1.988** | [+0.443, +3.781] | 0.009 | **+1.909** | [+0.360, +3.760] | 0.014 | 5/5 | stable | ✅ |

**The claim replicates.**  Under BP the saving moves from +1.988 to +1.909 measurements (−4 %),
the interval still excludes zero, five of five fresh seeds agree, no held-out chemotype flips the
sign, and the same holds in all five designs.  `E_random` is 6.906 in both runs (it is a property
of the task set, not of the model).

**Multiplicity.**  The fleet wrote **940 contrast rows: 126 registered and 814 exploratory**
(`DECISION_REPORT.md` §1 lists them by lead).  Benjamini–Hochberg on the discovery p-values leaves
this claim's BP row at **q = 0.039** within the registered family of 126 and **q = 0.025** over all
940, and every one of its five designs stays under q = 0.05 in both families.  The confirmation
run itself tested one claim, so no correction applies within it.

## 4. What the confirmation does **not** establish

Both refuters ran the scope test the lead had named as the first thing to demand and had
correctly declined to run unregistered.  It is decisive and it is not repaired by confirmation:

| regime | `saved` under BP | 95 % CI | tasks |
|---|---|---|---|
| pooled across laboratories (the confirmed claim) | **+1.909** | [+0.360, +3.760] | 455 |
| the same, rebuilt **one fold at a time** (no candidate predicted by a model trained on another candidate in its task) | **+0.906** of an E_random of 5.52 | — | 455 |
| candidate sets restricted to **one publication** | **+0.0046** | [−0.016, +0.022] | 1 230 |

Within a single laboratory's candidate set the saving is **0.07 % of the no-model cost**, the
interval contains zero in all five designs, and 69.6 % of those tasks give every candidate the
same direction call (0.0 % pooled).  On the same within-laboratory tasks a perfect sign call
would save 0.414 measurements, so the model captures 1.1 % of the available headroom there
against 47.3 % pooled.  A second refuter check found the same thing from the other side:
restricted to one chemotype the saving falls to +0.334 and fails the registered rule in all five
designs.

**So the confirmed quantity is real, it is between-laboratory, and its leak-free magnitude is
about 16 % rather than 29 %.**  It says the model knows which end of the series a system from
*some other* laboratory prefers.  It does not say a chemist choosing among the candidates in front
of them saves two measurements.  The honest one-line statement is in `DECISION_REPORT.md` §2 and
both limitations are carried everywhere the number is.

**One disclosed deviation from the sealed text applies to this claim** (`DECISION_REPORT.md` §10,
item 10): L3 used its own chemotype-blocked bootstrap at 2 000 replicates with a percentile
interval only, rather than the frozen `paired_contrasts` at 10 000 with percentile and BCa,
because its resampling unit is the task rather than the extractant.  The claim therefore meets
L3a's registered rule and the percentile half of P1; the BCa half was never computed for it.  The
permutation null, which is its primary evidence, is unaffected.

## 5. Files

`results/confirmation/`: `C1_L3A_measurements_saved_board.csv`,
`C1_L3A_measurements_saved_contrasts.csv`, `discovery_vs_confirmation.csv`, `verdicts.csv`,
`CONFIRMATION_RUN_ONCE.json`.  Runner `scripts/g16_confirm.py`; claim definition
`gen16/claims.py`, validated against the committed discovery CSV to 4.4e-16 on the point estimate
before it was pointed at these seeds.
