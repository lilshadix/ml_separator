# Figure priority

**If a reviewer saw only one results figure, it should be Figure 2.**

Figure 2 is the only figure that states the paper's actual contribution in the units a
chemist and a referee both care about: how much error a small number of experiments
removes, on extractants with no analogue in the training set, under a rule that could be
run tomorrow. It is also the figure with the strongest evidential base — 99 held-out
extractants, five split seeds, twelve support/query draws, every arm on identical draws so
every comparison is paired, an explicit oracle bound, an explicit model-free null, and a
panel showing that a third of extractants get *worse*. If the rest of the paper vanished,
Figure 2 would still support a defensible claim; every other figure exists to explain,
qualify or bound it.

---

## Ranking

### 1 — Figure 2, the few-shot frontier. **Indispensable.**

*Scientific importance:* it is the result. 1.036 → 0.654 for one measurement; the whole
modelling ladder is worth 0.025 + 0.031. *Novelty:* moderate — few-shot calibration is not
new in itself, but the measured statement that a single chosen experiment dominates every
representational improvement in a seven-generation study is a result the field does not
usually publish. *Strength of evidence:* highest in the paper; paired, seeded, bootstrapped
on the right block, with a null and a bound. *Interpretability:* a five-second read.
*Relevance:* the central claim. *Would a reviewer request it?* They would refuse to review
the paper without it.

### 2 — Figure 3, the flattening and its repair. **Very important.**

*Scientific importance:* the most transferable finding here — an error that pooled MAE is
blind to, a mechanistic diagnosis, and a cheap fix. Any group fitting condition-dependent
response curves with a global regressor can check for it in an afternoon.
*Novelty:* high. The diagnosis (the model was never told where a point sits in its own
measurement window, so a flat prediction is the correct answer to the question it was
asked) is not obvious from the loss curve. *Strength of evidence:* strong on the shape
metrics — 5/5 seeds, publication-blocked intervals excluding zero — but the axis rests on
25 extractants and 8 chemotypes, and the mechanism carries the query-set fragility of
Figure S2. *Interpretability:* panels A–C are immediate; D–F need one sentence.
*Would a reviewer request it?* Yes, and they would then ask for Figure S2, which is why S2
should be cited in the caption.

### 3 — Figure 4, where the error is. **Very important.**

*Scientific importance:* it converts "the model is not very good" into an accounting with
an intervention attached to each line, and it is what tells the reader that Figure 2's
result is not a lucky calibration trick — half the error genuinely *is* a per-extractant
constant. The near-coincidence between one optimally chosen measurement (0.504) and
knowing the true per-extractant level (0.507) is the paper's most quotable single fact.
*Novelty:* moderate; oracle decompositions are standard practice, applied unusually
thoroughly here. *Strength of evidence:* the cascade is exact arithmetic on stored
predictions and reproduces the frozen study to 5 × 10⁻⁹. *Interpretability:* panel A is
easy, panel B needs the caption, panel C is the most information-dense object in the paper.
*Would a reviewer request it?* They would ask "what is the remaining error made of?" and
this is the answer.

### 4 — Figure 1, methodology. **Important, but for a different reason.**

It carries no result. It is ranked fourth because without it a referee cannot tell whether
the *k*-shot numbers are a leakage artefact, and that is the single most likely reason for
rejection. Its job is to make the pool/query separation, the chemotype hold-out and the
frozen-model-plus-calibrator structure impossible to misread. *Would a reviewer request
it?* If it were missing, they would ask for exactly this diagram.

### 5 — Figure 5, generalisation and coverage. **Useful; the weakest main figure.**

*Scientific importance:* high in principle — "does it generalise?" is the question average
MAE cannot answer. *Strength of evidence:* mixed, and this is why it ranks fifth. Panel A
is observational and underpowered: the far − near difference at *k* = 0 is +0.35
[−0.04, +0.66] and does not exclude zero with 17 chemotypes per tercile. Panels B and C are
a genuinely controlled experiment with byte-identical held-out rows and two negative
controls, and they do separate (+0.463 [0.251, 0.685] on the most distant subset) — but
they use a different learner from panel A and carry the load-bearing qualification that 57
of 131 scoring units are clusters the restricted arm structurally cannot serve.
*Interpretability:* the two halves answer related but distinct questions and the reader has
to be told which is which. *Would a reviewer request it?* Yes — and they would probe it
hardest. See `FINAL_FIGURE_REPORT.md` §4 for the recommendation.

### 6 — Figure 6, acquisition. **Useful; publish if space allows, else supplementary.**

*Scientific importance:* it is the deployment recommendation, and it contains a clean
negative result (model uncertainty is the wrong signal, and the reason is structural rather
than a calibration failure) that saves other groups a wasted effort. *Novelty:* the
negative result is the novel part. *Strength of evidence:* strong and well-powered — 143
extractants, identical pools, seventeen policies, paired intervals — but it is logically a
corollary of Figure 2: it decomposes the *k* = 1 point rather than adding a new claim.
*Interpretability:* immediate. *Would a reviewer request it?* Only after reading Figure 4A
and noticing the 0.654 vs 0.504 gap — which the Figure 4 caption points at explicitly.

---

## Supplementary priority within the supplement

`S2` (query-set fragility) is the most important supplementary figure by a wide margin: it
is a negative result about the paper's own contribution and its absence would be a
reporting failure, not a stylistic choice. `S9` (dose–response) is next, because it is the
evidence that separates Figure 5's coverage effect from a class prior. `S4` and `S3` follow
as the ablations a methods-minded referee will want. `S1`, `S5`–`S8`, `S10` are ordinary
diagnostics.
