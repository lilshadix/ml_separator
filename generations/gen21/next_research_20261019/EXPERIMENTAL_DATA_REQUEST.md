# Request for experimental data (what the corpus cannot supply)

None of the following can be produced by software. Nothing here is a completed experiment, and no
sample size is claimed to guarantee R² ≥ 0.50.

## 1. Repeat whole curves (the noise ceiling of adjacent log SF)

**Why.** The laboratory repeatability of an adjacent log SF has never been measured in this corpus.
gen20.1 found only 115 run-paired candidate pairs in 2 chemotypes. Same-condition double reports give
sd(Δ log SF) ≈ 0.086 with a 0.98 error correlation between the two metals, but those are probably the
same measurement reported twice (`DATA_QUALITY_REPORT.md`). Without a ceiling we cannot tell whether
the remaining error of interpolation (MSE ≈ 0.029 at k = 3) is unexplained chemistry or noise.

**What.** For about 6 extractant systems spread over the main donor classes (one unsubstituted
tetraalkyl DGA such as TODGA; one α-substituted DGA; one N-heterocycle donor such as a BTBP or
phenanthroline diamide; one amide / malonamide; one phosphoryl donor), at one fixed condition each:
* the full La–Lu series (14 metals; Pm excluded), measured **in independent runs**: fresh phases, fresh
  stock solutions, on different days, ≥ 3 repeats;
* the inner-reference repeat: within one run, the same sample analysed twice. This separates analysis
  noise from preparation noise;
* recorded for every run: acid and concentration, nitrate including any added salt, aqueous complexant
  and its concentration, diluent and modifier fractions, extractant concentration, phase ratio, contact
  time, temperature, metal loading, the analytical method and its detection limit **per metal**.

**What it distinguishes.** A common-mode run error (a level shift that cancels in log SF) versus
per-metal error (which does not). The detection-limit statement removes the guesswork about floors
(DATA_QUALITY_REPORT D5).

## 2. New non-DGA chemistry (transfer, not noise)

**Why.** About four effective chemotypes carry every pooled number, and the DGA chemotype holds half the
rows. "Generalisation" here means a chemotype-plus-publication hold-out inside one literature corpus.

**What.** New extractants **not in the corpus**, especially N-heterocycle donors and α-substituted
DGAs (the classes where the Gd dip differs), each measured at 2–3 conditions with the full series and
the metadata above. They should be measured by a laboratory that is not among the corpus's main
publications.

**How it stays an external check.** Fix the frozen candidate (`manifests/freeze_manifest.json`) and its
read policy **before** the data arrive. Give the analyst only the 4 metals the policy selects as
inputs, and score the remaining adjacent log SF once. Never tune on these data.

## 3. Sources to recover (desk work, not the laboratory)

* Shimada et al., SX&IE 22(2) 147–161: Table 1 and Figure 5, to settle the 5 M Pr/Nd value of
  DMDPhPDA copy A.
* The published detection limits for the C5BTBP series (pub_6a0ef8528c) and for the ORNL-imported cells.
