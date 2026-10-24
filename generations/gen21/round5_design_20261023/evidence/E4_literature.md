# E4: Literature that could inform round 5 (physics, chemistry, ML of adjacent-Ln selectivity)

Prepared 2026-10-23 for the round-5 design. This is a literature review plus a few small exploratory checks of literature predictions against round-4 out-of-fold predictions and the C3 bench. It trains no models. It is not confirmatory evidence of any kind.

## How to read this note

**Evidence labels:**
- **[V]**: verified this session from the paper's abstract or bibliographic record (Europe PMC REST, publisher or repository pages).
- **[S]**: search-result snippet or secondary citation only. The primary text was not read.
- **[B]**: background domain knowledge, not re-verified this session. Treat as a hypothesis until checked.
- **[P]**: a project file (path given).
- **[D]**: a data check run for this note. It is exploratory development evidence on already-seen dev seeds; the method is in Appendix A.

**Access.** Many publisher pages (ScienceDirect, RSC, OSTI, PubMed HTML) refused automated access. Most items therefore rest on abstracts. Every numeric claim below names its source.

**Compliance note.** One WebFetch call to a J-STAGE PDF (Kawabe & Masuda 2001) was cached automatically by the fetch tool under `C:\Users\Bandai\.claude\projects\...\tool-results\`. That file was not opened and is not used here.

**Unit bridge.** All conversions are at 298 K.
- 1 log10 unit = 2.303·RT = **5.71 kJ/mol**.
- The C3 strict-adjacent target has a mean |log SF| of about 0.17 in the interior rows [D], which is **≈ 1 kJ/mol**.
- B1's Gd term, −0.072 log, is **0.41 kJ/mol**.
- Per-pair-type SDs of log SF are 0.12–0.35 [P `round4_20261022/results/REF/per_pair_BPG.csv`], which is 0.7–2.0 kJ/mol.

Any physical descriptor that is meant to move adjacent log SF must therefore resolve **sub-kJ/mol differences between neighbouring Ln**.

---

## 0. Bottom line

1. **The literature mostly supports what B1 already is.** log D is smooth in ionic radius (the contraction is quadratic in Z, so radius is the right coordinate), with one robust mid-series irregularity at Gd. Classical aqueous log K series show the same "Gd break". EDTA's Gd sits **0.27 log** below its Eu–Tb chord (Martell–Smith values [S]). In C3 extraction data, Gd sits **0.115 ± 0.013** below its Eu–Tb chord (n = 127 cells) [D]. The mean C3 log SF is −0.005 for Gd/Eu and +0.225 for Tb/Gd [D].
2. **Tetrad / double-double / inclined-W effects are real but small and context-signed.** Their sign and size follow the nephelauxetic (Racah-parameter) difference between the organic complex and the aqua ion [V Kawabe & Masuda 2001: ΔE3/ΔE1 = 0.29 ± 0.07 and 0.25 ± 0.06 for Peppard's two extraction systems]. C3 shows a convex-tetrad pattern on average: the first pair of each tetrad has the larger SF, with Ce/La 0.19 > Pr/Ce 0.11 ≈ Nd/Pr 0.11 and Tb/Gd 0.23 > Dy/Tb 0.09 [D]. But it does not transfer:
   - universal pair-type constants on top of B1: in-sample oracle **+0.012 R²**, leave-one-chemotype-out (LOCO) **−0.018** [D];
   - S02 lost 0.024 [P];
   - gen13 D7 measured a reproducible non-smooth amplitude of only 0.035 log [P].
3. **The Gd anomaly is additive, not proportional to the cell's selectivity amplitude.** The separable "log β ∝ 1/r" LFER (Solov'ev–Tsivadze proportionality) predicts that metal-specific wiggles scale with ligand strength. In C3 they do not:
   - Spearman(Gd chord deviation, cell slope) = −0.13 (p 0.15), and light-selective cells still show a −0.073 Gd chord [D];
   - an amplitude-scaled Gd term is worse even as an in-sample oracle (ΔR² −0.003) and under LOCO (−0.017) [D].

   This fits a metal-intrinsic origin common to all cells (the hydrated Ln³⁺ reference state plus electronic structure). It does not fit a ligand-strength-scaled origin.
4. **Donor softness modulates the Gd/tetrad amplitude, as RSPET predicts, but the modulation does not transfer.** Gd chord by family: N-heterocyclic −0.21, other amides −0.18, DGA −0.05 [D]. gen13 D7 found −0.127 outside sc009 and −0.025 inside [P]. A family-specific Gd scale is an in-sample oracle of +0.014. Under LOCO it is −0.001 to −0.004 [D], consistent with S01 = B1 [P].
5. **Heavy-end saturation is the one literature prediction with a positive LOCO sign.** The mechanisms behind it:
   - steric crowding and loss of coordination for small Ln (DMDOHEMA, La ≈ Eu > Lu [V Wang et al. 2024]);
   - Ln(BTP)₃ and N-donor drop-off for heavy Ln [S];
   - the plateau of DGA extraction for heavy Ln.

   What the C3 data show [D]:
   - The C3 heavy tail is flat: mean log SF Er/Ho 0.04, Tm/Er 0.01, Yb/Tm −0.02, Lu/Yb 0.01.
   - PWL_R's **constant** continuation beyond the last read leaves 12% of rows (10% of SSE) at R² ≈ −0.01.
   - A **damped** slope continuation adds **+0.004 to +0.006 R²** (in-sample oracle d = 0.375: +0.0060; LOCO: +0.0044 to +0.0051; family-shrunk LOCO: 5/5 seeds positive on both designs).

   This is small, below the +0.01 margin, and conflates curvature with slope-noise shrinkage.
6. **Aqueous complexation is the only physics-mandated, exactly known shape correction.** It is unusable now:
   - The bench cannot see the 19 C3 cells with an aqueous complexant [P PROVENANCE.md].
   - A holdback ligand subtracts its own log β series. That series can **invert** the Gd term, since EDTA's Gd dip would become a log D bump.
   - High-throughput work shows holdback agents re-shape selectivity strongly: oxaloacetic acid with HDEHP gives a 4× SF gain [V Augustine et al. 2026].
7. **Thermochemistry and DFT descriptors are 5–50× too coarse for adjacent Ln, unless errors cancel.**
   - Hydration free energies are good to about ±10 kJ/mol [V Kepp 2019], i.e. ±1.75 log.
   - DFT An/Ln ΔΔG has an MAE of 2.6 kJ/mol even with GPR on a curated set [V Sumiyoshi 2026].
   - DOTAM substitution energies of 5–15 kJ/mol are "comparable to DFT uncertainty" [S Shapiro 2026].
   - This explains the round-4 3D null (S06) and gen15's xTB composition trap [P].
8. **Published ML for Ln log D works only under random splits.**
   - Liu et al. 2022: 1,202 log D, 109 ligands, R² 0.85, MAE 0.34 on a random split [V].
   - The same 1,202 points under leakage-controlled **grouped** splits give R² **0.20 to −0.03** [V Harish 2026].
   - Chaube 2020: an adjacent-pair log K difference MAE of 0.11–0.34 [V], which is the size of the signal itself.

   No paper found reports held-out **adjacent log SF** skill under source-grouped splits. The programme's B1 = 0.452 with four reads has no published counterpart.
9. **Size-selective and preorganised ligands produce interior extrema and U-shapes** [V Roca-Sabio 2009; Hu 2020; Carolan 2013; Healy 2019]. These are the strongest literature shapes that radius interpolation cannot express, and they are almost absent from C3 (69% DGA by cells). They matter for data acquisition, not for modelling the current corpus.
10. **Implication for round 5.** Literature-mandated structure can justify adopting a small prior **by argument** (damped heavy-end continuation; holdback-ligand subtraction). With 13 source clusters (Kish 2.24) and POWER.md's 40–80 new sources for a B1-sized gain, no literature-sized effect (≤ 0.006 R²) can be **confirmed** on this corpus.

---

## 1. What radius interpolation already encodes

### 1.1 The lanthanide contraction

**Literature:**
- Quadrelli (2002) showed that Ln–O distances and Ln³⁺ radii decay at least quadratically across the 4f series [V, doi:10.1021/ic015580v].
- Seitz, Oliver & Raymond (2007) used an isostructural 8-coordinate La–Lu series (Pm excluded). Individual Ln–O bonds deviate from a quadratic, but the **sum** of bond lengths is almost ideally quadratic in the f count. They derive this from Slater's model [V, doi:10.1021/ja072750f].
- Raymond et al. (2010) argue that **1/r is linear in effective nuclear charge** [S, doi:10.1016/j.crci.2010.03.034].
- D'Angelo et al. (2011) give revised aqueous radii from EXAFS + MD that are "much more regular" than earlier sets [V, doi:10.1021/ic200260r].
- Sobolev & Sulyanova (2023) fit an LnF₃ radius system: La³⁺ 1.1671 Å to Lu³⁺ 0.9439 Å (−24%) as a cubic in Z [V, doi:10.3390/ijms242317080].

**Shape prediction for four reads.** log D is smooth in r, and the radius steps shrink toward Lu. Shannon CN8 steps are 0.017 Å for La→Ce and 0.008 Å for Yb→Lu [P `gen13sep/metals.py`]. So a fixed per-Z slope over-predicts heavy-pair SFs, while a radius coordinate does not.

**Beyond PWL_R + Gd?** No. Coordinate choice is second order. Over a read span h = 0.1 Å, the largest gap between 1/r and its chord is h²/(4r³). That is ≈ 2% of the segment's change in 1/r (h/r²): about 0.01 log for a 0.5-log segment. The data agree: PWL_R 0.418 vs PWL_Z 0.411 [P round-3 ATTEMPTS B0], PCHIP_R −0.018 [P R01], HBK@ML 0.419 [P].

### 1.2 Electrostatic LFERs and "thermodynamic radii"

**Literature:**
- In hard-donor Ln chemistry, log K rises with 1/r, with a small break at Gd [S; Choppin-school electrostatic picture, B].
- Solov'ev & Tsivadze fitted 2,854 log β1 values (445 organic ligands, 13 metals) with a proportionality between metals: log β1(Mj) = (r1,i / r1,j) · log β1(Mi). This uses fitted "thermodynamic radii" that decrease steadily across the series. Cross-validated error is **0.3–1.0 log** depending on the radius gap [S, istina record "Thermodynamic radii of lanthanide ions derived from metal–ligand complexes stability constants"; follow-up Comments Inorg. Chem. 43:16, 2022, doi:10.1080/02603594.2022.2087637, V metadata only].

**Shape prediction.** The model is separable: log β(Ln, L) = e_Ln × s_L. Every metal-specific irregularity, Gd included, would then **scale with the ligand's overall strength and slope**.

**Data check [D] (probe 4).** The prediction fails for the Gd anomaly.
- Spearman(Gd chord deviation, cell slope) = −0.129 (p = 0.15, n = 127).
- By slope band, the mean Gd deviation is −0.073 (light-selective, n = 27), −0.129 (slope 0–10 log/Å, n = 43), −0.119 (10–20, n = 50) and −0.163 (≥ 20, n = 7).
- Scaling B1's Gd component by the read-estimated |slope| is worse than B1 even as an in-sample oracle: ΔR² −0.0025 BPG / −0.0037 BP. Under LOCO it is −0.017 / −0.018, 0/5 seeds.

**Verdict.** The Gd term should stay additive. Proportional or amplitude-scaled anomaly priors are not supported.

### 1.3 Why external aqueous log K series do not fix the amplitude

**Literature.** Extraction trends can run **opposite** to complexation trends. For a nitrilotriacetamide, extraction efficiency correlates negatively with ionic radius while complexation strength depends positively on it [V Wang X. et al. 2025, doi:10.1021/acs.inorgchem.5c02125]. TOPO extraction enthalpies are about **29 kJ/mol in magnitude and constant from La to Er**, with a slight decrease Tm→Lu (sign as reported in the paper; not checked here) [V Grimes, Zalupski & Martin 2014, doi:10.1021/jp507727v]. So the series trend in log D is largely entropic or outer-sphere, not an inner-sphere enthalpy ladder.

**Project.** gen13: Spearman(observed amplitude, aqueous-log-K-predicted slope) = **+0.00** over 82 extractants [P `gen13_separation/analysis/stage2/STAGE2_REPORT.md` §2].

**Verdict.** Consistent. log D shape ≠ aqueous log K shape of an analogous ligand. Do not use aqueous log K series of extractant analogues as shape priors.

---

## 2. The Gd break, tetrad / double-double effect, inclined W, nephelauxetic effect

### 2.1 The "Gd break"

**Literature:**
- **Aqueous EDTA** (Martell & Smith via O'Brien et al. 2022/2023 and a UCT teaching reproduction) [S]: La 15.46, Eu 17.32–17.35, Gd 17.35–17.37, Tb 17.93, Dy 18.30, Ho 18.62, Er 18.85, Tm 19.32, Yb 19.51, Lu 19.80–19.83.
  - Gd lies **0.27–0.28 log below** the Eu–Tb chord.
  - The Gd/Eu step is ≈ 0.02, against ≈ 0.56 for Tb/Gd: the classical break.
  - Sources: doi:10.1039/d2cp01081j; correction doi:10.1039/d3cp90217j.
- **Kawabe (Ln–EDTA)**: log K shows a Gd break and a subtle convex tetrad, while ΔH and ΔS show **step-like changes in the middle Ln**. The author attributes the apparent Gd break to a **hydration change of the middle Ln–EDTA complexes**. Different measurement methods scatter by ±0.5 log [S, preprint "A puzzle of Gd-break and tetrad effect of aqueous lanthanide(III)-EDTA complex formation", nagoya.repo.nii.ac.jp/records/18232].
- **Persson et al. 2008, "the myth of the gadolinium break"** [V, doi:10.1002/chem.200701281]:
  - The aqua ions show **no abrupt** change of hydration number.
  - Ln–O distances match CN 9 for La–Nd, become shorter from Sm, and reach the CN-8 value at Ho.
  - Capping-water vacancies begin at Ho and reach x = 0.8 at Lu (Ln(H₂O)₉₋ₓ).

  So the aqua-ion contribution is a **gradual** Sm→Lu change, not a Gd step.
- **Verma & Saxena (1989)**, testing inclined-W and tetrad hypotheses on mixed-ligand chelates, found the effect **stronger at f⁷ (Gd) than at f³–f⁴ (Nd–Pm)** [S, Springer BF02840595 abstract snippet].

**Shape prediction for four reads.**
- A single-element **Gd dip**, not a step: D7 gave Gd dip r = +0.57 vs gd_break step r = +0.08 [P `gen13 D7 FINDINGS.md` §2].
- It is accompanied by **small Gd/Eu and large Tb/Gd SFs**.
- When Gd is read, the dip contaminates the interpolation of Eu and Tb. B1's projection removes that contamination [P `r4/refs.py`].

**Data [D] (probe 1, P3; probe 2).** The C3 mean log SF (heavier minus lighter) is Gd/Eu −0.005 and Tb/Gd +0.225 (n = 158 and 165 cells). The Gd chord by family is:

| family | Gd chord deviation | n cells |
|---|---|---|
| all C3 | −0.115 ± 0.013 (median −0.097) | 127 |
| diglycolamide | −0.053 ± 0.014 | 70 |
| amide_other | −0.184 ± 0.026 | 13 |
| n_heterocyclic_polydentate | −0.209 ± 0.026 (median −0.245) | 32 |
| light-selective cells (any family) | −0.073 ± 0.029 | 27 |

**Value beyond B1.** B1 already encodes the transferable part, and the extra lift is small:
- B1 raises Eu/Gd from 0.28 to 0.42 R² and Gd/Tb from 0.32 to 0.55 [P REPORT §3].
- A family-specific scale of B1's Gd component (DGA 0.78, amide 1.71, N-heterocyclic 2.15, phosphoryl 2.18, pooled 1.11) is an in-sample oracle of +0.014 R².
- The same scale under LOCO is **−0.0006 to −0.0044** [D probe 3]. When sc009 (the DGA supercluster) is held out, the "pooled" scale comes from non-DGA cells and over-corrects DGA.

**Verdict.** Literature and data agree that Gd is the one robust, additive, metal-intrinsic irregularity. Its donor dependence is real in sample but **not transferable with this chemotype structure**.

### 2.2 Tetrad effect and the "double-double" effect

**Literature:**
- **Origin.** Peppard, Mason & Lewey (1969) first described a "tetrad effect" in liquid–liquid Ln(III) distribution coefficients [S, J. Inorg. Nucl. Chem. 31:2271]. Fidelis & Siekierski introduced the equivalent "double-double effect" in complex-formation free energies and enthalpies [S, J. Inorg. Nucl. Chem. 1966 and 1971 (33:3191)].
- **Theory.** Jørgensen (1970) and Nugent (1970) explained it as a variation of the nephelauxetic ratio [S, J. Inorg. Nucl. Chem. 32:3127 and 32:3485].
- **RSPET.** Kawabe's refined spin-pairing energy theory gives tetrads whose **size and sign** follow the difference of Racah parameters (E¹, E³) between the two environments [V Kawabe 1992, Geochem. J. 26]. Kawabe & Masuda (2001) refit Peppard's two extraction systems with RSPET and got ΔE3/ΔE1 = **0.29 ± 0.07 and 0.25 ± 0.06**, close to the spectroscopic Nd(III) value 0.23 ± 0.02. Coordination changes largely cancel in ΔGr° [V, doi:10.2343/geochemj.35.215].
- **Hydration correction.** In Ln–carbonate complexation, a regular **convex** tetrad appears only after correcting for the hydration change of light Ln³⁺(aq) [V Kawabe et al. 1999, Geochem. J. 33:267].
- **Covalency.** The magnitude of the double-double effect in unit-cell volumes **increases as the ligand atom's electronegativity decreases**, i.e. with covalency [V Springer abstract BF02204319, Siekierski].
- **Electronic structure.** Ln(III)–Cl bonding has Ln 5d mixing that decreases across the series; 4f participation is marginal [V Löble et al. 2015, doi:10.1021/ja510067v].

**Shape prediction for four reads.**
- Cusps at the tetrad boundaries: Nd|Pm, Gd, Ho|Er.
- With convex (M-type in log D) tetrads, the **first** adjacent pair of each tetrad has the largest SF and the last the smallest. The boundary elements (Nd, Gd, Lu) sit low relative to a smooth curve.
- The **sign is environment-specific**: harder O-donor complexes compared with the O-donor aqua ion give small ΔRacah, hence small tetrads. Softer or N-donor complexes give larger ones.
- Pm is absent, so Nd/Sm is not strict-adjacent and the first boundary is invisible to the target.

**Data [D] (probe 1, P3).** C3 mean log SF by pair type:

| pair | Ce/La | Pr/Ce | Nd/Pr | Eu/Sm | Gd/Eu | Tb/Gd | Dy/Tb | Ho/Dy | Er/Ho | Tm/Er | Yb/Tm | Lu/Yb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean log SF | 0.19 | 0.11 | 0.11 | 0.10 | −0.005 | 0.23 | 0.09 | 0.10 | 0.04 | 0.01 | −0.02 | 0.01 |

- Ce and Pr lie +0.064 ± 0.013 above the La–Nd chord (n = 113; DGA +0.091). This is the convex first tetrad.
- gen13 D7 significant residuals: Nd −0.063, Gd −0.072, Lu −0.057 (boundary elements low, as convex tetrads predict) and Ho +0.058 (not predicted) [P].
- B1's per-pair biases (pred − y) match the convex first tetrad: La/Ce +0.024, Ce/Pr −0.054, Pr/Nd −0.037 [P per_pair_BPG.csv].

**Value beyond B1.** Real but untransferable. Universal pair-type constants on B1 are an in-sample oracle of +0.012 R² and **−0.018 under LOCO** [D probe 2]. Family × pair-type constants are an oracle of +0.073 but **−0.013 to −0.047 under LOCO**. Already measured in the project:
- S02 (electron-count shapes × donor class) −0.024 [P];
- B2 (14 potentials) −0.022 vs B1 [P];
- gen13 tetrad basis ≤ 0.006, "stop spending" [P EVIDENCE_INVENTORY E3].

**Verdict.** The literature explains why. The tetrad sign and size depend on ΔRacah between phases, which this corpus cannot infer for a held-out chemotype. Its magnitude (≈ 0.03–0.06 log) is at or below cell-level noise (D7: replicate SEM 0.205 per cell-metal [P]). **Do not reopen universal tetrad, double-double or inclined-W terms.**

### 2.3 "Inclined W" systematics

**Literature.** Sinha's "Inclined W systematics of the lanthanides" (NATO ASI volume, 1983, doi:10.1007/978-94-009-7175-2) [S]. Properties plotted against the ground-term orbital quantum number L form four linear segments on an inclined baseline [B].

**Prediction.** It is the same family as 2.2: piecewise-linear segments in L with breaks at f³/f⁴, f⁷ and f¹⁰/f¹¹.

**Verdict.** No extra information beyond 2.2. The same non-transfer applies.

### 2.4 Nephelauxetic / covalency effects and N- vs O-donors

**Literature:**
- For Ln(n-Pr-BTP)₃, the N K-edge XAS pre-edge energy is **inversely correlated with the Ln distribution ratio** [S, PSI conference abstract, indico.psi.ch/event/2245/contributions/3628].
- Relative Ln(BTP)₃ abundances (ESI-MS) correlate with D [S, New J. Chem. 2009, b9nj00371a].
- Kolarik's n-Pr-BTP gives Am/Eu ≈ 130 [S, SXIE 17:1155, 1999].
- Electronic-structure informatics for An/Ln with terpyridines: GPR on DFT gives MAE 17.5 kJ/mol for ΔG_bind and 2.6 kJ/mol for ΔΔG(An–Ln). Metal ionisation potentials and inner-site charge patterns drive selectivity [V Sumiyoshi 2026, doi:10.1021/acs.inorgchem.5c05680].

**Prediction.** Softer N-donors give larger electronic (tetrad/Gd) wiggles and a larger heavy-end fall-off.

**Data [D].** N-heterocyclic cells have the largest Gd chord (−0.21) and the strongest heavy-end damping (oracle d 0.25–0.375).

**Value beyond B1.** As for 2.1: in-sample yes, LOCO no.

**A literature-motivated probe the corpus cannot test (data acquisition).** Y³⁺ has no 4f electrons. Its CN8 radius (≈ 1.019 Å [B]) is almost that of Ho³⁺ (1.015 Å). So **log D(Y) minus the Ln-radius-interpolated log D at Y's radius** isolates the cell's non-radius (4f/covalency) component. Geochemistry uses Y/Ho fractionation and Y's "nomadic" position in the same way [B; Limaye & Saxena 1994 mention Y's "nomadic behaviour", V].

The archive holds only **49 Y rows from 4 DOIs** [D `dataset_all_metals/clean/master_clean.parquet`]. That is too few now, but Y is routinely measured in REE separation papers, so it is cheap to request.

---

## 3. Hydration, outer sphere and concentration: why log D shape ≠ log K shape

### 3.1 Hydration enthalpies and free energies

**Literature:**
- Kepp's thermochemically consistent ΔG_hyd values are accurate to about **±10 kJ/mol** [V Kepp 2019, doi:10.1021/acs.jpca.9b05140; 2018, doi:10.1021/acs.jpca.8b06674].
- The project's Kepp-scale values are [P `gen13sep/metals.py`]: La −3277, Ce −3332, Pr −3377, Nd −3412, Sm −3456, Eu −3476, Gd −3500, Tb −3549, Dy −3557, Ho −3602, Er −3627, Tm −3647, Yb −3688, Lu −3698 kJ/mol. Adjacent steps are irregular (8–55 kJ/mol) at a precision of ±10 kJ/mol.

**Prediction.** Extraction ΔG is a small difference of two large terms (organic complexation and dehydration). Irregularities in ΔG_hyd enter every cell identically unless the organic complex compensates.

**Value beyond B1.** Not as a descriptor: the ±10 kJ/mol (±1.75 log) precision is **10–20× the whole adjacent signal**. It is consistent with the **additive** Gd behaviour in 1.2. The same aqua-ion reference state is common to every extraction from aqueous Ln³⁺, which offers a mechanism for an additive, universal mid-series irregularity [interpretation].

### 3.2 Outer-sphere water and nitrate

**Literature:**
- TODGA's adjacent-Ln selectivity tracks **co-extracted water**, through hydrogen bonding of outer-sphere nitrate to water clusters in the nonpolar phase [V Baldwin et al. 2018, doi:10.1021/acscentsci.8b00223].
- Branching of DGA N-alkyl chains controls outer-sphere structure. Small changes near the carbonyl oxygens produce "dramatic shifts" in strength **and selectivity** across 15 DGAs (12 new) [V Stamberga et al. 2020, doi:10.1021/acs.inorgchem.0c02861].
- Me-TODGA diastereomers change f-element extraction by up to **two orders of magnitude** [V Wilden 2019, doi:10.1002/chem.201806161].

**Prediction.** Within one donor class, the heavy-end shape and the Gd/tetrad amplitude depend on alkyl branching, stereochemistry and diluent, i.e. on features that 2D topology summaries encode poorly.

**Value beyond B1.** It explains the residual cell-specific shape: gen13 D7 found 93% of non-smooth residual variance cell-specific or noise [P]. It does not supply a transferable feature. The stereochemistry point matters for **data curation** (PROVENANCE: mixed stereochemistry).

### 3.3 Coordination crowding at the heavy end

**Literature:**
- **DMDOHEMA** (malonamide): metadynamics free-energy surfaces explain the order La ≈ Eu > Lu. Coordination is retained La→Eu and **lost Eu→Lu**: ligand crowding sets selectivity, opposite to conventional extractants where charge-density enthalpy dominates [V Wang X. et al. 2024, doi:10.1039/d4sc05061d].
- A TODGA + SO₃-Ph-BTBP system shows a significant drop in heavy-Ln (Tb–Lu) D [S].
- Phenanthroline diamides: D **decreases with Z**, with adjacent SF ≈ 3 [V Ustynyuk 2015, doi:10.1039/c5cc01620g]. Another phenanthroline ligand also shows a steady decrease with Z [V Yang 2021, doi:10.1021/acs.inorgchem.1c01035].
- TODGA in **ionic liquids** is selective for the **middle** Ln, but for heavier Ln in isooctane [V Shimojo 2008, doi:10.1039/b810277p]. C3 has 0 ionic-liquid cells [D].

**Prediction for four reads.** log D flattens or turns over after a class-specific point. The constant continuation of PWL_R beyond the last read is then roughly right for strongly crowded systems. Full linear continuation is right for systems that keep their slope; the classical case is organophosphorus cation exchangers, for example P507 Tm/Er 3.34, Yb/Tm 3.56, Lu/Yb 1.78 [S, US patent 12024756 snippet]. C3 contains essentially no cation exchangers (chem_family: no organophosphorus acid; phosphoryl 4 cells) [D].

**Data [D] (probes 1 and 3).**
- DOPT_Q reads the lightest measured metal in every C3 query, so there is **no light-end extrapolation** (0 rows).
- Heavy side, BPG: 11.9% of rows and 9.7% of B1's SSE lie beyond the last read, with R²(B1) = −0.012 there (Yb/Lu 627 and Tm/Yb 265 of about 1,070 rows over 5 seeds). Inside the hull, R²(B1) = 0.471.
- A damped continuation ŷ = B1 + d · s_last · Δx beyond the last read, where s_last is the slope between the last two reads in −r_CN8, gives:

| variant | ΔR² BPG | ΔR² BP | seeds positive |
|---|---|---|---|
| in-sample oracle, d = 0.375 | +0.0060 | +0.0060 | — |
| LOCO-chosen global d | +0.0051 | +0.0044 | BPG 4/5 (−0.0012, +0.0024, +0.0081, +0.0157, +0.0003); BP 3/5 |
| family d shrunk to pooled (K = 200 rows), LOCO | +0.0041 | +0.0049 | **5/5 on both** |

- Family in-sample optimal d: amide_other 0.75 (SSE ratio 0.61), DGA 0.625 (0.82), N-heterocyclic 0.25–0.375 (0.87–0.89). This is the literature's order: crowded N-donors saturate more.
- **Caveat.** d < 1 is the optimum even for an exactly linear true curve when s_last is noisy, because noise in the slope estimate shrinks the optimal continuation. This check cannot separate **physical saturation** from **statistical shrinkage**.
- **Evidence that saturation is physical in the mean.** The mean C3 slope per Å of CN8 radius collapses after Ho: Ho/Dy 0.10 / 0.012 Å ≈ 8.6 log/Å, Er/Ho ≈ 3.9, Tm/Er ≈ 0.7, Yb/Tm ≈ −2.7, Lu/Yb ≈ 1.1 [D, from the pair-type means in §2.2]. How d* splits between physics and shrinkage is still open.

**Verdict.** This is the one literature-predicted shape element that PWL_R + Gd mishandles and that shows a positive held-out sign. It is a small, cheap round-5 candidate: 1 parameter, or 2–3 with family shrinkage. Its expected size (+0.004 to +0.006) is below the +0.01 planning margin and far below the power floor in POWER.md.

### 3.4 Acidity and concentration effects

**Literature:**
- Solvating extractants: D ∝ [NO₃⁻]³ · [L]ⁿ · (aqueous nitrate complexation)⁻¹. Any **Ln-dependence of n** (the extracted stoichiometry) or of nitrate complexation makes SF depend on [L] and [HNO₃] [B].
- Ln–nitrate stability constants are hard to compute. Cluster-continuum DFT needs an empirical LFER correction and involves both CN 8 and CN 9. Universal machine-learned potentials (MACE-MP0, MACE-MATPES-R2SCAN) **do not reproduce the detailed Ln trend** [V Dinpajooh 2026, doi:10.1039/d6cp00838k].
- TBP: the Ln order is often stated to reverse with HNO₃ concentration [B; Hesford, Jackson & McKay 1959, J. Inorg. Nucl. Chem. 9:279, content not verified].
- Synergistic and holdback systems are tuned by pH (Augustine 2026; 3.5).

**Project.** gen13 measured that 40% of amplitude variance within an extractant is condition-driven, but the coupling is extractant-specific. Acid slopes are heterogeneous (Q = 105.8 on 11 df, p = 1.3e−17), and a global condition→shape head **degraded** MAE by 0.061–0.079 [P STAGE2_REPORT §2].

**Prediction for four reads.** The reads already carry the cell's conditions. Conditions add information only if they predict **curvature or extrapolation** given the reads.

**Value beyond B1.** Not transferable on this corpus. In C3, acid ranges 0–7.4 M (median 1.5) and extractant 0–0.7 M (median 0.1) [D].

### 3.5 Aqueous complexation and holdback ("push–pull")

**Literature:**
- log D = log D₀ − log(1 + Σᵢ β_aq,i [Lᵢ]ⁿ) for each Ln [B; TALSPEAK review Nilsson & Nash 2007, SXIE 25:665, S].
- **Coupling HT DFT, automated experiments and Bayesian optimisation**: oxaloacetic acid as an aqueous holdback with HDEHP selectively extracts Eu, Dy and Ho over Nd at pH ≈ 2.0, and Eu from Dy/Ho at pH ≈ 0.5. Optimisation gives a **4× SF increase** over HDEHP alone [V Augustine et al. 2026, doi:10.1021/jacs.6c04301].
- **Reverse-size-selective aqueous complexants**: macrophosphi's La affinity exceeds Gd's by over 10⁵, with biphasic SF(Ce/La) up to 45 [S, Thiele, Fiszbein, Woods & Wilson, Inorg. Chem. 2020, file acs.inorgchem.0c02413].

**Shape prediction for four reads.** The holdback's own log β series is subtracted from log D:
- With EDTA (+4.3 log from La to Lu, and a Gd dip of −0.27) as the holdback, the cell's slope is reduced.
- If the complexed fraction is large, the **Gd term flips sign** (a Gd bump in log D).
- DTPA-like ligands, which plateau or fall in the heavy Ln [B], would create a heavy-end **turnover**.

This correction is exact in form and known in magnitude from NIST/IUPAC log β tables.

**Data [D, P].**
- 19 C3 cells have an aqueous complexant or salt (DTPA/NaNO₃ 7, NaNO₃ 4, …) and the frozen bench cannot see them [P PROVENANCE.md].
- Cells in carboxylic-acid media (all DGA): malonic 8, lactic 4, tartaric 4, citric 3. Their Gd chords are −0.02 to −0.09 (n 3–6): no sign flip, but these acids are weak Ln complexants at the reported acidities.

**Value beyond B1.** Physically mandated and potentially large in holdback cells, but untestable here with n = 7. It is a design rule for new data (record complexant identity and free concentration) and for the round-5 data contract: exclude or explicitly model complexant cells. It is not a modelling lever on C3.

---

## 4. Third phase and aggregation

**Literature:**
- TODGA extraction reflects the cations' affinity for **preformed reverse-micelle tetramers** (Nd only) [V Jensen, Yaita & Chiarizia 2007, doi:10.1021/la0631926].
- Am extraction correlates with TODGA aggregation, and Eu promotes aggregation [V Pathak 2010, doi:10.1016/j.jcis.2009.10.015].
- Mixed-extractant aggregate morphologies are non-additive [V Ellis 2013, doi:10.1021/jp401025e].
- n-Octanol dilutes aggregates and only penetrates their shell [V Lu, Dourdain & Pellet-Rostaing 2020, doi:10.1021/acs.langmuir.0c01554].
- Aromatic diluents lower D by about 10× [S, KIT].
- No source found compares light vs heavy Ln in aggregation or LOC.

**Prediction for four reads.** Aggregation mostly shifts the **level** common to all Ln, which cancels in log SF. It bends the series only if aggregate hydration or stoichiometry differs across the series (Baldwin, 3.2) or near loading limits.

**Data [D].** C3 metal concentration has median 0.06 mM, 75th percentile 0.6 mM and maximum 40 mM (193 cells with a value). Most cells are at trace loading.

**Verdict.** Not a round-5 lever. It is a curation flag for macro-loading cells only.

---

## 5. Steric bite, rigidity, preorganisation and size selectivity

**Literature:**
- **Hancock's chelate-ring rule**: five-membered chelate rings favour **large** ions and six-membered rings favour small ions. It may fail for rigid macrocycles [S, Hancock-group PhD thesis, Wits 1989; B Hancock & Martell, Chem. Rev. 1989, 89:1875].
- **bp18c6** (diaza-18-crown-6 with picolinate arms) [V Roca-Sabio 2009, doi:10.1021/ja808534w]:
  - log K(Ce) − log K(Lu) = **6.9**: strongly light-selective;
  - DFT shows a **conformational switch** of the macrocycle around Gd–Lu.
- **Macrodipa / macrotripa**: **dual** selectivity for both large light and small heavy Ln, i.e. a **U-shaped** log K vs radius. This comes from a toggle between a 10-coordinate and an 8-coordinate conformation, set by ligand strain against binding energy [V Hu, MacMillan & Wilson 2020, doi:10.1021/jacs.0c05217].
- **CHX-macropa**: a rigid core enhances large-over-small selectivity [S, Inorg. Chem. 2019, 58:10483].
- **DPP** (2,9-di(pyrid-2-yl)-1,10-phenanthroline): log K1 **peaks at Sm**, and molecular mechanics puts the strain minimum at Pm [V Carolan 2013, doi:10.1021/ic3002509].
- **BLPhen**: "unparalleled" light-Ln selectivity, modulated by **molecular rigidity** [V Healy et al. 2019, doi:10.1002/chem.201806443].
- **Lanmodulins** [V Mattocks 2023, doi:10.1038/s41586-023-05945-5; V Diep 2026, doi:10.1038/s41589-026-02176-3]:
  - Hans-LanM's La-induced dimer binds over **100× more tightly** than its Dy-induced dimer, giving single-stage Nd/Dy to over 98% purity.
  - A 96-well assay of **621 LanM orthologs × 15 REEs** found **eight distinct selectivity profiles**.
- **Project.** Tighter donor bite → heavy-selective: Spearman(amplitude, frac_donor_pairs_within_3) = −0.53 [P gen13 stage 3]. DGAs (two five-membered chelate rings) are heavy-selective. That is the **opposite** of what Hancock's five-ring rule alone predicts within Ln, which suggests that the electrostatic/charge-density term dominates ring strain for these flexible ligands [interpretation].

**Shape prediction for four reads.** Preorganised or rigid ligands produce **interior extrema** (a peak at the size match) and **U-shapes** (a conformational toggle). PWL through reads that bracket a peak underestimates the peak and mis-signs the adjacent pairs near it. The extremum's position is a property of the cavity or bite (DPP: Sm; bp18c6 switch: Gd–Lu).

**Project.** 27% of well-determined cells have an interior extremum. A curve "alphabet" oracle (SHAPE4) is worth +0.056, but it is **not predictable from the 2D graph** [P gen15].

**Value beyond B1.** Potentially the largest shape effect in the literature, but it lives in chemotypes almost absent from C3 (by cells: DGA 157/229, N-heterocyclic 33, amide 21, malonamide 8). It is an acquisition target, not a modelling lever. The LanM work (eight selectivity profiles from 621 variants) is an external analogue of gen15's alphabet: a small set of **discrete** shape classes, which a classifier can learn only with hundreds of labelled variants.

---

## 6. Recent ML / QSPR for Ln selectivity (2019–2026): features, data scale, results

| study | target / data | features | split | headline result | relevance |
|---|---|---|---|---|---|
| Liu, Johnson, Jansone-Popova & Jiang, JACS Au 2022 [V, doi:10.1021/jacsau.2c00122] | 1,202 log D, 109 ligands, 14 Ln (> 60 rows each), log D −4…+4 | 2,048 ECFP bits + 208 RDKit + 14 per-Ln + conditions (2,291 inputs); MLP | **random** (117 held out) | R² 0.85, RMSE 0.53, MAE 0.34. Four new DGAs: R² 0.78–0.92, MAE 0.21–0.41. Stated uncertainty about 0.5 log. **No SF evaluation** | data scale ≈ ours (109 vs 79 extractants) |
| Harish, ACS Omega 2026 [V, doi:10.1021/acsomega.6c07482] | same 1,202 log D, 93 ligands, 14 Ln, **45 sources** | RF and others; conformal calibration | row-level vs **leakage-controlled grouped** | RF R² 0.94 (row split) collapses to **0.20 … −0.03** grouped. Active learning no better than random for prediction. "High log D does not imply selectivity" | external replication of this programme's BP/BPG lesson |
| Chaube, Goverapet Srinivasan & Rai, Sci. Rep. 2020 [V, doi:10.1038/s41598-020-71255-9] | 6,583 log K1 (IUPAC), 698 ligands, 15 Ln cations, 8 media | 83 RDKit + 14 metal properties (incl. radius) + medium | random 10-fold | AdaBoost MAE 0.39. Six out-of-sample N-donors: MAE 0.95. Adjacent-pair differences: MAE 0.11 (Tb–Dy) to 0.34 (Ce–Pr). For ADPTZ the predicted series is almost **flat** (4.6–4.8) against experimental 4.0–4.6 | metal descriptors give a smooth level, but the SF is not learned |
| Mitrofanov et al., Molecules 2021 [V, doi:10.3390/molecules26113237] | 82–324 log K per Ln | Morgan FP; MLP; SHAP | stratified 80/20, 10-fold | R² ≈ 0.76, RMSE about 2.5 log K. Fragment contributions differ by Ln (soft aromatic N only for La) | descriptive only |
| Solov'ev & Tsivadze (≈ 2020, 2022) [S/V metadata] | 2,854 log β1 / 445 ligands; 947 log β2 / 156 ligands; 13 metals | per-metal "thermodynamic radius" proportionality | CV | error 0.3–1.0 log depending on radius gap | its separable assumption fails for the Gd anomaly in C3 [D, §1.2] |
| Unnamed, Sep. Purif. Technol. 2026, pii S1383586626013559 [S, authors not verified] | 3,343 REE SX points, 14 Ln | CatBoost + SHAP | not verified | R² 0.96, MAE 0.14. SHAP: **conditions outweigh extractant** | likely a row-level split (unverified) |
| Udawattha & Alam 2026 [S, doi:10.1016/j.gsme.2026.03.001] | curated REE SX equilibria | ANN on pH and extractant concentration | — | "strong agreement" | process-level only |
| Augustine et al., JACS 2026 [V] | HT experiments + DFT + multi-objective BO | holdback-agent screening | prospective | 4× SF gain | HT experimentation as the source-diversity lever |
| Zhang B. et al., JACS 2026 [V, doi:10.1021/jacs.5c19738] | SAFE-MolGen LLM agent + supervised ML on SAFE + automated 3D/QM | — | prospective (computational) | new ligands predicted to beat CyMe₄BTBP for Am/Eu | uses SAFE; Am/Eu, not adjacent Ln |
| Sumiyoshi, Kuroki & Mori, Inorg. Chem. 2026 [V] | DFT database, terpyridines × Am, Cm, Eu, Gd | compact electronic descriptors; GPR + SHAP | — | MAE 17.5 kJ/mol ΔG_bind; **2.6 kJ/mol ΔΔG (An–Ln)** | computed precision ≈ 0.45 log, coarser than adjacent Ln |
| Shapiro et al., Chem. Sci. 2026 [V, doi:10.1039/d6sc03531k] | 19 REE-DOTAM crystal structures + DFT + ML | template geometry | — | heavy/light selectivity attributed to hydration energies (fixed templates) or twist angles (relaxed). Energies 5–15 kJ/mol "comparable to DFT uncertainty" [S] | DFT resolution limit |
| Li, Jansone-Popova & Jiang, Sci. Rep. 2024 [V, doi:10.1038/s41598-024-62074-3] | CSD: 49,472 Ln structures (27,858 mononuclear; 2,226 phen) | — | — | CN and donor statistics as groundwork for generative design | structural prior source |
| Yang X. et al., JACS Au 2024 [V, doi:10.1021/jacsau.4c00684] | DFT Am/Eu for three ligands | EDA/NOCV | — | ranking matches experiment; Am/Eu SF 51–57 | not adjacent Ln |
| Hou et al., J. Phys. Chem. A 2025 [V, doi:10.1021/acs.jpca.5c06349] | NTAamide An/Ln pairs | DFT + ML retrosynthesis | — | "Difference" vs "Similarity" models for same-group vs same-period pairs | qualitative |
| SAFE (LANL / Utah / Fordham) [V site, safe.lanl.gov] | 8,716 records, 52 extractants, 39 metals (site snapshot at search time) | — | — | community database for ML | needs login (memory note); overlap audit pending (round-4 DATA_REQUEST) |

**What the ML literature implies for round 5:**
1. **Nobody has solved this task.** No paper reports adjacent log SF skill on held-out sources. The best log D models (MAE 0.14–0.34 on random splits) have errors at or above the adjacent signal (≈ 0.17). Under grouped splits they approach R² ≈ 0. Their per-Ln errors would have to be strongly correlated within a cell to give useful SFs, and no paper tests that.
2. **Data scale.** All published Ln log D sets are of the same order as ours: about 100 ligands and about 45 sources. None shows that more rows of the same kind fix transfer. The scale-changing efforts are **high-throughput and automated experiments** (Augustine 2026; 621 proteins × 15 REEs in Diep 2026), not literature curation.
3. **Features.** Metal descriptors (radius, ionisation potentials, hydration) only parameterise a smooth level. Every SHAP-style finding (conditions > extractant; hydration and twist angles; IP and charges) is in-distribution. Three-dimensional or DFT selectivity features need **≤ 0.5 kJ/mol relative precision** for adjacent Ln, while DFT ΔΔG errors are 2.6–15 kJ/mol. Only isostructural, composition-matched series with systematic error cancellation could reach that (gen15's composition trap [P]).

---

## 7. Synthesis: can each item add information beyond PWL_R + Gd, given four reads?

| # | literature item | predicted shape given four reads | already in B1? | tested in the programme | check in this note [D] | round-5 verdict |
|---|---|---|---|---|---|---|
| 1 | lanthanide contraction (quadratic in Z) | smooth in r; radius steps shrink to Lu | yes (radius coordinate) | PWL_R > PWL_Z, PCHIP_R −0.018 | — | nothing to add |
| 2 | electrostatic / proportional LFER | anomalies scale with ligand strength | n/a | — | Gd not amplitude-scaled; scaled term LOCO −0.017 | reject proportional anomaly priors |
| 3 | Gd break / hydration change | single-element Gd dip; Gd/Eu small, Tb/Gd large | **yes** | B1 +0.034; S01 = B1 | chord −0.115 (−0.05 DGA to −0.21 N-het); family scale LOCO −0.001…−0.004 | keep B1 additive; no family scale |
| 4 | tetrad / double-double / inclined W | convex tetrads: first pair of each tetrad larger SF; Nd, Gd, Lu low | partly (Gd) | S02 −0.024; B2 −0.022; gen13 ≤ 0.006 | pair-type constants oracle +0.012, LOCO −0.018 | do not reopen |
| 5 | nephelauxetic / covalency (N vs O donor) | larger tetrads and Gd dips, more heavy-end fall-off for N-donors | no | S01 (donor contrast) = B1 | N-het Gd −0.21, heavy damping strongest | no transferable lever; Y as a probe needs data |
| 6 | hydration enthalpies | metal-intrinsic, additive irregularities | implicitly | gen13 hydration basis (not separately useful) | precision ±10 kJ/mol ≫ signal | no feature use |
| 7 | outer-sphere water/nitrate; alkyl branching; stereo | cell-specific shape within a class | no | S05/S06 (descriptors, 3D) failed | — | curation (stereo), not modelling |
| 8 | heavy-end crowding / coordination loss | saturation or turnover after a class-specific point | **no** (constant continuation) | HBK@ML smoother extrapolation, 0.419 vs PWL_R 0.418 (Yb/Lu worse) | damped continuation +0.004…+0.006; 5/5 seeds with family shrinkage | **small candidate** (register, 1–3 parameters) |
| 9 | aqueous complexation / holdback | subtract the holdback's log β series; Gd sign can flip | no | bench blind to complexant (19 cells) | carboxylic-acid cells n = 3–8, no flip | data contract: exclude or model; acquisition |
| 10 | acidity / ligand concentration (stoichiometry changes) | SF depends on conditions | in the reads | gen13: condition→shape head hurts | — | no |
| 11 | aggregation / third phase | mostly level; bends only near LOC | n/a | — | C3 mostly trace | curation flag |
| 12 | size selectivity / rigidity / preorganisation | interior extrema, U-shapes, extremum at the size match | no | gen15: alphabet oracle +0.056, not predictable | — | acquisition of rigid chemotypes; no lever on C3 |
| 13 | ML log D models | smooth level; SF not learned | n/a | gen13–gen20 (CatBoost, encoders) | — | do not expect an off-the-shelf model to beat B1 |

---

## 8. Round-5 hypotheses that follow from the literature, each with a falsification test

**H1 (small, cheap): damped heavy-end continuation on B1.**
- **Form.** Beyond the last read, ŷ(m) = B1(m) + d · s_last · (x_m − x_last), with x = −r_CN8. d is fitted in the outer training folds with an inner CV over {0, 0.125, …, 1}; d = 0 recovers B1. An optional variant shrinks d by donor family toward the pooled value.
- **Literature.** Coordination crowding and loss for small Ln (DMDOHEMA, N-donors); DGA heavy plateau; the slope is kept by cation exchangers (absent here).
- **Expected size.** +0.004 to +0.006 R² [D]. It should also improve the Tm/Yb and Yb/Lu pair R² (≈ 0 now).
- **Falsification.** The inner-CV d is 0 in most folds, or ΔR² ≤ 0 on BPG. Use the round-4 controls; LOCO is already positive, so the sealed version is a real test. Run a **noise-matched control**: simulate exactly linear tails with the observed read noise. If d* ≈ 0.4 there too, the gain is slope shrinkage, not physics, and should not be described as saturation.
- **Interaction.** HBK@ML already extrapolates slope partially and made Yb/Lu worse (−0.07 → −0.14 [P round-2 ATTEMPTS]). Damping on top of B1's PWL is a different, more conservative operator.

**H2 (data contract): aqueous-complexant cells.** Either exclude the 19 complexant cells from model fitting, or subtract a tabulated log β series of the holdback (NIST 46 / IUPAC) before interpolation. With n = 7 DTPA cells this is a **correctness** rule, not a scored hypothesis. The prediction to check on new data: a holdback with a known Gd dip turns B1's Gd dip into a bump.

**H3 (acquisition): yttrium as a free fifth reading.** In new sources, request log D(Y) wherever it was measured. Test whether Y's deviation from the Ln radius curve predicts the cell's Gd/tetrad amplitude, which literature 2.4 predicts. Untestable now (49 Y rows / 4 DOIs).

**H4 (acquisition): rigid and preorganised chemotypes.** Prioritise 14-Ln series for macrocycles, phenanthroline diamides and BLPhen, and N-donors. The literature shows interior extrema and U-shapes there. Round-4 PROVENANCE already lists open-access 14-Ln tables, for example Molecules 29:3548 (2024), a phenanthroline diamide with a numeric SI table [P].

**H5 (do not run): universal or family tetrad, inclined-W or double-double terms; amplitude-scaled anomalies; hydration-energy or DFT-ΔΔG features without isodesmic error cancellation; family-specific Gd scales.** Each is either measured dead in the project or fails LOCO here [D].

---

## 9. Open hypotheses and unresolved points

- **Peppard's original extraction tetrads.** Their sign and magnitude in log units were not obtained (only the ΔE3/ΔE1 ratios). If the original HDEHP/HCl tetrads were large (≥ 0.1 log), the small C3 tetrads would reflect **O-donor neutral extractants** specifically, not a general property.
- **Gd chord vs B1 coefficient.** The Gd chord (−0.115) is larger than B1's fitted term (−0.072). Part of the gap is shrinkage, and part is the chord estimand: Eu and Tb are both pushed up by +0.049 and +0.067, consistent with a pure Gd point dip of about −0.11 to −0.12. Whether a less-shrunk Gd coefficient is better out of chemotype was not tested here. B1's inner CV already tunes the shrinkage.
- **Family-dependent heavy-end damping.** Whether this is physical (N-donor crowding) or a property of each family's read placement (where s_last is estimated) is unresolved.
- **Convex first tetrad.** The first-tetrad convexity (Ce, Pr +0.064 above the La–Nd chord) explains B1's largest per-pair biases (Ce/Pr −0.054). A one-parameter "La-anchored bulge" is the only untested tetrad-type item, and S02's failure argues against it.
- **TBP acid-dependent reversal of the Ln order.** Stated from background and not verified. It is irrelevant to C3 unless TBP cells exist.
- **Unverified ML paper.** The CatBoost 3,343-point paper's authors and split are unverified.

---

## References (cited above; [V] = abstract or metadata verified this session)

**Contraction, radii, hydration**
- Quadrelli E. A. *Inorg. Chem.* 2002, 41, 167–169. doi:10.1021/ic015580v [V]
- Seitz M., Oliver A. G., Raymond K. N. *J. Am. Chem. Soc.* 2007, 129, 11153–11160. doi:10.1021/ja072750f [V]
- Raymond K. N. et al. *C. R. Chimie* 2010. doi:10.1016/j.crci.2010.03.034 [S]
- D'Angelo P. et al. *Inorg. Chem.* 2011, 50, 4572–4579. doi:10.1021/ic200260r [V]
- Lundberg D., Persson I. et al. *Inorg. Chem.* 2010, 49, 4420. doi:10.1021/ic100034q [V]
- Persson I. et al. *Chem. Eur. J.* 2008, 14, 3056–3066. doi:10.1002/chem.200701281 [V]
- Sobolev B. P., Sulyanova E. A. *Int. J. Mol. Sci.* 2023, 24, 17080. doi:10.3390/ijms242317080 [V]
- Kepp K. P. *J. Phys. Chem. A* 2019, 123, 6536. doi:10.1021/acs.jpca.9b05140 [V]; 2018, 122, 7464. doi:10.1021/acs.jpca.8b06674 [V]

**Gd break, tetrad, double-double, inclined W, nephelauxetic**
- Peppard D. F., Mason G. W., Lewey S. *J. Inorg. Nucl. Chem.* 1969, 31, 2271 [S]
- Fidelis I., Siekierski S. *J. Inorg. Nucl. Chem.* 1966; 1971, 33, 3191 [S]
- Siekierski S. *J. Inorg. Nucl. Chem.* 1971, 33, 377 [S]; unit-cell double-double magnitude vs electronegativity, Springer doi:10.1007/BF02204319 [V abstract]
- Jørgensen C. K. *J. Inorg. Nucl. Chem.* 1970, 32, 3127 [S]; Nugent L. J. *J. Inorg. Nucl. Chem.* 1970, 32, 3485 [S]
- Kawabe I. *Geochem. J.* 1992, vol. 26, "Lanthanide tetrad effect in the Ln³⁺ ionic radii and refined spin-pairing energy theory" [V title and year; page number from memory, B]; Kawabe I. et al. *Geochem. J.* 1999, 33, 249–265 and 267–275 [V]
- Kawabe I., Masuda A. *Geochem. J.* 2001, 35, 215–224. doi:10.2343/geochemj.35.215 [V]
- Kawabe I. preprint, Ln–EDTA Gd break: nagoya.repo.nii.ac.jp/records/18232 [S]
- Sinha S. P. (ed.) *Systematics and the Properties of the Lanthanides*, NATO ASI C109, 1983. doi:10.1007/978-94-009-7175-2 [S]
- Verma & Saxena 1989 (tetrad stronger at f⁷ than at f³–f⁴), Springer BF02840595 [S]
- Limaye S. N., Saxena M. C. *J. Indian Inst. Sci.* 1994, 74(5) [V]
- Löble M. W. et al. *J. Am. Chem. Soc.* 2015, 137, 2506. doi:10.1021/ja510067v [V]
- O'Brien et al. *Phys. Chem. Chem. Phys.* 2022. doi:10.1039/d2cp01081j; correction doi:10.1039/d3cp90217j (EDTA log K from Martell & Smith, *Critical Stability Constants*, 1974) [S]

**Extraction mechanism, outer sphere, crowding, thermodynamics**
- Baldwin A. G. et al. *ACS Cent. Sci.* 2018, 4, 739–747. doi:10.1021/acscentsci.8b00223 [V]
- Stamberga D. et al. *Inorg. Chem.* 2020, 59, 17620–17630. doi:10.1021/acs.inorgchem.0c02861 [V]
- Wilden A. et al. *Chem. Eur. J.* 2019. doi:10.1002/chem.201806161 [V]
- Shimojo K. et al. *Dalton Trans.* 2008. doi:10.1039/b810277p [V]
- Sasaki Y. et al. *Anal. Sci.* 2020 (TODGA Dy/Nd SF > 20). doi:10.2116/analsci.20p129 [V]
- Wang X., Peroutka A. A., Kravchuk D. V., Shafer J. C., Wilson R. E., Servis M. J. *Chem. Sci.* 2024, 15, 16494. doi:10.1039/d4sc05061d [V]
- Wang X. et al. *Inorg. Chem.* 2025 (NTAamide). doi:10.1021/acs.inorgchem.5c02125 [V]
- Grimes T. S., Zalupski P. R., Martin L. R. *J. Phys. Chem. B* 2014, 118, 12725. doi:10.1021/jp507727v [V]
- Ibrahim S. M. et al. *ACS Omega* 2019 (HDEHDGA: SF Ce/La 6.68, Pr/Ce 2.79, Nd/Pr 2.65). doi:10.1021/acsomega.9b03241 [V/S]
- Ustynyuk Y. A. et al. *Chem. Commun.* 2015. doi:10.1039/c5cc01620g [V]
- Yang X. F. et al. *Inorg. Chem.* 2021. doi:10.1021/acs.inorgchem.1c01035 [V]
- Kolarik Z., Müllich U., Gassner F. *Solvent Extr. Ion Exch.* 1999, 17, 1155 [S]
- Dinpajooh M. et al. *Phys. Chem. Chem. Phys.* 2026. doi:10.1039/d6cp00838k [V]
- Hesford E., Jackson E. E., McKay H. A. C. *J. Inorg. Nucl. Chem.* 1959, 9, 279 [S citation only]
- Nilsson M., Nash K. L. *Solvent Extr. Ion Exch.* 2007, 25, 665–701 [S]
- Augustine L. J. et al. *J. Am. Chem. Soc.* 2026. doi:10.1021/jacs.6c04301 [V]
- Industrial adjacent SFs: P507 Tm/Er 3.34, Yb/Tm 3.56, Lu/Yb 1.78 (US 12024756) [S]; HDEHP–P350 with acetic acid, β Ce/La 4.09, Pr/Ce 1.96, Nd/Pr 1.53, Er/Ho 1.45, Tm/Er 2.58, Yb/Tm 2.77, Lu/Yb 1.77 (J. Rare Earths 2021, pii S100207212100123X) [S]; HDEHP adjacent SF ≈ 2.5 (Nature 194, 84, doi:10.1038/194084a0) [S]

**Aggregation**
- Jensen M. P., Yaita T., Chiarizia R. *Langmuir* 2007, 23, 4765. doi:10.1021/la0631926 [V]
- Pathak P. N. et al. *J. Colloid Interface Sci.* 2010, 342, 114. doi:10.1016/j.jcis.2009.10.015 [V]
- Ellis R. J. et al. *J. Phys. Chem. B* 2013, 117, 5916. doi:10.1021/jp401025e [V]
- Lu Z., Dourdain S., Pellet-Rostaing S. *Langmuir* 2020, 36, 12121. doi:10.1021/acs.langmuir.0c01554 [V]

**Size selectivity, preorganisation, proteins**
- Hancock R. D., Martell A. E. *Chem. Rev.* 1989, 89, 1875 [B]; Hancock-group thesis (Wits 1989) [S]
- Roca-Sabio A. et al. *J. Am. Chem. Soc.* 2009, 131, 3331. doi:10.1021/ja808534w [V]
- Hu A., MacMillan S. N., Wilson J. J. *J. Am. Chem. Soc.* 2020, 142, 13500. doi:10.1021/jacs.0c05217 [V]
- Thiele N. A. et al. *Inorg. Chem.* 2019, 58, 10483 (CHX-macropa) [S]; Thiele, Fiszbein, Woods, Wilson *Inorg. Chem.* 2020 (macrophosphi) [S]
- Carolan A. N. et al. *Inorg. Chem.* 2013. doi:10.1021/ic3002509 [V]
- Healy M. R. et al. *Chem. Eur. J.* 2019. doi:10.1002/chem.201806443 [V]
- Mattocks J. A. et al. *Nature* 2023, 618, 87. doi:10.1038/s41586-023-05945-5 [V]
- Diep P. et al. *Nat. Chem. Biol.* 2026, 22, 829. doi:10.1038/s41589-026-02176-3 [V]

**ML / QSPR**
- Liu T., Johnson K. R., Jansone-Popova S., Jiang D.-E. *JACS Au* 2022, 2, 1428. doi:10.1021/jacsau.2c00122 [V]
- Harish K. *ACS Omega* 2026, 11, 48692. doi:10.1021/acsomega.6c07482 [V]
- Chaube S., Goverapet Srinivasan S., Rai B. *Sci. Rep.* 2020, 10, 14322. doi:10.1038/s41598-020-71255-9 [V]
- Mitrofanov A. A. et al. *Molecules* 2021, 26, 3237. doi:10.3390/molecules26113237 [V]
- Solov'ev V., Tsivadze A. Yu. *Comments Inorg. Chem.* 2022, 43, 16. doi:10.1080/02603594.2022.2087637 [V metadata]; "Thermodynamic radii of lanthanide ions…" (istina record) [S]
- Solov'ev V. P. et al. *Mol. Inform.* 2018 (HOMO/LUMO of N-donor heterocycles for Ln separation). doi:10.1002/minf.201800025 [V metadata]
- Sumiyoshi T., Kuroki N., Mori H. *Inorg. Chem.* 2026, 65, 7676. doi:10.1021/acs.inorgchem.5c05680 [V]
- Zhang B. et al. *J. Am. Chem. Soc.* 2026, 148, 5520. doi:10.1021/jacs.5c19738 [V]
- Shapiro N. M. et al. *Chem. Sci.* 2026, 17, 17370. doi:10.1039/d6sc03531k [V]
- Li S., Jansone-Popova S., Jiang D.-E. *Sci. Rep.* 2024, 14, 11301. doi:10.1038/s41598-024-62074-3 [V]
- Yang X. et al. *JACS Au* 2024, 4, 4744. doi:10.1021/jacsau.4c00684 [V]
- Hou Y.-C. et al. *J. Phys. Chem. A* 2025, 129, 11552. doi:10.1021/acs.jpca.5c06349 [V]
- Sep. Purif. Technol. 2026, pii S1383586626013559 (CatBoost, 3,343 points) [S]
- SAFE database: https://safe.lanl.gov [V site]

**Project files used**
- `generations/gen21/round4_20261022/`: REPORT.md, POWER.md, PROVENANCE.md, EVIDENCE_INVENTORY.md, `results/REF/per_pair_BPG.csv`, `results/S01/TABLE.md`, `predictions/REF_BPG.parquet`, `predictions/REF_BP.parquet`, `r4/refs.py`
- `generations/gen21/ATTEMPTS.md` (round 2) and `generations/gen21/next_research_20261019/ATTEMPTS.md` (round 3)
- `generations/gen13_separation/analysis/stage2/D7/FINDINGS.md`, `.../stage2/STAGE2_REPORT.md`, `gen13sep/metals.py`, `manifests/cohort_exact.parquet`
- memory notes for gen13 stage 2/3 and gen15

---

## Appendix A: the data checks behind the [D] items (exploratory; seen dev data)

**Status.** All checks reuse round-4 REF out-of-fold predictions (5 dev seeds × 1,802 strict-adjacent rows, k = 3, DOPT_Q, C3; BPG and BP) and the frozen cohort manifest. These seeds and folds have been seen many times, so the numbers are **exploratory development evidence**, not confirmation.

**What "LOCO" means here.** The parameter for each chemotype's rows is chosen on the pooled OOF rows of all *other* chemotypes. The held-out chemotype never contributes to its own parameter. B1's own β was fitted inside the original outer folds.

**Scripts.** They live in the session scratchpad and were not written to the project:
- `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e4_probe.py`
- `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e4_probe2.py`
- `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e4_probe3.py`
- `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e4_probe4.py`

Each ran in one process for under 1 minute.

**Definitions.**
- R² is the protocol's mean over seeds of per-seed pooled R². x = −r(Shannon CN8), as in PWL_R.
- **Region.** A row is "beyond_heavy" if B's index exceeds the largest read index, "beyond_light" if A's index is below the smallest read index, and "inside" otherwise.
- **Damped continuation.** s_last = (logD(last read) − logD(second-last read)) / (x_last − x_2nd). The offset for m beyond the last read is d · s_last · (x_m − x_last). The pair increment is offset(A) − offset(B), added to B1. Read log D values come from the cohort manifest; they are legal, being the four measured metals.
- **Chord deviation** of metal m between neighbours a and b: logD(m) minus the linear-in-x interpolation from a and b, on cells where all three are measured.
- **Family**: the cohort's `chem_family`.
- **O1, O2, O3.**
  - O1: per-pair-type mean residual added to B1.
  - O2: per (family, pair-type) mean residual added to B1.
  - O3: B0 + c_family · (B1 − B0).

  Each was run in sample (oracle) and LOCO. In probe 3, a shrinkage pseudo-count K pulls family estimates toward pooled ones.
- **Amplitude scaling (probe 4).** B0 + c · f · (B1 − B0), with f = |read slope| / median or the signed read slope / median, run in sample and LOCO.

**Results.**

*Probe 1:*
- **Regions.** BPG: beyond_heavy 11.9% of rows, 9.7% of SSE, R²(B1) −0.012, mean |y| 0.093; inside R²(B1) 0.471, mean |y| 0.174. BP is similar (12.3% / 10.1%). There are no beyond_light rows.
- **Damping grid (BPG).**
  - Heavy side: d = 0 0.4521, 0.125 0.4552, 0.25 0.4572, **0.375 0.4581**, 0.5 0.4579, 0.625 0.4567, 0.75 0.4544, 0.875 0.4511, 1 0.4466, 1.25 0.4345.
  - Light side: flat (no rows).
  - LOCO global: BPG +0.0051, BP +0.0044.
- **Family heavy-side oracle d:** amide_other 0.75 (n = 162 rows), DGA 0.625 (618), N-het 0.375 BPG / 0.25 BP (254 / 259).

*Probe 2:*
- O1 oracle: +0.0120 / +0.0111; O1 LOCO: −0.0182 / −0.0189.
- O2 oracle: +0.0731 / +0.0725.
- O3 oracle: +0.0141 / +0.0135. Scales: DGA 0.78, amide 1.71, N-het 2.15, phosphoryl 2.18, malonamide 0.52.
- Per-family B1 R² (BPG): DGA 0.451, N-het 0.424, amide 0.467, malonamide 0.149 (54 rows/seed), phosphoryl 0.751 (43 rows/seed).

*Probe 3 (LOCO with K = 0 / 50 / 200):*
- Family Gd scale: −0.0006 / −0.0028 / −0.0042 (BPG); −0.0012 / −0.0032 / −0.0044 (BP).
- Family × pair-type: −0.047 / −0.014 / −0.013.
- Family heavy damping: +0.0018 / +0.0029 / **+0.0041** (BPG) and +0.0033 / +0.0041 / **+0.0049** (BP). With K = 200, all 5 seeds are positive on both designs.

*Probe 4:*
- Gd chord vs slope: Spearman −0.129, p = 0.148, n = 127.
- First-tetrad bulge vs slope: Spearman +0.132, p = 0.164, n = 113.
- Gd × |amp|: c = 0.79, oracle ΔR² −0.0025, LOCO −0.0170 (BPG). Gd × signed amp: LOCO −0.0185. Amplitude-scaled pair template: LOCO −0.0174 / −0.0184. 0/5 seeds positive.

*Chord table (P3, all C3 cells; mean ± SE, n):* Ce|La–Pr +0.056 ± 0.022 (118); Pr|Ce–Nd +0.001 ± 0.019 (119); Ce+Pr|La–Nd +0.064 ± 0.013 (113); Eu|Sm–Gd +0.049 ± 0.012 (154); Gd|Eu–Tb −0.115 ± 0.013 (127); Tb|Gd–Dy +0.067 ± 0.009 (163); Ho|Dy–Er +0.024 ± 0.007 (172); Er|Ho–Tm +0.014 ± 0.009 (130); Yb|Tm–Lu −0.014 ± 0.005 (131); Tm|Er–Yb +0.015 ± 0.010 (131).

**Caveats.**
- Chord deviations are not independent: a pure Gd dip of δ raises the Eu and Tb chords by about −δ/2 each, and the observed +0.049 / +0.067 fit δ ≈ −0.11.
- Families are confounded with publications. sc009 (DGA) dominates: 65% of rows in one source cluster.
- None of these analyses is sealed. A round-5 registration of H1 must refit d strictly inside outer training folds.
