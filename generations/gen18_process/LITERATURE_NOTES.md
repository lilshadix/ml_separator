# Literature notes for the Pr/Nd case (collected 2026-09-13)

Every number below carries a **status**. The corpus (`dataset with 3D structures/`) holds no
PC88A, Cyanex 272 or D2EHPA rows, so the Pr/Nd case study is parameterised only from these
entries. Nothing here is a measurement made in this repository. Statuses:

- `literature_verified` — read on 2026-09-13 in an accessible source (patent full text, open-access
  article); the quoted passage is reproduced.
- `literature_cited_by_task_doc` — stated in `docs/kshot_next_steps_20260913.md` with a DOI; the
  full text was not accessible to verify (Elsevier abstracts are elided in Semantic Scholar,
  OpenAlex and Crossref; publisher pages return 403).
- `not_found` — searched, no accessible source; the database must carry a null and the case study
  must sweep an assumed range labelled `assumed`.

## 1. Sources actually read

### S1. Shin-Etsu patent EP2388344A1 (priority 2009-06-17), "Method for extracting and separating rare earth elements" — full text read on Google Patents. `literature_verified`

Verbatim passages:
- "DODGAA had a Nd/Pr separation factor of 2.5 whereas D2EHPA had a Nd/Pr separation factor of 1.23"
- "PC-88A had a Nd/Pr separation factor of 1.4 under the same conditions."
- "The kerosene solution of PC-88A effected a separation factor Nd/Pr of 1.4" — counter-current
  mixer-settler: **72-stage extracting section, 72-stage scrubbing section, 8-stage back-extracting
  section**.
- "The isododecane solution of DODGAA effected a separation factor Nd/Pr of 2.5" — 24-stage
  extracting, 24-stage scrubbing, 8-stage back-extracting section, 40 °C.
- Dy/Tb: DODGAA 1.3 vs PC-88A 2.1.

Use: PC88A Nd/Pr SF = 1.4 (kerosene, chloride-type feed as per the patent's examples; acidity and
loading of the comparison "same conditions" are not reproduced here — record them as unknown).
The 72/72/8 circuit is the **consistency target** for the cascade model: with SF 1.4, a
two-section (extraction + scrub) cascade of that size should reach high-purity Nd and Pr at high
recovery; the purity/recovery the patent achieved is not quoted in the passages read, so the check
is "the stage count for a >99 % / >99 % split at SF 1.4 is of the order of 70 + 70, not 7 + 7".

### S2. Arellano Ruiz, Kuchi, Parhi, Lee, Jyothi, Sci. Rep. 10, 16911 (2020), doi 10.1038/s41598-020-74041-9 — open access, read on PMC. `literature_verified`

Chloride medium; Exxsol D80 diluent; 1500 mg/L each of Dy, Tb, Nd; A/O 1:1; 298 K; 60 min.
- Extraction order: **D2EHPA > PC 88A > Cyanex 272**.
- Extraction at 0.8 mol/L extractant, initial pH 4.0:
  | metal | D2EHPA | PC 88A | Cyanex 272 |
  |---|---|---|---|
  | Dy | ~99 % | 98 % | 76 % |
  | Tb | ~99 % | 91 % | 61 % |
  | Nd | ~99 % | **64 %** | **27 %** |
- Equilibrium pH after extraction at 0.8 mol/L (initial pH 4.0): D2EHPA 0.76–1.29; PC 88A
  1.02–1.42; Cyanex 272 1.22–1.70. **This is the acid release by cation exchange under loading**
  (4500 mg/L total REE ≈ 0.03 mol/L; 3 H+ per metal ≈ 0.09 mol/L H+ released, consistent with pH
  falling from 4 to about 1.0–1.7). Any K back-calculated from these D values must use the
  *equilibrium* pH, not the initial one.
- Slopes of log D vs pH (H+ released per metal): D2EHPA Dy 2.68 / Tb 2.88 / Nd 2.95;
  PC 88A 2.60 / 2.50 / **2.22**; Cyanex 272 2.38 / 2.0 / **2.0**. Log D vs log[extractant] slopes:
  2–3 for all three. Mechanism written as nRE3+ + n(HA2)org ⇌ nRE(HA2)n,org + nH+.
- Diluents: aliphatic > aromatic (Exxsol D80 > n-heptane > cyclohexane > xylene > toluene).
- Thermodynamics at 0.4 mol/L: ΔH > 0 for all (endothermic); ΔG(298 K) for Nd: D2EHPA −5.96,
  PC 88A +0.42, Cyanex 272 +3.52 kJ/mol.
- Strip: 0.5 mol/L oxalic acid strips > 99.9 % Nd from loaded D2EHPA (4400 mg/L Nd), 2 stages at
  O/A 4:1, 11.7-fold enrichment; Nd oxalate → Nd2O3 (1173 K).
- **No Pr in this study; no Nd/Pr separation factor.**

Use: relative strength of PC88A vs Cyanex 272 for Nd in chloride at matched conditions
(D_Nd = 0.64/0.36 = 1.78 vs 0.27/0.73 = 0.37 at the *initial* condition set, but at different
equilibrium pH; back out K with the recorded equilibrium pH range and the recorded slope, and
carry the range as the uncertainty). Also the loading/acid-release evidence.

### S3. Afonin et al., Compounds 4(1) 172–181 (2024), doi 10.3390/compounds4010008 — abstract only (OpenAlex). `literature_verified` (abstract)

"the distribution ratios of all REEs decrease with increasing concentration of these metals in the
initial solution, which is associated with loading of the organic phase … The extractability of
REEs increases with atomic number of element, as is typical for acidic organophosphorus
extractants." P507 (= PC88A) + Cyanex 272 (1:1) mixtures, chloride. Full text blocked (403).

### S4. Tachimori, Sasaki, Suzuki, Solvent Extr. Ion Exch. 20(6), 687–699 (2002), doi 10.1081/SEI-120016073 — abstract via OpenAlex. `literature_verified` (abstract)

"0.1 M TODGA-n-dodecane … 0.008 M Nd(III) with an aqueous 3 M HNO3" (loading capacity / third-phase
limit); loading capacity depends on acidity, temperature, alkane chain length and aqueous anion;
DHOA > 0.5 M eliminates the Nd third phase; 0.2 M TODGA + 1 M DHOA loads highly with slightly lower
extractability. Temperature not in the abstract. Use: `phase.loc_metal_M = 0.008 mol/L` for the
corpus TODGA/nitrate/aliphatic system at 0.1 M TODGA / 3 M HNO3 (see
`addenda/ORCHESTRATOR_prefit_20260913.md`).

### S5. Nobley & Jensen, RSC Adv. 15, 43941 (2025), doi 10.1039/d5ra06417a — open access, read on PMC. `literature_verified`

TODGA/n-dodecane, 0.060–0.40 M, HNO3; contacts use equal volumes (O/A = 1) with the initial
aqueous concentration reported (e.g. 4.90 mM Nd). No numeric LOC of its own. Used only to
corroborate the O/A and metal-concentration convention assumed for corpus loading rows.

### E2 publication identities (abstracts only; details in the pre-fit addendum)

pub_5a68dc5665 = Sasaki et al. SEIE 2015 (doi 10.1080/07366299.2015.1087209, DGA loading
capacities); pub_0e7f3e0563 = Narayanan et al. ChemistrySelect 2022 (doi 10.1002/slct.202202610,
D3DODGA third phase); pub_d3c970567f = Sasaki et al. Anal. Chim. Acta 2005 (doi
10.1016/j.aca.2005.04.061, extraction-capacity method); pub_917a4583d4 = Metwally et al. JRNC 2010
(doi 10.1007/s10967-010-0641-2, Ce/TODGA); pub_15b174237f = Rama Swami et al. NJC 2018 (doi
10.1039/c8nj01074a, TEHDGA + HDEHP modifier).

## 2. Sources cited by the task document, not verifiable today

- **Banda, Jeon, Lee, J. Ind. Eng. Chem. 21, 436–442 (2014)**, doi 10.1016/j.jiec.2014.03.002,
  "Separation of Nd from mixed chloride solutions with Pr by extraction with saponified PC 88A and
  scrubbing". Task document: maximum SF only about 1.5; extraction combined with scrubbing gave an
  Nd recovery scheme. `literature_cited_by_task_doc`. Saponification degree, pH, concentrations,
  scrub solution: unknown.
- **Thakur, Jayawant, Iyer, Koppiker, Hydrometallurgy (1993)**, doi 10.1016/0304-386X(93)90084-Q,
  "Separation of neodymium from lighter rare earths using alkyl phosphonic acid, PC 88A". Task
  document: counter-current tests gave > 5 kg Nd2O3 at **97 % purity** with **> 85 % recovery**.
  `literature_cited_by_task_doc`. Stage counts, O/A, feed: unknown.
- **Liu, Jeon, Lee, Hydrometallurgy 150, 61–67 (2014)**, doi 10.1016/j.hydromet.2014.09.015,
  "Solvent extraction of Pr and Nd from chloride solution by the mixtures of Cyanex 272 and amine
  extractants". Task document: Cyanex 272 + Alamine 336 raised Pr and Nd extraction (web snippet:
  maximum synergistic factors 14.2 for Pr and 12.2 for Nd at Cyanex 272 mole fraction 0.5).
  Nd/Pr separation factor: **not accessible**.
- Liu, Jeon, Lee, Met. Mater. Int. (2015), doi 10.1007/s12540-015-5113-3 (Pr and Nd from La with
  Cyanex 272 + Alamine 336) — page requires a cookie handshake; not read.
- Kashi et al., J. Rare Earths 36, 317–323 (2018), doi 10.1016/j.jre.2017.09.016 (La, Pr, Nd with
  Cyanex 272 in kerosene + lactic acid) — abstract not accessible.
- Xie, Zhang, Dreisinger, Doyle, Miner. Eng. 56, 10–28 (2014), doi 10.1016/j.mineng.2013.10.021,
  the review with adjacent-pair SF tables — page 403; not read.

## 3. What the database may therefore contain for the case study

| quantity | value | status | source |
|---|---|---|---|
| PC88A, Nd/Pr SF, kerosene, chloride | 1.4 | literature_verified | S1 |
| PC88A, Nd/Pr SF, saponified, chloride, with scrubbing | ≈ 1.5 (max) | literature_cited_by_task_doc | Banda 2014 |
| D2EHPA, Nd/Pr SF, same conditions as S1 | 1.23 | literature_verified | S1 |
| Cyanex 272, Nd/Pr SF | **null** | not_found | — |
| Cyanex 272 vs PC88A strength for Nd, chloride, 0.8 M, pH_init 4, 1500 mg/L ×3 | 27 % vs 64 % extraction; eq. pH 1.22–1.70 vs 1.02–1.42 | literature_verified | S2 |
| pH slopes Nd | PC88A 2.22, Cyanex 272 2.0, D2EHPA 2.95 | literature_verified | S2 |
| Industrial PC88A Nd/Pr circuit | 72 ext + 72 scrub + 8 strip stages | literature_verified | S1 |
| Thakur PC88A counter-current result | 97 % Nd2O3, > 85 % recovery | literature_cited_by_task_doc | Thakur 1993 |
| Loading lowers D for acidic organophosphorus extractants in chloride | qualitative | literature_verified (abstract) | S3 |

For Cyanex 272 the case study must sweep an **assumed** Nd/Pr SF range and say so in every table.
The only defensible prior is "adjacent-pair selectivity of a phosphinic acid is not larger than
that of the phosphonic acid PC88A in chloride" (S2's slopes are lower, and reviews describe
Cyanex 272 as the weakest and least selective of the three for light lanthanides — a statement
found only in search snippets, `not_found` at source level). Recommended sweep: SF(Nd/Pr) ∈
{1.1, 1.2, 1.3, 1.4}, labelled `assumed`, with the PC88A comparison at 1.4 (verified) and 1.5
(cited).

## 4. Reproducibility of this note

URLs fetched (2026-09-13): doi.org redirects for the three Elsevier DOIs (403 at the publisher);
api.semanticscholar.org, api.openalex.org and api.crossref.org records for the same DOIs (abstract
elided/null); patents.google.com/patent/EP2388344A1/en (read); pmc.ncbi.nlm.nih.gov/articles/
PMC7547677/ (read); mdpi.com/2673-6918/4/1/8 (403); link.springer.com (cookie redirect);
jme.shahroodut.ac.ir/article_476.html (read: D2EHPA/Cyanex 272 mixtures, La/Gd/Nd/Dy in nitrate,
no Pr — Dy/Nd 720, Dy/Gd 3640 for the mixture).
