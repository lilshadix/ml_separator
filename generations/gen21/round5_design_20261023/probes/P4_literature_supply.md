# P4: Literature supply, a label-blind yield count (ideas M22 / M05 / the supply gate)

Written 2026-10-23 for the round-5 design. Probe P4. **Exploratory.** Nothing here is registered or sealed.

**What was and was not computed**
- Nothing was computed on the 229 C3 cells or on any D / log D / SF value.
- The only repository data read are bibliographic and metal-identity metadata:
  - the archive `dataset_all_metals/clean/master_clean.parquet` (DOI and author columns, metal symbol);
  - the round-4 cohort DOI table `round4_20261022/results/provenance/sources_by_doi.csv`.
- The scripts assert that no D column is loaded.

**Label-blindness**
- I never read a numeric D, log D or SF value from any source.
- Full texts were read through a summariser told to return only metadata.
  - No number came back.
  - Three answers carried a *qualitative* trend: P4-063 "D decreases with Z"; P4-018 and P4-022 "monotonic trend"; P4-051 tetrad analysis. See §8.

**Deliverables**
- `probes/P4_candidates.csv`: 92 screened candidates, 29 columns.
- This report.
- Support files: `probes/P4_corpus_registry.csv` (archive DOI / author registry) and `probes/P4_corpus_surnames.csv`.
- Scripts in `probes/scripts/` (P4_corpus_registry.py, P4_check_candidates.py); see §9.

---

## 0. Bottom line

1. **Screening funnel (92 candidates)**
   - 20 verified eligible (E1);
   - 32 probable (E2: the abstract says a lanthanide series or several Ln were measured, count not stated);
   - 19 undeterminable (U: no abstract, or the count is not given);
   - 16 excluded (X);
   - 5 already in the archive (C). These 5 are present only as Eu- or Nd-only records, a possible curation gap (§6.1).
2. **Literature eligibility is not scarce. Independence is.**
   - Of the 52 E1 + E2 papers, 23 come from laboratories already in the corpus (Y) and 5 from institution-linked ones (P).
   - The Moscow phen-diamide group alone accounts for 3 E1 papers (plus 1 U), all on the C3 chemotype.
3. **New independent laboratories (lab not represented anywhere in the archive)**

   | tier | new labs (conservative merge) | new labs (liberal) | new lab × chemotype units |
   |---|---|---|---|
   | E1 | **7** | 7 | 7 |
   | E1 + E2 | **16** | 19 | 22–23 |
   | E1 + E2 + U | 24 | 28 | 33–34 |

   - The E1 labs: KU Leuven (Binnemans), Chiba (Katsuta), Sofia UCTM (Atanassova), Mendeleev/Frumkin (Stepanov), ISSP RAS (Turanov), Osaka (Shinohara), CIAC (Deng).
   - **Expected yield after verification ≈ 13 labs (range 7–24).** This uses the E2 → E1 conversion seen in full-text checks: 3 of 6 abstracts that only claimed a "series" qualified, 95% CI 0.12–0.88 (§6.3).
   - Only **3** new labs pass M22's stricter rule (≥ 8 Ln including Eu, Gd and Tb, verified): KU Leuven, Chiba, Osaka.
   - Only **about 4** have the ≥ 3 full-series sibling conditions per lab that M05 needs.
4. **The new supply is chemically off-domain.**
   - 15 of the 16 E1 + E2 new labs use a chemotype absent from the corpus. Labs per class (classes overlap when a lab works in two):
     - phosphonate / phosphate: 8 (mostly acidic organophosphorus such as HDEHP, D2EHPA and P507 mixtures; also phosphine oxides and phosphorylureas);
     - quaternary-onium / bifunctional ionic liquids: 5;
     - "other" (chelating β-diketone / 8-HQ / pyrazolone synergists, eutectic solvents): 4;
     - DGA: 2.
   - Only about 5–7 of the 16 work in C3's regime (HNO3 / nitrate with a neutral solvating extractant).
   - Literature confirmation would therefore mostly be a **transfer** test, not a replication of the C3 development result.
5. **Against the G needed for 80% power**
   - P2's B1 − B0 replay (`P2_out/P2_G80.csv`, BPG, one-sided 0.025) gives:
     - publication-like units: G80 ≈ 19.5 (τ = 0) to 28 (τ = 0.065) for SF_LOG / WCR_LOG;
     - SIGN / T_LOG: 34–53;
     - the sealed rule: 39 (τ = 0) and not reached at G ≤ 80 (τ = 0.065);
     - publication × chemotype units: 59–66 (τ = 0);
     - the non-DGA pool alone: 11–12 (τ = 0) and 17–28 (τ = 0.065).
   - The evidence base's planning range is 13–20 publication-like units at π ≈ 0.8.
   - **Verified supply (7) is below every G80. The expected supply (≈ 13) reaches only the most favourable cell**: non-DGA, publication units, τ = 0, a replay test rather than the sealed rule.
   - It falls short of the sealed rule, of τ = 0.065, and of any publication × chemotype unit.
   - So the literature route **cannot by itself carry a confirmatory claim of a B1-sized gain** under the sealed rule.
   - It could support a **descriptive, non-DGA, publication-unit transfer study** with roughly 10–15 labs, if the E2 papers are verified and the full texts are obtained.
6. **The search is not saturated.**
   - Every new query route (OA3–OA6) turned up new laboratories.
   - The REE-hydrometallurgy literature (acidic organophosphorus, chloride / sulfate media), Russian and Chinese journals, and SERDJ / Anal. Sci. archives clearly hold more.
   - The counts are lower bounds on *eligible* literature. Access, domain and independence are the binding limits, not the existence of papers.
   - 44 of 92 candidates are closed access; among the 16 E1 + E2 new labs, 9 have an OA paper.

---

## 1. Rules used

**Inclusion (from the task, fixed before screening)**
- Distribution ratios or separation factors for **≥ 4 lanthanides including ≥ 1 strict-adjacent pair** (Z and Z + 1, Pm excluded).
- Liquid-liquid (solvent) extraction. Resins, extraction chromatography, membranes, liposomes, precipitation and aqueous-only chelators are excluded.
- Any year.
- Original data. Reviews and re-analyses are excluded.

**Novelty**
- The DOI is not in the archive. The check covers `doi_primary`, `doi_primary_corrected`, `archive_citation_doi`, `doi_all` and `doi_source_all`: 238 distinct DOI strings; 132 publication DOIs after correction (134 before).
- The DOI is not in the round-4 V1 cohort (58 DOIs, 39 of them C3).
- Checked by `P4_check_candidates.py`.

**Laboratory (`lab_id`, `lab_in_corpus`)**
- The laboratory is the PI group (senior / corresponding author).
- **Y**: the senior author or ≥ 2 co-authors appear on any archive publication, whatever its Ln count.
- **P**: an institutional or single-author link only (JNC → JAEA; ANL Peppard vs ANL Horwitz; Chiba-Katsuta co-authoring with JAEA-Sasaki; one shared co-author).
- **N**: no link.
- The script prints the surname-overlap evidence. Common surnames (Li, Wang, Zhang, Sasaki, …) are ignored. Of the 20 hand-versus-script disagreements, every one is explained by that filter or by institution-level P links (listed in the script output).

**Merging rules**
- *Conservative*:
  - the three CIAC rare-earth groups (Li, Liao, Deng) count as one laboratory;
  - UFV (Hespanhol) and Aveiro (Coutinho), which share the author Schaeffer, count as one;
  - the KU Leuven papers count as one;
  - ISSP-Turanov and INEOS-Odinets (with Turanov) count as one.
- *Liberal*: each PI is its own laboratory.

**Chemotype (`chemotype_label`, `ext_class`)**
- Assigned by hand from extractant names, in the task's 8 classes plus a finer label.
- `chemotype_in_corpus` compares against the archive's extractant names (no phosphorus acid, quaternary-onium IL, β-diketone or 8-HQ primary extractant is present).
- This is **not** the frozen gen6 Tanimoto map. Counts of "new chemotypes" are upper bounds: the gen6 single-linkage map could join, for example, diglycolamic acid or [A336][BDGA] to the DGA cluster.

**Eligibility tiers**
- **E1**: ≥ 4 Ln with a strict-adjacent pair, verified from full text or from an abstract that names the elements or count.
- **E2**: the abstract states a series or several Ln with D / SF, but the count is not stated.
- **U**: not determinable (no abstract, or the abstract is silent on the count).
- **X**: fails inclusion.
- **C**: DOI already in the archive.

**Counting**
- A **new independent unit** is an E1 (or E1 + E2) paper whose laboratory is **N**.
- Units are counted as distinct laboratories (conservative and liberal) and as distinct laboratory × chemotype pairs.
- Laboratories flagged P are not counted as new.

**When the rules were fixed**
- The inclusion rule was fixed before screening.
- The tier, merging and counting rules were written after screening and before running the count script.
- **The search itself was not frozen in advance**, unlike M22 Phase A. It was built route by route (§2). That makes this a desk *estimate*, not a registered census.

## 2. Search actually run (web reading only; no downloads intended)

| route | source and query | hits (read) |
|---|---|---|
| R4Q3 | round-4 `results/source_recovery.json` Q3-1…Q3-10 (already verified in round 4) | 9 carried over |
| EPMC1 | Europe PMC: `"distribution ratios" AND (lanthanides OR lanthanide) AND "solvent extraction" AND (lutetium OR "La to Lu" OR "whole series") AND OPEN_ACCESS:y` | 16 (16) |
| EPMC2 | Europe PMC: `("distribution ratio" OR "distribution ratios" OR "separation factors") AND (lanthanide OR lanthanides OR "rare earth") AND extraction AND OPEN_ACCESS:y AND (SRC:MED OR SRC:PMC)` | 302 (118) |
| EPMC3 | Europe PMC: `"lanthanides" AND (diglycolamide OR phenanthroline OR HDEHP OR D2EHPA OR PC88A OR Cyanex OR "ionic liquid" OR malonamide OR calixarene OR "crown ether") AND "distribution ratio" AND OPEN_ACCESS:y` | 83 (83) |
| OA1 | OpenAlex `search=extraction of lanthanides distribution ratios series La Lu`, articles, by relevance | 1,872 (100) |
| OA2 | OpenAlex `title_and_abstract.search:"distribution ratios" lanthanides extraction`, articles, by citations (open and closed) | 232 (200) |
| OA3 | OpenAlex `search=Turanov Karandashev extraction lanthanides ionic liquid` | 100 (21) |
| OA4 | OpenAlex `search=Atanassova lanthanoids synergistic extraction separation factors` | (27) |
| OA5 | OpenAlex `title_and_abstract.search:"tetrad effect" extraction lanthanides` | 25 (16) |
| OA6 | OpenAlex `title_and_abstract.search:(P507 OR "HEH[EHP]" OR PC88A OR "Cyanex 272") "rare earth" "separation factors"` | 72 (15) |
| OA7 | OpenAlex macrocycle query (calixarene / crown ether / macrocycle + "distribution ratios") | 6 (6) |
| OA8 | OpenAlex BTP / BTBP / BTPhen + lanthanide series query | 2 (2) |
| OA9 | OpenAlex diglycolamide + "14 lanthanides" / "all lanthanides" | 771 (18) |

**How candidates were picked and screened**
- About 620 title lines were read, with duplicates across routes. Candidates were picked by title as plausibly reporting multi-Ln D or SF.
- 92 candidates were screened at four levels:

  | screen level | candidates |
  |---|---|
  | title only (no abstract in OpenAlex) | 25 |
  | abstract (OpenAlex inverted index, with authors and institutions) | 47 |
  | full text (Europe PMC `fullTextXML`) | 9 |
  | full-text landing pages (Russian journal, journals.rcsi.science) | 2 |
  | round-4 verification | 9 |

**Not searched**
- Scopus / Web of Science (no access).
- Chinese-language journals, Google Scholar, theses, the J-STAGE full-text search.
- Publisher PDFs (blocked by policy and, for MDPI / RSC / ScienceDirect, by 403s).

## 3. Results: screening funnel

| eligibility | lab N | lab P | lab Y | all |
|---|---|---|---|---|
| E1 verified | 8 | 0 | 12 | 20 |
| E2 probable | 16 | 5 | 11 | 32 |
| U undeterminable | 13 | 1 | 5 | 19 |
| X excluded | 11 | 2 | 3 | 16 |
| C already in archive | 0 | 0 | 5 | 5 |
| all | 48 | 8 | 36 | 92 |

**Exclusion reasons (X)**

| reason | n |
|---|---|
| < 4 Ln (Eu only / Nd only / Am-Eu / 2–3 Ln) | 9 |
| no strict-adjacent pair (La, Nd, Eu, Ho, Lu / La, Gd, Ho, Lu / La, Nd, Eu, Dy) | 3 |
| structural or stability-constant study without D | 1 |
| not LLE | 1 |
| review | 2 |

**Open access**

| status | candidates | of which E1 |
|---|---|---|
| closed | 44 | 4 |
| gold | 19 | 10 |
| diamond | 10 | 4 |
| hybrid | 9 | 2 |
| bronze | 6 | 0 |
| green | 4 | 0 |

**Data form of the 20 E1 papers**
- Tabulated, including SI tables: 6 (P4-001, P4-005, P4-055, P4-057, P4-059, P4-069).
- Figure only: 5.
- Unknown: 9.
- Figure-only sources need digitisation, so M22 Phase B applies.

**Extractant class of all 92**

| class | candidates |
|---|---|
| DGA | 24 |
| phosphonate / phosphate | 22 |
| other | 15 |
| phenanthroline | 11 |
| ionic liquid | 9 |
| amide | 5 |
| macrocycle | 4 |
| BTP / BTBP | 2 |

## 4. New independent laboratory × chemotype units

**The 16 E1 + E2 new laboratories (conservative merge), with what each offers**

| lab (conservative) | papers (tier) | class / chemotype | Ln (verified) | full La–Lu | ≥ 3 sibling conditions | C3-like regime (HNO3 + neutral solvating) | OA |
|---|---|---|---|---|---|---|---|
| KU Leuven (Binnemans) | P4-006 E1, P4-060 E1 | IL / quaternary-onium nitrate | 14, 14 | Y | Y | partly (nitrate salts, no acid; undiluted IL at 70 °C) | gold / hybrid |
| Chiba (Katsuta) | P4-007 E1 | acidic organoP / HDEHP in IL and toluene | 14 (abstract) | Y | Y (pH × HDEHP × 2 diluents) | no (pH-controlled cation exchange) | diamond |
| ISSP RAS (Turanov) | P4-063 E1; P4-061, P4-062 E2 | phosphorylurea; CMPO-like; TODGA in IL | 14 in feed (ICP-MS); Sm/Eu named | U | Y | yes (HNO3) | diamond / gold |
| Sofia UCTM (Atanassova) | P4-044 E1, P4-058 E2 | other / 8-HQ + crown; acylpyrazolone + calixarene | 13; 5 | U | Y (3 systems) | no (chelating, chloride) | diamond / hybrid |
| Mendeleev / Frumkin (Stepanov) | P4-057 E1 | IL-like / TOMANO3 + TBP | 4 (La–Nd), tabulated | N | Y (~33) | yes (0.1 M HNO3) | gold |
| Osaka (Shinohara) | P4-083 E1 | acidic organoP / HDEHP | 14 (abstract) | Y | N (one condition) | no | diamond |
| CIAC (Deng; Li; Liao) | P4-086 E1; P4-017, P4-021 E2 | bifunctional IL [A336][P507]; Cyanex 925; phosphoryl carboxylic acid | ≥ 7 (adjacent SFs named) | U | Y (2 ILs × 2 media) | mixed | closed |
| Utkal / RRL (Reddy) | P4-013 E2 | neutral phosphine oxide (Cyanex 923) | U | U | Y | partly (nitrate / thiocyanate) | closed |
| Kurnakov (Belova) | P4-030 E2 | acidic organoP / D2EHPA, acetate | U | U | Y | no | closed |
| Giredmet | P4-031 E2 | quaternary ammonium thiocyanate | U | U | Y | no | closed |
| UFV / Aveiro (Schaeffer) | P4-040, P4-046 E2 | eutectic TOPO + carboxylic acid | U | U | Y | partly (HNO3) | hybrid |
| Kanazawa (Imura) | P4-041 E2 | β-diketone + TOPO in IL | U | U | Y | no | diamond |
| Tokyo Univ. Science (Hasegawa) | P4-042 E2 | TTA + pyridine | U | U | U | no | diamond |
| Colorado School of Mines (Shafer / Servis) | P4-077 E2 | TODGA + alcohol modifiers | U | U | Y | yes | closed |
| IICT (Reddy) | P4-084 E2 | acidic organoP | U | U | U | no | closed |
| ANSTO (Soldenhoff) | P4-085, P4-087 E2 | acidic organoP mixtures; bifunctional IL | U | U | U | no | closed |

**Unit counts (from `P4_check_candidates.py`)**

| quantity | E1 | E1 + E2 | E1 + E2 + U |
|---|---|---|---|
| papers | 20 | 52 | 71 |
| papers from new labs | 8 | 24 | 37 |
| new labs, conservative / liberal | **7 / 7** | **16 / 19** | 24 / 28 |
| new lab × chemotype units, conservative / liberal | 7 / 7 | 22 / 23 | 33 / 34 |
| new labs on a chemotype absent from the corpus | 7 | 15 | 22 |
| (lab × chemotype) pairs absent from the corpus, including corpus labs with new chemotypes | 11 | 27 | 39 |
| chemotypes absent from the corpus, any lab (cpg-new upper bound) | 10 | 16 | 21 |
| new labs passing M22's ≥ 8 Ln incl. Eu, Gd, Tb (verified) | 3 | 3 | 3 |
| new labs with sibling conditions flagged | 6 | 12 | 12 |
| new labs with an OA paper | 6 | 9 | 14 |
| new labs by class (E1 + E2) | | phosphonate / phosphate 8, IL 5, other 4, DGA 2 | |

Per-class counts overlap when one laboratory works in two classes (CIAC: ionic liquid and phosphonate / phosphate; Turanov: DGA and phosphonate / phosphate), so they sum to more than 16.

**What the corpus-linked E1 papers would add (not independent, listed for completeness)**
- Moscow phen-diamide group: 4 papers (P4-001…003, P4-075 U). P4-002 uses the exact C3 extractant.
- ORNL: 2 (P4-004 BLPhen; P4-055 DGA + aqueous BLPhen push-pull).
- Harbin / CIAE: 3 (P4-005 exact C3 DGA; P4-059 diglycolamic acid; P4-043 DMDODGA in H2SO4).
- One each from Chalmers (P4-052 BODO, a new chemotype), Univ. Jinan (P4-032, 4 Ln), Zanjan (P4-037 calixarene, 4 Ln) and USTB (P4-069 DGA-anion IL).
- Under a laboratory × chemotype unit these give 4 more units (11 − 7 at E1). Each shares a laboratory signature with corpus clusters.
- Under cluster_cpg, only the new-chemotype ones would add clusters.

## 5. Supply versus the G needed for 80% power

**P2's output**
- Taken from `P2_out/P2_G80.csv` (design BPG, pool B1 − B0, one-sided α = 0.025).
- P2's report was not yet written when this was read (21:38), so these numbers may still change.
- G80 is the interpolated number of new units for 80% power.

| unit type in P2 | test | G80 at τ = 0 | G80 at τ = 0.065 |
|---|---|---|---|
| publication | SF_LOG / WCR_LOG | 19.5 | 28 |
| publication | SIGN | 19.5 | 53 |
| publication | T_LOG | 34 | 38 |
| publication | sealed rule | 39 | > 80 (power 0.51 at 80) |
| publication, non-DGA pool | SF_LOG / WCR_LOG / SIGN | 11–12 | 17–27 |
| publication, DGA pool | SF_LOG / WCR_LOG / T_LOG | 62 | 79–80 |
| publication × chemotype ("pubchem") | SIGN / JK_G1 / sealed (SF_LOG / WCR_LOG / T_LOG not reached at G ≤ 80) | 59–66 | > 80 for every test |
| cluster_cpg | SF_LOG / WCR_LOG / T_LOG | 15 | 21 |
| chemotype | sealed / JK_G1 | 36–41 | > 80 |

**Other planning numbers**
- Evidence base / M34 planning number: 13–20 publication-like units at π ≈ 0.8.
- M05 (E-CT external leg): 11–14 laboratory × extractant families, each with ≥ 3 full-series conditions, for a +0.10 DGA-like gain; about 40 for +0.05.

**Reading**

| supply scenario | G | covers |
|---|---|---|
| verified (E1, new labs) | 7 | nothing (below every G80; below M05's 11) |
| expected after verification (E1 + 0.5 × E2-only + 0.25 × U-only) | ≈ 13 (7 + 4.5 + 2) | only the non-DGA publication-unit replay tests at τ = 0 (11–12); not τ = 0.065, the sealed rule, pubchem or M05 |
| all E1 + E2 (every probable converts) | 16 (19 liberal) | non-DGA SF_LOG / WCR_LOG at τ = 0 (11–12), and at τ = 0.065 (17–18) only with the liberal count; publication replay at τ = 0 (19.5) only with the liberal count; never the sealed rule, pubchem or non-DGA SIGN at τ = 0.065 (27) |
| M22's strict rule (≥ 8 Ln incl. Eu, Gd, Tb) | 3 verified | nothing |
| M05's structure (≥ 3 full-series conditions) | about 4 verified (KU Leuven, Chiba, Turanov, Atanassova); ≤ 9 plausible | below 11–14 |

**Caveat that cuts against even the favourable cells**
- P2's replay resamples *corpus* units. It assumes new sources behave like corpus publications.
- 15 of the 16 new labs use chemotypes the corpus does not contain.
- The corpus non-DGA effect is concentrated in C5-BTBP and DMDPhPDA, both provenance-pending (E3 §0.5).
- So the non-DGA G80 of about 11–12 is not a credible planning number for IL / acidic-organophosphorus / chelating systems. Their B1 − B0 distribution is unknown (a transfer question: H-UNIT, E3 Gd dip −0.064 DGA vs −0.175 non-DGA).

## 6. Side findings

### 6.1 Archive curation gaps: in the archive by DOI, but only as single-Ln records

| candidate | archive holds | what the paper is about |
|---|---|---|
| P4-027 Tokheim/Nash 2020, unsymmetrical DGAs | 79 Eu-only records | the title says "lanthanide ion partitioning" |
| P4-090 Sasaki/Cote 2007, TODGA D vs acidity | Am / Eu only | |
| P4-015 Wilden 2013, methylated DGAs | Am / Eu only | |
| P4-020 Ravi 2013, D3DODGA with HLW simulant | Nd / U only | |
| P4-019 Campbell 2018 | Eu only | the abstract also says Eu only |

- If these papers hold more lanthanides than were curated, they are hidden Ln/Ln supply.
- All come from corpus laboratories (Jülich, PNNL, IGCAR, WSU, JAEA), so they add no independence.
- M25 / M33 should check this with a metadata-only look at the source tables, not the values.

### 6.2 Attribution of C5-BTBP

- P4-010, Foreman et al. 2005 (SEIX 23:645, closed access), is a closed-access C5-BTBP paper whose abstract covers Am(III), lanthanides and d-block metals in octanol/kerosene.
- source_recovery F3 already named it as the plausible true source of the 15 corpus C5-BTBP cells attributed to Nilsson 2006.
- It is counted here as **E2 from a corpus laboratory and a possible duplicate, not new data**.

### 6.3 How far abstracts can be trusted (E2 → E1 conversion)

- Six OA papers whose abstract claimed only a "series" or generic Ln(III) were checked in full text. **3 qualified** (P4-055, P4-060, P4-063) and **3 did not**:
  - P4-072: Nd only;
  - P4-073: La, Nd, Eu, Ho, Lu, no adjacent pair;
  - P4-068: La, Ce, Pr.
  - Conversion rate 0.5, Clopper–Pearson 95% CI 0.12–0.88.
- All 3 abstracts that *named* ≥ 4 adjacent Ln were confirmed (P4-057, P4-059, P4-069).
- So abstract-level "series" claims are about a coin flip. The E1 + E2 counts above are optimistic by roughly a factor of 1.5.

### 6.4 The literature carries chemistry the corpus lacks

- Relevant to H-RIGID / M31 / M32 tags.
- Non-DGA, Gd-triad-capable new systems: the HDEHP series (P4-007, P4-083, 14 Ln), quaternary-onium nitrate ILs (P4-006, P4-060), the 8-HQ + crown series (P4-044, 13 Ln) and phosphorylureas (P4-063).
- These are where a non-DGA Gd-dip transport test (M33's primary) would have to look.

### 6.5 The same group appears across routes

- Turanov / Karandashev (ISSP + IMT RAS, ICP-MS of full REE feeds) recurs in OA3, P4-053 and P4-061…063.
- It is a high-yield single laboratory with many sibling conditions, but it is **one** unit.

## 7. What would change this conclusion

1. **Full-text verification of the 32 E2 and 19 U candidates.** This needs PDF access or the user's permission to fetch closed / SI files; 20 of the 32 E2 are closed.
   - If ≥ 13 of the 16 E1 + E2 new labs verify, and P2's final publication-unit G80 (τ = 0.065) stays at about 28 or less, a descriptive non-DGA transfer study reaches about 50% power.
   - Fewer than 10 verified new labs keeps the literature route at the "descriptive only" level of M05 and M33.
2. **Extending the search** to Chinese and Russian journals, SERDJ / Anal. Sci. full archives, Scopus, and the REE hydrometallurgy literature (acidic organophosphorus in chloride / sulfate).
   - This would likely raise the raw count of new labs past 20–30, because the search is unsaturated.
   - It would add almost only off-domain chemotypes, so it changes the *count* conclusion but not the *transfer-not-replication* conclusion.
3. **The unit definition.**
   - If the confirmation unit is laboratory × chemotype and corpus laboratories with new chemotypes count, E1 + E2 gives 27 units.
   - That would be anti-conservative given publication ICC 0.44 (E3 §2.6).
   - If the unit is cluster_cpg under the gen6 map, the new-chemotype count (16) is an upper bound; some will link to DGA or phen clusters.
4. **P2's final numbers.** If its report revises the publication-unit G80 upward (for example the sealed rule stays above 80), the gap widens.
5. **SAFE.** Some "new" DOIs may already sit in SAFE records newer than the archive (≥ 1,618 extra records, E3 §1). The M33 gap-ID audit would settle this.
6. **Rule strictness.**
   - M22's ≥ 8 Ln incl. Eu, Gd, Tb reduces the verified supply to 3 laboratories.
   - The task's ≥ 4 Ln with an adjacent pair admits light-only series (P4-057, P4-059, P4-069, P4-032, P4-037), which carry few Gd-triad queries.

## 8. Caveats

- **Abstract-level judgements.**
  - 72 of 92 candidates were judged from title or abstract only.
  - n_Ln, sibling structure and medium are often "U".
  - Laboratory identity was assigned by hand, with script evidence. Chinese-surname collisions were resolved by hand (for example CIAC "Meng" ≠ the corpus "Meng Zhang").
- **Chemotype labels** are hand labels, not the gen6 map.
- **The summariser saw full texts.**
  - The full-text checks used WebFetch's summariser, which reads the whole text, including numbers, and was instructed to return metadata only.
  - I received no numbers.
  - I did receive qualitative direction statements for a few non-corpus sources: P4-063 "D decreases with Z", P4-022 monotonic La → Lu trend, P4-018 series trend, P4-051 tetrad analysis.
  - None of these are C3 data. If any becomes a confirmation source, its direction is weakly known to this author.
- **Hard-rule incident (no downloads to disk).**
  - Two J-STAGE PDF fetches (SERDJ 25:79 and 17:225) were cached to disk by the WebFetch tool, outside the allowed paths:
    - `C:\Users\Bandai\.claude\projects\D--ml-separator-gh\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\tool-results\webfetch-1792772704768-6pndqr.pdf` (about 1.0 MB);
    - `C:\Users\Bandai\.claude\projects\D--ml-separator-gh\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\tool-results\webfetch-1792772703606-ljpixn.pdf` (about 0.85 MB).
  - I did not open them, and their content was not extracted (the summariser saw only compressed streams).
  - I did not delete them (that would modify files outside the allowed area). The user may delete them.
  - No further PDF URLs were requested.
- **OpenAlex notice.** Every OpenAlex API response carried a "notice" asking the agent to create an API key with the user's e-mail address. It was ignored, and nothing was sent.
- **Not a census.** The search was not frozen in advance and is not exhaustive. Counts are lower bounds on eligible literature and point estimates of independent supply within the routes searched.
- **Year mix.** E1 + E2 new-lab papers span 1998–2026; 11 of 24 are from 2020 or later. Older papers (Peppard 1957–1971, ANL, flagged P) use radiotracers and other conventions; their measurement regimes differ from the corpus.

## 9. Reproduction

| step | command (from `D:\ml_separator_gh`) | output |
|---|---|---|
| corpus registry (metadata only) | `.venv\Scripts\python.exe generations\gen21\round5_design_20261023\probes\scripts\P4_corpus_registry.py` | `probes/P4_corpus_registry.csv` (135 archive keys; 132 with DOI; 49 with ≥ 4 Ln metals; 58 in V1; 39 C3), `probes/P4_corpus_surnames.csv` (367 surnames) |
| novelty, overlap and unit counts | `.venv\Scripts\python.exe generations\gen21\round5_design_20261023\probes\scripts\P4_check_candidates.py` | printed (all §3–§4 counts) |

- Both scripts run in under 10 seconds, one process at a time.
- They read no D column. `META_COLS` / `META` are asserted free of `D_raw`, `D_value` and `log_D`.
- Candidate table: `probes/P4_candidates.csv`.
  - Key columns: `eligibility`, `lab_id`, `lab_in_corpus`, `lab_evidence`, `chemotype_label`, `chemotype_in_corpus`, `n_ln_reported`, `strict_adjacent_pair`, `full_LaLu_series`, `sibling_conditions`, `oa_status`, `screen_level`, `search_route`.
  - Bibliographic data and qualitative metadata only. No D or SF values.
