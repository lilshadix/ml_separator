# Pre-fit addendum (orchestrator, 2026-09-13, written before DATA_AUDIT.md exists and before sealing)

Status of the pre-registration's open items U2 and U6 (`DESIGN.md` §16, `PRE_REGISTRATION.md` §2
"Declared assumptions"). Nothing here changes a decision rule; it records what was read, at the
abstract level only, and what it implies. No result had been seen when this was written (no fit
had been run; the implementation fleet was still building the code).

## U2 — sources of the E2 loading-series publications (abstracts read via OpenAlex; full texts not accessible)

| publication_id | DOI | what it is | O/A stated in abstract | metal-concentration semantics |
|---|---|---|---|---|
| pub_5a68dc5665 | 10.1080/07366299.2015.1087209 | Sasaki, Sugo, Morita, Nash, *Solvent Extr. Ion Exch.* 2015, "The Effect of Alkyl Substituents on Actinide and Lanthanide Extraction by Diglycolamide Compounds" — reports lanthanide **loading capacity** for TODGA, TDDGA, TDdDGA, TEHDGA in n-dodecane at 3 M HNO3, capacity order TDdDGA > TDDGA > TODGA > TEHDGA | not stated | the corpus rows (Nd 4.9–12 mM, D 23.5 → 0.60) are the loading-capacity experiment; treated as initial aqueous concentration (assumption retained) |
| pub_0e7f3e0563 | 10.1002/slct.202202610 | Narayanan, Rama Swami, Prathibha, Venkatesan, *ChemistrySelect* 2022 — **third-phase formation** of Nd(III) in 0.2 M D3DODGA/n-dodecane by DLS and FTIR; states that D3DODGA loads higher Nd without third phase | not stated | the corpus range 0.1–2180 mM contains values that only make sense as g/L in one copy; consistent with the unit-slip rule of `DESIGN.md` §4.6 (7 rows) |
| pub_d3c970567f | 10.1016/j.aca.2005.04.061 | Sasaki, Sugo, Suzuki, Kimura, *Anal. Chim. Acta* 2005, "A method for the determination of **extraction capacity** …" — the three flat TODGA/Nd series at 3 M HNO3 / 0.1, 0.2, 0.3 M TODGA are capacity determinations (organic near saturation), not dilute loading curves | not stated | assumption retained |
| pub_917a4583d4 | 10.1007/s10967-010-0641-2 | Metwally, Saleh, Abdel-Wahaab, El-Naggar, *J. Radioanal. Nucl. Chem.* 2010, cerium by TODGA from HNO3 — D rises with Ce concentration (0.011–52 mM) | not stated | assumption retained; the rising direction is not explained by the abstract |
| pub_15b174237f | 10.1039/c8nj01074a | Rama Swami, Kumaresan, Venkatesan, Antony, *New J. Chem.* 2018 — TEHDGA/n-dodecane with HDEHP as a "reactive" phase modifier (3 Nd points) | not stated | assumption retained |

Corroboration for the family: Nobley & Jensen, *RSC Adv.* 2025, 15, 43941 (doi 10.1039/d5ra06417a,
open access, read in full): TODGA/n-dodecane lanthanide extractions use "equal volumes of fresh,
Nd(III) containing aqueous phase and preequilibrated organic phase" and report the **initial
aqueous** concentration (e.g. 4.90 mM Nd³⁺). This is the same laboratory convention the
pre-registration assumes (O/A = 1, initial aqueous). **Decision:** the declared assumptions stand
for every corpus loading row, status `assumed`, flag `OA_ASSUMED`; no series-specific replacement.

## U6 — a sourced third-phase limit for TODGA in n-dodecane

Tachimori, Sasaki, Suzuki, *Solvent Extr. Ion Exch.* 20(6), 687–699 (2002), doi
10.1081/SEI-120016073, abstract (OpenAlex reconstruction; publisher page 403): the loading capacity
of TODGA/n-dodecane "depend[s] not only on aqueous acidity and temperature but also on molecular
size of alkane solvent [and] kinds of aqueous anions"; **"0.1 M TODGA-n-dodecane … 0.008 M Nd(III)
with an aqueous 3 M HNO3"**; adding DHOA above 0.5 M eliminated the Nd(III) third phase; 0.2 M TODGA
+ 1 M DHOA gives a satisfactorily high loading capacity with slightly lower extractability.
Temperature is not in the abstract (the full text was not accessible). Status
`literature` (abstract-level locator), value **loc_metal_M = 0.008 mol/L** for
(0.1 M TODGA, n-dodecane, 3 M HNO3, Nd).

Implications, recorded before any fit:
1. The Sasaki 2015 TODGA/Nd series (4.9 → 12 mM, D 23.5 → 0.60) crosses this LOC: its collapse is
   a **third-phase boundary**, not ligand depletion. The pre-registered rule R2 still counts the
   series as it falls (no exclusion, no re-weighting); the report must interpret the C1 miss on
   this series with the LOC, and the cascade must raise `THIRD_PHASE_RISK` above 8 mM Nd in the
   organic for this system when the sourced value is in the entry.
2. To be applied at integration (not by the parallel owners, whose files are open): add to
   `gen18proc/literature.py` a table `PHASE_LITERATURE` keyed by system_id with the `Sourced`
   above, merged into the corpus entry's `phase.loc_metal_M` by `g18_build_db.py`; the value applies
   only at 0.1 M TODGA / 3 M HNO3 and the note must say so (the LOC scales with TODGA
   concentration, acidity, temperature and diluent per the same abstract).

## What was not read

No full text of any of the six papers. No O/A ratio was found for any E2 series. The LOC's
temperature is unknown. These stay as recorded here; anyone with journal access can replace the
abstract-level locators.
