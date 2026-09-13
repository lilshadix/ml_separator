*regime: cohort=Pr/Nd case: literature placeholder entries PC88A and Cyanex 272 (no corpus rows), feed_status assumed (U1); holdout=none (computed regimes, consistency checks are not validation); averaging_unit=draw (parameter draw, seed 18); status_of_parameters=PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders), feed_status assumed, spec_status assumed*

| system_id | parameter | value | unit | status | range | source | note |
|---|---|---|---|---|---|---|---|
| sys_29976921e156a0a0 | organic_ligands[0].concentration | 0.8 | mol/L | assumed | [0.2, 1.5] | 10.1038/s41598-020-74041-9 \| extractant concentration series, 0.8 mol/L point | formal monomer concentration used in S2; the case sweeps the range as a design variable |
| sys_29976921e156a0a0 | organic_ligands[0].stoichiometry.ligands_per_metal | 3 | 1 | assumed | [3.0, 3.0] | 10.1038/s41598-020-74041-9 \| mechanism nRE3+ + n(HA2)org | dimers per Ln3+, ideal dilute-regime stoichiometry |
| sys_29976921e156a0a0 | organic_ligands[0].stoichiometry.protons_released_per_metal | 3 | 1 | assumed | [3.0, 3.0] | 10.1038/s41598-020-74041-9 \| mechanism |  |
| sys_29976921e156a0a0 | organic_ligands[0].stoichiometry.anions_per_metal | 0 | 1 | assumed | [0.0, 0.0] | none \| cation exchange transports no anion |  |
| sys_29976921e156a0a0 | medium.salting_anion_M |  | mol/L | unknown |  | none |  |
| sys_29976921e156a0a0 | medium.ionic_strength_M |  | mol/L | unknown |  | none |  |
| sys_29976921e156a0a0 | medium.temperature_C | 25 | Cel | assumed | [20.0, 30.0] | 10.1038/s41598-020-74041-9 \| 298 K |  |
| sys_29976921e156a0a0 | params.PC88A.20-30C.log_k.Nd | -1.95 | 1 | assumed | [-2.9, -1.0] | 10.1038/s41598-020-74041-9 \| Nd 64 % extracted at 0.8 M PC88A, initial pH 4.0, equilibrium pH 1.02-1.42, 1500 mg/L each Nd/Tb/Dy, A/O 1 | derive_logk_from_extraction: log D_Nd = 0.250; sum [M]_org = 24.3 mM; [(HA)2]_f = 0.400 - 3*0.0243 = 0.327 M; a = b = 3; pH_eq in [1.02, 1.42] gives [-2.55, -1.35]; widened by 0.3 for exponent uncertainty; with b = 2.22 (S2 slope) the window is [-1.45, -0.56]. Verify against Banda 2014 doi 10.1016/j.jiec.2014.03.002 and Thakur 1993 doi 10.1016/0304-386X(93)90084-Q. |
| sys_29976921e156a0a0 | params.PC88A.20-30C.log_k.Pr | -2.1 | 1 | assumed | [-3.08, -1.11] | 10.1016/j.jiec.2014.03.002 \| maximum SF(Nd/Pr) about 1.5 (task document); patent EP2388344A1: PC-88A SF(Nd/Pr) 1.4 in kerosene | log K_Pr = log K_Nd - log10 SF(Nd/Pr), SF placeholder range [1.3, 1.5] (log 0.114-0.176), central 1.4 (log 0.146) |
| sys_29976921e156a0a0 | params.PC88A.20-30C.delta_h_kj_mol |  | kJ/mol | unknown |  | 10.1038/s41598-020-74041-9 \| thermodynamics section: endothermic, values not transcribed |  |
| sys_29976921e156a0a0 | params.PC88A.20-30C.a_dimer | 3 | 1 | assumed | [2.0, 3.0] | 10.1038/s41598-020-74041-9 \| log D vs log[extractant] slopes 2-3 | ideal 3 |
| sys_29976921e156a0a0 | params.PC88A.20-30C.b_proton | 3 | 1 | assumed | [2.0, 3.0] | 10.1038/s41598-020-74041-9 \| log D vs pH slope, PC 88A Nd 2.22 | ideal 3; S2 measured 2.22 for Nd |
| sys_29976921e156a0a0 | phase.loc_metal_M |  | mol/L | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.loc_acid_M |  | mol/L | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.third_phase_observed |  | 1 | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.disengagement_s |  | s | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.ligand_loss_mol_per_L_aq |  | mol/L_aq | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.max_loading_fraction_studied |  | 1 | unknown |  | none |  |
| sys_f02db527a94a5e86 | organic_ligands[0].concentration | 0.8 | mol/L | assumed | [0.2, 1.5] | 10.1038/s41598-020-74041-9 \| extractant concentration series, 0.8 mol/L point | formal monomer concentration used in S2; the case sweeps the range as a design variable |
| sys_f02db527a94a5e86 | organic_ligands[0].stoichiometry.ligands_per_metal | 3 | 1 | assumed | [3.0, 3.0] | 10.1038/s41598-020-74041-9 \| mechanism nRE3+ + n(HA2)org | dimers per Ln3+, ideal dilute-regime stoichiometry |
| sys_f02db527a94a5e86 | organic_ligands[0].stoichiometry.protons_released_per_metal | 3 | 1 | assumed | [3.0, 3.0] | 10.1038/s41598-020-74041-9 \| mechanism |  |
| sys_f02db527a94a5e86 | organic_ligands[0].stoichiometry.anions_per_metal | 0 | 1 | assumed | [0.0, 0.0] | none \| cation exchange transports no anion |  |
| sys_f02db527a94a5e86 | medium.salting_anion_M |  | mol/L | unknown |  | none |  |
| sys_f02db527a94a5e86 | medium.ionic_strength_M |  | mol/L | unknown |  | none |  |
| sys_f02db527a94a5e86 | medium.temperature_C | 25 | Cel | assumed | [20.0, 30.0] | 10.1038/s41598-020-74041-9 \| 298 K |  |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.log_k.Nd | -3.45 | 1 | assumed | [-4.5, -2.4] | 10.1038/s41598-020-74041-9 \| Nd 27 % extracted at 0.8 M Cyanex 272, initial pH 4.0, equilibrium pH 1.22-1.70, 1500 mg/L each Nd/Tb/Dy, A/O 1 | derive_logk_from_extraction: log D_Nd = -0.432; sum [M]_org = 15.6 mM; [(HA)2]_f = 0.400 - 3*0.0156 = 0.353 M; a = b = 3; pH_eq in [1.22, 1.70] gives [-4.18, -2.74]; widened by 0.3 for exponent uncertainty; with b = 2.0 (S2 slope) the window is [-2.48, -1.52]. |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.log_k.Pr | -3.55 | 1 | assumed | [-4.65, -2.36] | none \| SF(Nd/Pr) assumed sweep {1.1, 1.2, 1.3, 1.4}; no source found | log K_Pr = log K_Nd - log10 SF(Nd/Pr) with SF an assumed sweep {1.1, 1.2, 1.3, 1.4} (log 0.041-0.146), central 1.25 (log 0.097); no Cyanex 272 Nd/Pr value found on 2026-09-13; candidate sources doi 10.1016/j.hydromet.2014.09.015, doi 10.1016/j.jre.2017.09.016, doi 10.1016/j.mineng.2013.10.021 |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.delta_h_kj_mol |  | kJ/mol | unknown |  | 10.1038/s41598-020-74041-9 \| thermodynamics section: endothermic, values not transcribed |  |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.a_dimer | 3 | 1 | assumed | [2.0, 3.0] | 10.1038/s41598-020-74041-9 \| log D vs log[extractant] slopes 2-3 | ideal 3 |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.b_proton | 3 | 1 | assumed | [2.0, 3.0] | 10.1038/s41598-020-74041-9 \| log D vs pH slope, Cyanex 272 Nd 2.0 | ideal 3; S2 measured 2.0 for Nd |
| sys_f02db527a94a5e86 | phase.loc_metal_M |  | mol/L | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.loc_acid_M |  | mol/L | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.third_phase_observed |  | 1 | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.disengagement_s |  | s | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.ligand_loss_mol_per_L_aq |  | mol/L_aq | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.max_loading_fraction_studied |  | 1 | unknown |  | none |  |
| feed | feed.flow_L_h | 1 | L/h | assumed | [0.1, 10.0] | none \| DESIGN.md 13.1 default; basis flow, results scale linearly |  |
| feed | feed.metals_mM.Nd | 75 | mmol/L | assumed | [7.5, 225.0] | none \| DESIGN.md 13.1: Nd:Pr 3:1 mol, total 0.1 M | range spans the sensitivity sets (total 0.03-0.3 M, ratio 1:1-3:1) |
| feed | feed.metals_mM.Pr | 25 | mmol/L | assumed | [7.5, 150.0] | none \| DESIGN.md 13.1: Nd:Pr 3:1 mol, total 0.1 M | range spans the sensitivity sets (total 0.03-0.3 M, ratio 1:1-3:1) |
| feed | feed.h_M | 0.01 | mol/L | assumed | [0.001, 0.1] | none \| DESIGN.md 13.1: [H+] 0.01 M (pH 2) | aqueous [H+] of the feed; the cascade computes the equilibrium acidity per stage |
| feed | feed.anion_M | 0.31 | mol/L | assumed | [0.1, 1.0] | none \| DESIGN.md 13.1: chloride 0.31 M = 3 x 0.1 M metal + 0.01 M HCl | total chloride of the feed |
| feed | feed.complexant_total_M | 0 | mol/L | assumed | [0.0, 0.0] | none \| no aqueous complexant in the PC88A / Cyanex 272 case |  |
| feed | feed.sodium_M | 0 | mol/L | assumed | [0.0, 0.0] | none \| no sodium in the feed; saponification adds it in the organic reserve |  |
| feed | feed.temperature_C | 25 | Cel | assumed | [20.0, 30.0] | none \| DESIGN.md 13.1: 25 C (band 20-30C of the literature entries) |  |
| feed | feed.sensitivity.total_metal_M | [0.03, 0.1, 0.3] | mol/L | assumed | [0.03, 0.3] | none \| DESIGN.md 13.1 sensitivity sets | Nd:Pr ratio held at the feed ratio; chloride = 3 x total metal + [H+] |
| feed | feed.sensitivity.ratio_nd_pr | ['1:1', '3:1'] | 1 | assumed | [1.0, 3.0] | none \| DESIGN.md 13.1 sensitivity sets | mol Nd per mol Pr at the feed total |
| spec | spec.purity_min_grid | [0.95, 0.97, 0.99] | 1 | assumed | [0.95, 0.99] | none \| DESIGN.md 13.1 grid | fraction of Nd among metals in the product, mol basis |
| spec | spec.recovery_min_grid | [0.8, 0.85, 0.9] | 1 | assumed | [0.8, 0.9] | none \| DESIGN.md 13.1 grid | recovery_from_feed (origin-labelled, section 7.8), never recovery_total |
