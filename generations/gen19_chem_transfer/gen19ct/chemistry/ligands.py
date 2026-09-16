"""``chemistry/ligands.py`` -- extractant / component representation (brief sections 4.2, 27, 28 A.5/A.11).

One record per unique canonical component SMILES in the archive ``components`` column (roles
``organic_extractant`` / ``phase_modifier`` / ``aqueous_complexant`` / ``aqueous_holdback``), plus
flagged records for name-only components.  For every structure this module provides

* RDKit descriptors (MW, Crippen MolLogP, TPSA, HBD, HBA, rotatable bonds, heavy atoms, charge,
  fragments, InChIKey, a stereo/charge/salt-free connectivity key, Morgan r=2 on-bits);
* a donor-atom census by SMARTS (:data:`DONOR_SMARTS`) and a denticity proxy (:func:`denticity`);
* an acidity class (:func:`acidity_class`): ``ACIDIC / NEUTRAL / BASIC / IONIC / UNKNOWN``;
* a chemical FAMILY by ordered SMARTS rules.  The rules live in ``descriptors/family_rules.json``
  (written from :data:`FAMILY_RULES_SPEC` by ``scripts/g19_build_extractants.py``);
  :class:`FamilyClassifier` reads that file.

Name handling follows the archive's rule "structures are trustworthy, names are not"
(``dataset_all_metals/README.md`` point 2).  :class:`AliasIndex` merges spelling variants of a name
(case / hyphen / space / underscore) only when they are attached to the same structure, and never
lets a name that is owned by another structure, fails a name-structure plausibility check, or is a
masking-agent name (the ``ST*.json`` sub-source pattern) become a structure's canonical name.

Denticity proxy (documented rule)
---------------------------------
1. Donor atoms are the atoms matched by the donor SMARTS marked ``site: true`` in
   :data:`DONOR_SMARTS` (thiophene S, ester and P-O-C ether oxygens are not donor sites).
2. Donor atoms are collapsed into *sites*: atoms bonded to the same central atom (the P of a
   phosphoryl / P-OH / P=S head, the C of a carboxylic acid, the S of a sulfonate) form one site, and
   all pyridine-type aromatic N atoms of one aromatic ring form one site (a ring binds through one N).
3. Two sites can chelate together when the shortest topological distance between any of their atoms
   is 3 or 4 bonds -- the 1,4 and 1,5 relationships that close a 5- or 6-membered chelate ring --
   unless that shortest path runs through >= 3 bonds of one ring of <= 7 atoms (a transannular
   pair such as the ring O and the 4-keto O of a 4-pyranone, which point away from each other).
4. ``denticity_proxy`` = size of the largest connected set of sites under that relation (1 for a
   monodentate donor, 0 when there is no donor site); ``n_chelating_units`` = number of connected
   sets with at least two sites.  Known limitations: 7-membered chelates (e.g. pyridine-N-oxide
   bis-phosphine oxides) are not credited; conformational reach of flexible bis-ligands (two DGA
   heads on one linker) is not modelled, so ``denticity_proxy`` is the largest single chelating
   unit, not the total binding capacity; backbone ether O / thioether S in saturated rings fused to
   an N-heterocycle (Cy5-O-Me4-BTBP, Cy5S-Me4-BTBP) are counted as sites although they point away
   from the N4 cavity (these two BTBPs read 6 instead of 4).

Also here: :data:`REFERENCE_EXTRACTANTS` (definitional SMILES of named industrial extractants for
the presence audit and tests), :data:`NAME_ONLY_LOOKUP` / :data:`KNOWN_NAME_OVERRIDES`, and
:func:`resolve_occurrence` (the chemistry one archive component entry stands for, with the
name-correction rules documented there).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from gen19ct import paths

RDLogger.DisableLog("rdApp.*")

FAMILY_RULES_PATH = paths.DESCRIPTORS_DIR / "family_rules.json"
RULES_SCHEMA = "gen19.family_rules.v1"

ROLES = ("organic_extractant", "phase_modifier", "aqueous_complexant", "aqueous_holdback")
AQUEOUS_ROLES = ("aqueous_complexant", "aqueous_holdback")
ACIDITY_CLASSES = ("ACIDIC", "NEUTRAL", "BASIC", "IONIC", "UNKNOWN")

# ----------------------------------------------------------------------------------------------
# Donor-atom census.  ``first atom of the SMARTS`` is the donor atom that is counted.
# ----------------------------------------------------------------------------------------------
DONOR_SMARTS: dict[str, dict[str, Any]] = {
    "amide_carbonyl_O": {"smarts": "[OX1]=[#6X3]~[#7X3]", "site": True,
                         "note": "C=O of an amide or aromatic lactam"},
    "thioamide_S": {"smarts": "[SX1]=[#6X3]~[#7X3]", "site": True, "note": "C=S of a thioamide"},
    "ketone_O": {"smarts": "[OX1]=[#6X3](~[#6])~[#6]", "site": True,
                 "note": "C=O with two carbon neighbours (incl. aromatic 4-pyranone)"},
    "ether_O": {"smarts": "[OX2,o;H0;!$(O-[#6X3]=[O,S]);!$(O~[#15,#16,#7])](~[#6])~[#6]", "site": True,
                "note": "C-O-C ether incl. furan/pyran ring O; excludes esters and P-O-C / S-O-C"},
    "phosphoryl_O": {"smarts": "[OX1]=[PX4]", "site": True, "note": "P=O"},
    "P_OH_acidic_O": {"smarts": "[OX2H1,OX1-]-[PX4]", "site": True, "note": "P-OH or P-O(-)"},
    "P_S": {"smarts": "[SX1,SX2H1,SX1-;$(S~[PX4])]", "site": True, "note": "P=S, P-SH, P-S(-)"},
    "carboxylic_acid_O": {"smarts": "[$([OX1]=[CX3]-[OX2H1,OX1-]),$([OX2H1,OX1-]-[CX3]=[OX1])]", "site": True,
                          "note": "both O of COOH / COO(-); counted per atom, collapsed per group"},
    "oxime_N": {"smarts": "[NX2;$(N(=[#6])-[OX2H1,OX1-])]", "site": True, "note": "C=N-OH oxime N"},
    "hydroxyl_O": {"smarts": "[OX2H1]-[#6;!$([#6]=[O,S])]", "site": True,
                   "note": "alcohol or phenol OH (not acid)"},
    "aromatic_N_pyridine_type": {"smarts": "[nX2]", "site": True,
                                 "note": "two-connected aromatic N not in a 1,2,4-/1,3,5-triazine ring"},
    "triazine_N": {"smarts": "[nX2]", "site": True,
                   "note": "two-connected aromatic N in a six-membered ring holding >= 3 N"},
    "amine_N_tertiary": {"smarts": "[NX3;H0;!$(N-[#6X3]=[O,S,N]);!$(N-a);!$(N-[#7,#8,#15,#16])](-[#6])(-[#6])-[#6]",
                         "site": True, "note": "sp3 tertiary amine (not amide, not aniline)"},
    "amine_N_primary_secondary": {"smarts": "[NX3;H1,H2;!$(N-[#6X3]=[O,S,N]);!$(N-a);!$(N-[#7,#8,#15,#16])]-[#6]",
                                  "site": True, "note": "sp3 primary/secondary amine"},
    "ammonium_N_quaternary": {"smarts": "[NX4+;H0](-[#6])(-[#6])(-[#6])-[#6]", "site": False,
                              "note": "quaternary ammonium (no lone pair)"},
    "thioether_S": {"smarts": "[SX2;H0;!a](-[#6])-[#6]", "site": True, "note": "aliphatic C-S-C"},
    "thiophene_S": {"smarts": "[sX2]", "site": False, "note": "aromatic S (census only, not a site)"},
    "N_oxide_O": {"smarts": "[OX1-]-[#7+;!$([#7+]=O)]", "site": True, "note": "N-oxide O (not nitro/nitrate)"},
    "sulfonate_O": {"smarts": "[OX1,OX2H1,OX1-;$(O~[SX4](=O)(=O)[#6])]", "site": False,
                    "note": "sulfonate/sulfonic acid O (census only)"},
}

# ----------------------------------------------------------------------------------------------
# Family rules.  Ordered; first non-subsumed rule that fires gives the label.  Each rule is
# ``any_of`` clauses, each clause ``all_of`` predicates:
#   {"smarts": S, "min": n, "max": m}   unique substructure match count on the whole molecule
#   {"descriptor": d, "min": x, "max": y}   one of :func:`_rule_descriptors`
# ``subsumes`` declares nesting: a fired rule listed there does not make the assignment AMBIGUOUS.
# ``overlay: true`` (hydrophilic agents): the label replaces the structural family, which is kept
# as ``core_family``.
# ----------------------------------------------------------------------------------------------
_ALL_CORE = "__ALL__"

FAMILY_RULES_SPEC: dict[str, Any] = {
    "schema": RULES_SCHEMA,
    "version": "2026-09-15.1",
    "semantics": {
        "order": "rules are evaluated in list order; the label is the first fired rule not subsumed by another fired rule",
        "ambiguous": "AMBIGUOUS is flagged when two or more fired rules remain after removing every rule listed in the 'subsumes' of another fired rule (non-nested rules)",
        "subsumes_when": "a fired rule also subsumes the named fired rule when every predicate of the entry holds (same predicate syntax as any_of clauses); used where the chemistry of one group dominates only in a stated structural situation",
        "overlay": "an overlay rule (hydrophilic_aqueous_agent, electrolyte_salt) replaces the label; the structural label computed without overlays is kept as core_family",
        "smarts_predicate": "count of unique substructure matches on the whole molecule (all fragments) must lie in [min, max]",
        "descriptor_predicate": "see gen19ct.chemistry.ligands._rule_descriptors",
        "fallback": "other",
    },
    "families": [
        {"name": "electrolyte_salt", "overlay": True, "subsumes": [_ALL_CORE],
         "description": "inorganic salts / simple anions (nitrate, chloride, sulfate, Tf2N-)",
         "any_of": [[{"descriptor": "n_carbon", "max": 0}],
                    [{"smarts": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F", "min": 1},
                     {"descriptor": "n_carbon", "max": 2}]]},
        {"name": "hydrophilic_aqueous_agent", "overlay": True, "subsumes": [_ALL_CORE],
         "description": "water-soluble complexants / holdback agents: sulfonated N-donors, polyaminocarboxylates, hydrophilic (Crippen logP <= 1) donors such as TEDGA, polyols, hydrophilic N-heterocyclic acids",
         "any_of": [[{"smarts": "[#6]-[SX4](=[OX1])(=[OX1])[OX2H1,OX1-]", "min": 1}],
                    [{"smarts": "[#7]([CH2][CX3](=[OX1])[OX2H1,OX1-])[CH2][CX3](=[OX1])[OX2H1,OX1-]", "min": 1}],
                    [{"descriptor": "logp_largest_fragment", "max": 1.0},
                     {"descriptor": "largest_ring_size", "max": 11},
                     {"descriptor": "heavy_atoms_largest_fragment", "min": 5},
                     {"descriptor": "n_donor_sites", "min": 1}],
                    [{"descriptor": "aliphatic_OH", "min": 2}, {"descriptor": "logp_largest_fragment", "max": 2.5}],
                    [{"descriptor": "aromatic_N", "min": 1}, {"descriptor": "acid_groups", "min": 1},
                     {"descriptor": "logp_largest_fragment", "max": 2.0}]]},
        {"name": "quaternary_ammonium", "subsumes": ["ionic_liquid", "amine_basic"],
         "description": "tetraalkylammonium cations (Aliquat 336)",
         "any_of": [[{"smarts": "[NX4+;H0](-[#6])(-[#6])(-[#6])-[#6]", "min": 1}]]},
        {"name": "ionic_liquid", "subsumes": ["n_heterocyclic", "amine_basic"],
         "description": "organic cation (imidazolium/pyridinium, phosphonium, sulfonium) with a charge or counter-ion",
         "any_of": [[{"smarts": "[n+,P+,S+;!$([#7+]-[O-]);!$([#7+]=O);!$([#15+]-[O-])]", "min": 1},
                     {"descriptor": "charged_or_multifragment", "min": 1}]]},
        {"name": "dithiophosphinic_acid", "subsumes": ["thiophosphorus_acid", "sulfur_donor_other"],
         "description": "R2P(=S)SH (Cyanex 301 type)",
         "any_of": [[{"smarts": "[PX4](=[SX1])(-[#6])(-[#6])-[SX2H1,SX1-]", "min": 1}]]},
        {"name": "thiophosphorus_acid", "subsumes": ["sulfur_donor_other"],
         "description": "other P(=S)-OH / P-SH acids (Cyanex 302 type, thiophosphonic acids)",
         "any_of": [[{"smarts": "[PX4](=[SX1,OX1])-[SX2H1,SX1-]", "min": 1}],
                    [{"smarts": "[PX4](=[SX1])-[OX2H1,OX1-]", "min": 1}]]},
        {"name": "phosphoric_acid", "subsumes": [],
         "description": "(RO)2P(=O)OH / ROP(=O)(OH)2, no P-C bond (D2EHPA, HDEHP, P204)",
         "any_of": [[{"smarts": "[PX4;!$(P~[#6])](=[OX1])-[OX2H1,OX1-]", "min": 1}]]},
        {"name": "phosphonic_acid_monoester", "subsumes": [],
         "description": "R(RO)P(=O)OH (PC88A, P507, HEH[EHP])",
         "any_of": [[{"smarts": "[PX4](=[OX1])(-[#6])(-[OX2]-[#6])-[OX2H1,OX1-]", "min": 1}]]},
        {"name": "phosphonic_acid", "subsumes": [],
         "description": "RP(=O)(OH)2",
         "any_of": [[{"smarts": "[PX4](=[OX1])(-[#6])(-[OX2H1,OX1-])-[OX2H1,OX1-]", "min": 1}]]},
        {"name": "phosphinic_acid", "subsumes": [],
         "description": "R2P(=O)OH (Cyanex 272)",
         "any_of": [[{"smarts": "[PX4](=[OX1])(-[#6])(-[#6])-[OX2H1,OX1-]", "min": 1}]]},
        {"name": "carbamoylmethylphosphine_oxide",
         "subsumes": ["phosphine_oxide", "phosphonate_phosphinate_neutral", "neutral_organophosphate", "monoamide", "polyamide_other"],
         "description": "P(=O)-CH2-C(=O)N (CMPO; also carbamoylmethylphosphonates)",
         "any_of": [[{"smarts": "[PX4;!$(P-[OX2H1]);!$(P-[OX1-])](=[OX1])-[CX4]-[CX3](=[OX1])-[NX3]", "min": 1}]]},
        {"name": "amide_phosphoryl_hybrid",
         "subsumes": ["phosphine_oxide", "phosphonate_phosphinate_neutral", "monoamide", "polyamide_other", "amine_basic"],
         "description": "P(=O)-C-[O,N]-C-C(=O)N (DGA analogues with one amide replaced by a phosphoryl)",
         "any_of": [[{"smarts": "[PX4;!$(P-[OX2H1])](=[OX1])-[CX4]-[#8,#7]-[CX4]-[CX3](=[OX1])-[NX3]", "min": 1}]]},
        {"name": "neutral_organophosphate", "subsumes": [],
         "description": "(RO)3P=O (TBP)",
         "any_of": [[{"smarts": "[PX4;!$(P~[#6])](=[OX1])(-[OX2]-[#6])(-[OX2]-[#6])-[OX2]-[#6]", "min": 1}]]},
        {"name": "phosphonate_phosphinate_neutral", "subsumes": [],
         "description": "R(RO)2P=O or R2(RO)P=O (DBBP)",
         "any_of": [[{"smarts": "[PX4](=[OX1])(-[#6])(-[OX2]-[#6])-[OX2]-[#6]", "min": 1}],
                    [{"smarts": "[PX4](=[OX1])(-[#6])(-[#6])-[OX2]-[#6]", "min": 1}]]},
        {"name": "phosphine_oxide", "subsumes": [],
         "description": "R3P=O (TOPO, Cyanex 923)",
         "any_of": [[{"smarts": "[PX4](=[OX1])(-[#6])(-[#6])-[#6]", "min": 1}]]},
        {"name": "beta_diketone", "subsumes": [],
         "description": "R-C(=O)-CH2-C(=O)-R and enol (HTTA, acetylacetone)",
         "any_of": [[{"smarts": "[#6]-[CX3](=[OX1])-[CX4;H1,H2]-[CX3](=[OX1])-[#6]", "min": 1}],
                    [{"smarts": "[#6]-[CX3](=[OX1])-[CX3]=[CX3](-[OX2H1])-[#6]", "min": 1}]]},
        {"name": "hydroxyoxime", "subsumes": [],
         "description": "hydroxy-oxime chelators (LIX 84/860 type, alpha-hydroxyoximes)",
         "any_of": [[{"smarts": "[OX2H1]-[#6]~[#6]-[#6X3]=[NX2]-[OX2H1]", "min": 1}],
                    [{"smarts": "[OX2H1]-[#6X4]-[#6X3]=[NX2]-[OX2H1]", "min": 1}]]},
        {"name": "acylpyrazolone", "subsumes": ["beta_diketone", "monoamide", "polyamide_other", "n_heterocyclic", "amine_basic"],
         "description": "4-acyl-5-pyrazolones, keto and enol forms (HPMBP / PMBP): beta-keto-lactam O,O chelators, acidic",
         "any_of": [[{"smarts": "[#6]-[CX3](=[OX1])-[#6;R1]1~[#6;R1](~[OX1,OX2H1])~[#7;R1]~[#7;R1]~[#6;R1]~1", "min": 1}]]},
        {"name": "hydroxyquinoline", "subsumes": ["n_heterocyclic"],
         "description": "8-hydroxyquinolines (N,O chelators, Kelex 100 type)",
         "any_of": [[{"smarts": "[OX2H1]-c1cccc2cccnc12", "min": 1}]]},
        {"name": "hydroxyketone_catechol", "subsumes": [],
         "description": "O,O chelators: alpha-hydroxy ketone on an sp2 carbon (3-hydroxyflavones such as quercetin, maltol, tropolone) or a catechol",
         "any_of": [[{"smarts": "[OX2H1]-[#6X3;!$([#6]=[OX1])]~[#6X3]=[OX1]", "min": 1}],
                    [{"smarts": "[OX2H1]-c:c-[OX2H1]", "min": 1}]]},
        {"name": "aminopolycarboxylic_acid", "subsumes": ["carboxylic_acid", "amine_basic"],
         "description": "amine N carrying two CH2-COOH arms (EDTA, DTPA, HEDTA, CDTA, NTA)",
         "any_of": [[{"smarts": "[NX3;!$(N-[CX3]=[OX1])](-[CH2]-[CX3](=[OX1])-[OX2H1,OX1-])-[CH2]-[CX3](=[OX1])-[OX2H1,OX1-]", "min": 1}]]},
        {"name": "carboxylic_acid", "subsumes": [],
         "description": "R-COOH (Versatic, naphthenic, 2-bromoalkanoic acids)",
         "any_of": [[{"smarts": "[#6]-[CX3](=[OX1])-[OX2H1,OX1-]", "min": 1}]]},
        {"name": "diglycolamide", "subsumes": ["monoamide", "polyamide_other", "podand_ether", "sulfur_donor_other"],
         "description": "N-C(=O)-C-X-C-C(=O)-N with X = ether O (incl. ring-ether THF/pyran and bis-DGA variants: TODGA, TEHDGA) or thioether S (thiodiglycolamides)",
         "any_of": [[{"smarts": "[NX3]-[CX3](=[OX1])-[#6;X4,a]-[#8,#16;X2,a]-[#6;X4,a]-[CX3](=[OX1])-[NX3]", "min": 1}]]},
        {"name": "malonamide", "subsumes": ["monoamide", "polyamide_other"],
         "description": "N-C(=O)-C-C(=O)-N (DMDOHEMA)",
         "any_of": [[{"smarts": "[NX3]-[CX3](=[OX1])-[CX4]-[CX3](=[OX1])-[NX3]", "min": 1}]]},
        {"name": "amino_polyamide", "subsumes": ["monoamide", "polyamide_other", "amine_basic"],
         "description": "amine N carrying two CH2-C(=O)N arms (MIDOA, NTAamide, ADAAM)",
         "any_of": [[{"smarts": "[NX3;!$(N-[CX3]=[O,S])](-[CX4]-[CX3](=[OX1])-[NX3])-[CX4]-[CX3](=[OX1])-[NX3]", "min": 1}]]},
        {"name": "pyridine_carboxamide", "subsumes": ["n_heterocyclic", "monoamide", "polyamide_other", "sulfur_donor_other"],
         "description": "aromatic N adjacent to a carboxamide/thioamide (picolinamides, pyridine-2,6-dicarboxamides, phenanthroline carboxamides)",
         "any_of": [[{"smarts": "[nX2]:[c]-[CX3](=[OX1,SX1])-[NX3]", "min": 1}]]},
        {"name": "crown_ether_calixarene", "subsumes": ["podand_ether", "monoamide", "polyamide_other", "n_heterocyclic"],
         "description": "macrocyclic polyethers (ring >= 12 atoms with >= 4 O) and calixarenes (ring >= 12 atoms with >= 4 aromatic atoms and >= 3 CH2)",
         "any_of": [[{"descriptor": "largest_ring_size", "min": 12}, {"descriptor": "macro_ring_O", "min": 4}],
                    [{"descriptor": "largest_ring_size", "min": 12}, {"descriptor": "macro_ring_aromatic", "min": 4},
                     {"descriptor": "macro_ring_CH2", "min": 3}]]},
        {"name": "podand_ether", "subsumes": ["monoamide", "polyamide_other"],
         "description": "open-chain polyether donors: two acyclic ether O separated by C-C (DOODA)",
         "any_of": [[{"smarts": "[OX2;H0;!R](-[#6])-[CX4;!R]-[CX4;!R]-[OX2;H0;!R]-[#6]", "min": 1}]]},
        {"name": "n_heterocyclic", "subsumes": [],
         "subsumes_when": [
             {"rule": "carboxylic_acid",
              "all_of": [{"descriptor": "aromatic_N_chelate_sites", "min": 2},
                         {"descriptor": "carboxyl_O_near_aromatic_N", "max": 0}],
              "note": "a chelating aromatic-N pocket (>= 2 ring-N sites 3-4 bonds apart) with a carboxylic acid that cannot close a 5/6-membered chelate with any ring N: the acid is a remote substituent (EsPyTri: CH2COOH on pyridine C4 of a bis-triazolylpyridine), not the donor"}],
         "description": "neutral aromatic N-donors (BTP, BTBP, BTPhen, bipyridine, phenanthroline, triazinyl/pyrazolyl/triazolyl pyridines)",
         "any_of": [[{"smarts": "[nX2]", "min": 1}]]},
        {"name": "polyamide_other", "subsumes": ["monoamide"],
         "description": ">= 2 amide groups not in a DGA / malonamide / amino-polyamide / pyridine-carboxamide motif",
         "any_of": [[{"smarts": "[CX3](=[OX1])-[NX3]", "min": 2}]]},
        {"name": "monoamide", "subsumes": [],
         "description": "exactly one C(=O)N (DEHiBA, DHOA; the archive's 'DEHPA' is here)",
         "any_of": [[{"smarts": "[CX3](=[OX1])-[NX3]", "min": 1, "max": 1}]]},
        {"name": "amine_basic", "subsumes": [],
         "description": "aliphatic amines (tertiary amine extractants)",
         "any_of": [[{"smarts": "[NX3;!$(N-[#6X3]=[O,S,N]);!$(N-a);!$(N-[#7,#8,#15,#16])]-[#6]", "min": 1}]]},
        {"name": "sulfur_donor_other", "subsumes": [],
         "description": "neutral S donors: phosphine sulfides, thioamides, acyclic thioethers (a ring thioether in a saturated backbone ring, e.g. Cy5S-Me4-BTBP, does not fire)",
         "any_of": [[{"smarts": "[PX4;!$(P-[SX2H1]);!$(P-[OX2H1])]=[SX1]", "min": 1}],
                    [{"smarts": "[CX3]=[SX1]", "min": 1}],
                    [{"smarts": "[SX2;H0;!a;!R](-[#6])-[#6]", "min": 1}]]},
        {"name": "aliphatic_alcohol", "subsumes": [],
         "description": "alcohols without N/P/S or carbonyl (phase modifiers: 1-octanol, isodecanol)",
         "any_of": [[{"smarts": "[CX4]-[OX2H1]", "min": 1}, {"smarts": "[#7,#15,#16,$([#6]=[OX1])]", "max": 0}]]},
    ],
    "name_structure_checks": [
        {"id": "ACID_NAME_REQUIRES_ACID_GROUP", "name_regex": "(?i)acid", "requires": "acid_groups>=1"},
        {"id": "BROMO_NAME_REQUIRES_Br", "name_regex": "(?i)bromo", "requires": "element:Br"},
        {"id": "CHLORO_NAME_REQUIRES_Cl", "name_regex": "(?i)chloro", "requires": "element:Cl"},
        {"id": "COSAN_NAME_REQUIRES_B", "name_regex": "(?i)((^|[^a-z])cosan($|[^a-z])|dicarbollide|carborane)", "requires": "element:B"},
        {"id": "ALKANOIC_ACID_CHAIN_LENGTH", "name_regex": "(?i)(hex|hept|oct|non|dec|undec|dodec)anoic acid",
         "requires": "n_carbon==alkanoic_chain"},
        {"id": "DGA_NAME_REQUIRES_DIGLYCOLYL", "name_regex": "DGA", "requires": "smarts:[#7]-[#6](=[#8])~[#6]~[#8,#16]~[#6]~[#6](=[#8])-[#7]",
         "note": "a diglycolAMIDE (or S-bridged thiodiglycolamide): both carbonyls must be amides, so thiodiglycolic acid under 'TDGA' fails"},
        {"id": "DIPIC_NAME_REQUIRES_PYRIDINE_DICARBOXAMIDE", "name_regex": "(?i)dipic(?!olinic)",
         "requires": "smarts:[#7X3]-[CX3](=[OX1])-c1cccc(-[CX3](=[OX1])-[#7X3])n1",
         "note": "'DIPIC' read as dipicolinamide = pyridine-2,6-dicarboxamide (INFERRED); dipicolinic acid names are excluded"},
        {"id": "BT_P_NAME_REQUIRES_TRIAZINE", "name_regex": "(BTBP|BTP|BTPhen|BTTP)", "requires": "triazine_N>=1"},
        {"id": "PHEN_NAME_REQUIRES_AROMATIC_N", "name_regex": "(PHEN|BTPhen|phenanthrolin)", "requires": "aromatic_N>=2"},
        {"id": "CMPO_NAME_REQUIRES_P_AND_AMIDE", "name_regex": "CMPO", "requires": "smarts:P(=O)-C-C(=O)N"},
        {"id": "P_EXTRACTANT_NAME_REQUIRES_P", "name_regex": "(?i)^(TBP|TOPO|HDEHP|D2EHPA|PC-?88A|P507|P204|Cyanex|HEH\\[EHP\\]|Ionquest)", "requires": "element:P",
         "note": "DEHPA is deliberately absent: in the monoamide literature DEHPA = N,N-di(2-ethylhexyl)propanamide, which is what the archive stores (the D2EHPA name trap)"},
        {"id": "ALCOHOL_NAME_REQUIRES_OH", "name_regex": "(?i)(alcohol|diol|tetraol|anol$|6OH)", "requires": "hydroxyl_O>=1"},
        {"id": "SULFONATE_NAME_REQUIRES_S", "name_regex": "(SO3|(?i:sulfon))", "requires": "element:S"},
    ],
}

_ALKANOIC_CHAIN = {"hex": 6, "hept": 7, "oct": 8, "non": 9, "dec": 10, "undec": 11, "dodec": 12}

#: Name-only components (no SMILES in the archive).  ``smiles`` is set only when the name denotes one
#: structure beyond reasonable doubt; ``in_corpus`` means "use the structure that owns this name in
#: the archive itself".  Nothing here is a measured value.
NAME_ONLY_LOOKUP: dict[str, dict[str, Any]] = {
    "TBP": {"in_corpus": True, "basis": "NAME_OWNED_BY_CORPUS_STRUCTURE (INFERRED)"},
    "DHOA": {"in_corpus": True, "basis": "NAME_OWNED_BY_CORPUS_STRUCTURE (INFERRED)"},
    "DOHyA": {"in_corpus": True, "basis": "NAME_OWNED_BY_CORPUS_STRUCTURE (INFERRED)"},
    "NaNO3": {"in_corpus": True, "basis": "NAME_OWNED_BY_CORPUS_STRUCTURE (INFERRED)"},
    "HDEHP": {"smiles": "CCCCC(CC)COP(=O)(O)OCC(CC)CCCC", "family": "phosphoric_acid",
              "basis": "REFERENCE_STRUCTURE: HDEHP = D2EHPA = bis(2-ethylhexyl) hydrogen phosphate (reference SMILES from the gen19 task brief); not present as a structure anywhere in the archive"},
    "1-octanol": {"smiles": "CCCCCCCCO", "family": "aliphatic_alcohol", "basis": "IUPAC_NAME_UNAMBIGUOUS"},
    "ndecanol": {"smiles": "CCCCCCCCCCO", "family": "aliphatic_alcohol",
                 "basis": "NAME_LOOKUP (INFERRED: 'ndecanol' read as n-decanol = 1-decanol)"},
    "octanol": {"smiles": None, "family": "aliphatic_alcohol",
                "basis": "NAME_LOOKUP: isomer not specified (archive keeps 'octanol' apart from '1-octanol'); no SMILES"},
    "isodecanol": {"smiles": None, "family": "aliphatic_alcohol", "basis": "NAME_LOOKUP: isomer mixture; no SMILES"},
    "isododecanol": {"smiles": None, "family": "aliphatic_alcohol", "basis": "NAME_LOOKUP: isomer mixture; no SMILES"},
    "isodecyl": {"smiles": None, "family": "aliphatic_alcohol",
                 "basis": "NAME_LOOKUP (INFERRED: truncated 'isodecyl' read as isodecanol modifier); no SMILES"},
}

#: Occurrence-level chemistry that the recorded structure cannot carry (brief section 5: known
#: chemistry takes priority).  Applied only when the named check fails for the recorded structure.
KNOWN_NAME_OVERRIDES: dict[str, dict[str, Any]] = {
    "Br-Cosan": {"requires_failed_check": "COSAN_NAME_REQUIRES_B", "family_by_name": "metallacarborane_anion",
                 "acidity_class": "ACIDIC", "mechanism": "ACIDIC_CATION_EXCHANGE",
                 "basis": "KNOWN_NAME_OVERRIDE: brominated cobalt bis(dicarbollide) (COSAN) is a lipophilic "
                          "anion used in its H+ form as a cation-exchange synergist; the archive structure "
                          "(a chloroacetanilide without B or Co) cannot represent it (INFERRED name-resolution error)"},
}


#: Named industrial / benchmark extractants used by the presence audit and the tests.  Structures are
#: definitional (the compound the name denotes), not measured values; ``basis`` says where each
#: SMILES comes from.  ``aliases`` are the search terms of the presence audit.
REFERENCE_EXTRACTANTS: dict[str, dict[str, Any]] = {
    "D2EHPA": {"smiles": "CCCCC(CC)COP(=O)(O)OCC(CC)CCCC", "expected_family": "phosphoric_acid",
               "aliases": ["D2EHPA", "HDEHP", "P204", "DEHPA"],
               "basis": "gen19 task brief: bis(2-ethylhexyl) hydrogen phosphate"},
    "PC88A": {"smiles": "CCCCC(CC)COP(=O)(O)CC(CC)CCCC", "expected_family": "phosphonic_acid_monoester",
              "aliases": ["PC88A", "PC-88A", "P507", "EHEHPA", "HEH[EHP]", "Ionquest 801"],
              "basis": "gen19 task brief: 2-ethylhexyl hydrogen (2-ethylhexyl)phosphonate"},
    "Cyanex 272": {"smiles": "CC(CC(C)(C)C)CP(=O)(O)CC(C)CC(C)(C)C", "expected_family": "phosphinic_acid",
                   "aliases": ["Cyanex 272", "Cyanex272"],
                   "basis": "gen19 task brief: bis(2,4,4-trimethylpentyl)phosphinic acid"},
    "TODGA": {"smiles": "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC", "expected_family": "diglycolamide",
              "aliases": ["TODGA"], "basis": "N,N,N',N'-tetraoctyldiglycolamide (definitional)"},
    "TBP": {"smiles": "CCCCOP(=O)(OCCCC)OCCCC", "expected_family": "neutral_organophosphate",
            "aliases": ["TBP"], "basis": "tri-n-butyl phosphate (definitional)"},
    "TOPO": {"smiles": "CCCCCCCCP(=O)(CCCCCCCC)CCCCCCCC", "expected_family": "phosphine_oxide",
             "aliases": ["TOPO"], "basis": "tri-n-octylphosphine oxide (definitional)"},
    "CMPO": {"smiles": "CCCCCCCCP(=O)(CC(=O)N(CC(C)C)CC(C)C)c1ccccc1",
             "expected_family": "carbamoylmethylphosphine_oxide", "aliases": ["CMPO"],
             "basis": "octyl(phenyl)-N,N-diisobutylcarbamoylmethylphosphine oxide (definitional)"},
    "HTTA": {"smiles": "O=C(CC(=O)C(F)(F)F)c1cccs1", "expected_family": "beta_diketone", "aliases": ["HTTA"],
             "basis": "2-thenoyltrifluoroacetone, diketo tautomer (definitional)"},
    "Aliquat 336": {"smiles": "C[N+](CCCCCCCC)(CCCCCCCC)CCCCCCCC.[Cl-]", "expected_family": "quaternary_ammonium",
                    "aliases": ["Aliquat"],
                    "basis": "methyltrioctylammonium chloride, idealised main component of a C8/C10 commercial mixture (INFERRED)"},
}


# ----------------------------------------------------------------------------------------------
# Rules file I/O
# ----------------------------------------------------------------------------------------------
def rules_digest(rules: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()


def load_family_rules(path: Path | None = None) -> dict[str, Any]:
    """The rules the classifier uses: ``descriptors/family_rules.json`` when it exists, otherwise a
    deep copy of :data:`FAMILY_RULES_SPEC` (the file is written from the spec by the build script;
    ``tests/test_ligands.py`` asserts the two agree)."""
    p = Path(path) if path is not None else FAMILY_RULES_PATH
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: v for k, v in data.items() if k in FAMILY_RULES_SPEC}
    if path is not None:
        raise FileNotFoundError(p)
    return copy.deepcopy(FAMILY_RULES_SPEC)


# ----------------------------------------------------------------------------------------------
# Molecule helpers
# ----------------------------------------------------------------------------------------------
def _is_missing(x: object) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    return isinstance(x, str) and not x.strip()


@lru_cache(maxsize=4096)
def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if _is_missing(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def canonical_smiles(smiles: str) -> str | None:
    m = mol_from_smiles(smiles)
    return None if m is None else Chem.MolToSmiles(m)


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(frags, key=lambda f: (f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)))


@lru_cache(maxsize=4096)
def parent_key(smiles: str) -> str | None:
    """Connectivity key that ignores salts, charges and stereo: first InChIKey block of the
    neutralised, stereo-stripped largest fragment.  Two canonical SMILES with the same parent key are
    forms of one ligand (DTPA vs DTPA.nitrate, HEDTA vs HEDTA(3-), CDTA with/without stereo)."""
    m = mol_from_smiles(smiles)
    if m is None:
        return None
    frag = Chem.Mol(largest_fragment(m))
    try:
        frag = rdMolStandardize.Uncharger().uncharge(frag)
    except Exception:  # noqa: BLE001 -- keep the charged fragment rather than fail
        pass
    Chem.RemoveStereochemistry(frag)
    key = Chem.MolToInchiKey(frag)
    return key.split("-")[0] if key else None


@lru_cache(maxsize=4096)
def parent_isomeric_smiles(smiles: str) -> str | None:
    """Canonical isomeric SMILES of the neutralised largest fragment (stereo kept)."""
    m = mol_from_smiles(smiles)
    if m is None:
        return None
    frag = Chem.Mol(largest_fragment(m))
    try:
        frag = rdMolStandardize.Uncharger().uncharge(frag)
    except Exception:  # noqa: BLE001
        pass
    return Chem.MolToSmiles(frag)


def _has_stereo(smiles: str | None) -> bool:
    return smiles is not None and ("@" in smiles or "/" in smiles or "\\" in smiles)


def _triazine_ring_atoms(mol: Chem.Mol) -> set[int]:
    out: set[int] = set()
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            if sum(mol.GetAtomWithIdx(i).GetSymbol() == "N" for i in ring) >= 3:
                out.update(ring)
    return out


@lru_cache(maxsize=64)
def _patt(smarts: str) -> Chem.Mol:
    p = Chem.MolFromSmarts(smarts)
    if p is None:
        raise ValueError(f"invalid SMARTS: {smarts}")
    return p


def _donor_atoms(mol: Chem.Mol) -> dict[str, list[int]]:
    triazine = _triazine_ring_atoms(mol)
    out: dict[str, list[int]] = {}
    for key, spec in DONOR_SMARTS.items():
        idx = sorted({m[0] for m in mol.GetSubstructMatches(_patt(spec["smarts"]), uniquify=True)})
        if key == "aromatic_N_pyridine_type":
            idx = [i for i in idx if i not in triazine]
        elif key == "triazine_N":
            idx = [i for i in idx if i in triazine]
        out[key] = idx
    return out


def donor_census(mol: Chem.Mol) -> dict[str, int]:
    """Donor-atom counts per :data:`DONOR_SMARTS` key (``carboxylic_acid`` counted per group)."""
    atoms = _donor_atoms(mol)
    census = {k: len(v) for k, v in atoms.items()}
    census["carboxylic_acid_groups"] = len(mol.GetSubstructMatches(
        _patt("[CX3](=[OX1])[OX2H1,OX1-]"), uniquify=True))
    census.pop("carboxylic_acid_O", None)
    return census


def donor_sites(mol: Chem.Mol) -> list[frozenset[int]]:
    """Donor atoms collapsed into sites (see module docstring, rule 2)."""
    atoms = _donor_atoms(mol)
    site_atoms: set[int] = set()
    for key, idx in atoms.items():
        if DONOR_SMARTS[key]["site"]:
            site_atoms.update(idx)
    groups: dict[tuple[str, int], set[int]] = defaultdict(set)
    ring_of: dict[int, int] = {}
    rings = mol.GetRingInfo().AtomRings()
    for r_i, ring in enumerate(rings):
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            for i in ring:
                ring_of.setdefault(i, r_i)
    for a in sorted(site_atoms):
        atom = mol.GetAtomWithIdx(a)
        key: tuple[str, int] = ("atom", a)
        if atom.GetSymbol() in ("O", "S") and atom.GetDegree() == 1:
            nb = atom.GetNeighbors()[0]
            if nb.GetSymbol() == "P" or (nb.GetSymbol() == "C" and _is_carboxyl_c(mol, nb.GetIdx())):
                key = ("center", nb.GetIdx())
        elif atom.GetSymbol() == "O" and atom.GetDegree() == 2 and atom.GetTotalNumHs() == 1:
            nb = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1][0]
            if nb.GetSymbol() == "P" or (nb.GetSymbol() == "C" and _is_carboxyl_c(mol, nb.GetIdx())):
                key = ("center", nb.GetIdx())
        elif atom.GetSymbol() == "S" and atom.GetDegree() == 1:
            pass
        if atom.GetIsAromatic() and atom.GetSymbol() == "N" and a in ring_of:
            key = ("ring", ring_of[a])
        groups[key].add(a)
    return [frozenset(v) for _, v in sorted(groups.items())]


def _is_carboxyl_c(mol: Chem.Mol, c_idx: int) -> bool:
    c = mol.GetAtomWithIdx(c_idx)
    os_ = [n for n in c.GetNeighbors() if n.GetSymbol() == "O"]
    return len(os_) == 2 and any(
        b.GetBondType() == Chem.BondType.DOUBLE for b in c.GetBonds()
        if b.GetOtherAtom(c).GetSymbol() == "O")


def _transannular(mol: Chem.Mol, a: int, b: int, small_rings: list[set[int]]) -> bool:
    """True when the shortest a-b path uses >= 3 bonds inside one ring of <= 7 atoms."""
    path = Chem.GetShortestPath(mol, a, b)
    for ring in small_rings:
        in_ring_bonds = sum(1 for u, v in zip(path, path[1:]) if u in ring and v in ring)
        if in_ring_bonds >= 3:
            return True
    return False


def denticity(mol: Chem.Mol) -> tuple[int, int, int]:
    """``(n_donor_sites, denticity_proxy, n_chelating_units)`` -- rule in the module docstring."""
    sites = donor_sites(mol)
    if not sites:
        return 0, 0, 0
    dm = Chem.GetDistanceMatrix(mol)
    small_rings = [set(r) for r in mol.GetRingInfo().AtomRings() if len(r) <= 7]
    n = len(sites)
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            pairs = [(a, b) for a in sites[i] for b in sites[j] if dm[a][b] in (3, 4)]
            if any(not _transannular(mol, a, b, small_rings) for a, b in pairs):
                adj[i][j] = adj[j][i] = True
    seen = [False] * n
    sizes: list[int] = []
    for s in range(n):
        if seen[s]:
            continue
        stack, size = [s], 0
        seen[s] = True
        while stack:
            u = stack.pop()
            size += 1
            for v in range(n):
                if adj[u][v] and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        sizes.append(size)
    return n, max(sizes), sum(1 for x in sizes if x >= 2)


_ACID_SMARTS = (
    "[OX2H1,OX1-]-[PX4]", "[SX2H1,SX1-]-[PX4]", "[CX3](=[OX1])[OX2H1,OX1-]",
    "[SX4](=[OX1])(=[OX1])[OX2H1,OX1-]",
    "[#6]-[CX3](=[OX1])-[CX4;H1,H2]-[CX3](=[OX1])-[#6]", "[#6X3]=[NX2]-[OX2H1]", "[OX2H1]-c",
    "[#6]-[CX3](=[OX1])-[#6;R1]1~[#6;R1](~[OX1,OX2H1])~[#7;R1]~[#7;R1]~[#6;R1]~1",   # 4-acyl-5-pyrazolone (HPMBP)
)
_ACID_GROUP_SMARTS = ("[PX4]-[OX2H1,OX1-,SX2H1,SX1-]", "[CX3](=[OX1])[OX2H1,OX1-]",
                      "[SX4](=[OX1])(=[OX1])[OX2H1,OX1-]")


def acid_group_count(mol: Chem.Mol) -> int:
    """Number of Bronsted acid heads (P-OH/P-SH per P, COOH, SO3H) -- phenols/enols excluded."""
    centers: set[int] = set()
    for s in _ACID_GROUP_SMARTS:
        for m in mol.GetSubstructMatches(_patt(s), uniquify=True):
            centers.add(m[0])
    return len(centers)


def acidity_class(mol: Chem.Mol | None) -> str:
    """``IONIC`` if the largest fragment carries a net charge; else ``ACIDIC`` for any P-OH/P-SH,
    COOH, SO3H, beta-diketone, oxime-OH or phenol; else ``BASIC`` for an aliphatic amine (aromatic N
    is treated as neutral, as in solvent-extraction usage); else ``NEUTRAL``.  ``UNKNOWN`` without a
    structure."""
    if mol is None:
        return "UNKNOWN"
    frag = largest_fragment(mol)
    if sum(a.GetFormalCharge() for a in frag.GetAtoms()) != 0:
        return "IONIC"
    if any(frag.HasSubstructMatch(_patt(s)) for s in _ACID_SMARTS):
        return "ACIDIC"
    if frag.HasSubstructMatch(_patt(
            "[NX3;!$(N-[#6X3]=[O,S,N]);!$(N-a);!$(N-[#7,#8,#15,#16])]-[#6]")):
        return "BASIC"
    return "NEUTRAL"


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def rdkit_descriptors(mol: Chem.Mol) -> dict[str, Any]:
    frag = largest_fragment(mol)
    return {
        "mw": round(Descriptors.MolWt(mol), 4),
        "mol_logp": round(Crippen.MolLogP(mol), 4),
        "mol_logp_largest_fragment": round(Crippen.MolLogP(frag), 4),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 4),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "n_fragments": len(Chem.GetMolFrags(mol)),
        "net_formal_charge": int(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "inchikey": Chem.MolToInchiKey(mol) or None,
        "morgan_r2_2048_onbits": " ".join(str(b) for b in sorted(_MORGAN.GetFingerprint(mol).GetOnBits())),
    }


def aromatic_n_chelation(mol: Chem.Mol) -> tuple[int, int]:
    """``(aromatic_N_chelate_sites, carboxyl_O_near_aromatic_N)``.

    Ring-N sites are the aromatic N donor atoms grouped per aromatic ring (rule 2 of the module
    docstring).  Two ring sites chelate together when some pair of their N atoms is 3 or 4 bonds apart
    (bipyridyl / terpyridyl-type pockets); the first value is the size of the largest connected set of
    such sites.  The second value counts carboxylic-acid O atoms 3 or 4 bonds from any aromatic N donor,
    i.e. acids that could close a 5/6-membered chelate with a ring N (picolinic-acid type)."""
    atoms = _donor_atoms(mol)
    arom_n = sorted(set(atoms["aromatic_N_pyridine_type"]) | set(atoms["triazine_N"]))
    if not arom_n:
        return 0, 0
    dm = Chem.GetDistanceMatrix(mol)
    near = sum(1 for o in atoms["carboxylic_acid_O"] if any(dm[o][n] in (3, 4) for n in arom_n))
    ring_of: dict[int, int] = {}
    for r_i, ring in enumerate(mol.GetRingInfo().AtomRings()):
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            for i in ring:
                ring_of.setdefault(i, r_i)
    groups: dict[Any, list[int]] = defaultdict(list)
    for n in arom_n:
        groups[ring_of.get(n, ("atom", n))].append(n)
    sites = list(groups.values())
    k = len(sites)
    adj = [[any(dm[a][b] in (3, 4) for a in sites[i] for b in sites[j]) if i != j else False for j in range(k)]
           for i in range(k)]
    seen, best = [False] * k, 0
    for s in range(k):
        if seen[s]:
            continue
        stack, size = [s], 0
        seen[s] = True
        while stack:
            u = stack.pop()
            size += 1
            for v in range(k):
                if adj[u][v] and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        best = max(best, size)
    return best, near


def _rule_descriptors(mol: Chem.Mol) -> dict[str, float]:
    frag = largest_fragment(mol)
    chelate_sites, carboxyl_near = aromatic_n_chelation(mol)
    rings = mol.GetRingInfo().AtomRings()
    big = max(rings, key=len) if rings else ()
    atoms = [mol.GetAtomWithIdx(i) for i in big]
    census = donor_census(mol)
    n_sites, _, _ = denticity(mol)
    return {
        "n_carbon": sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms()),
        "heavy_atoms_largest_fragment": frag.GetNumHeavyAtoms(),
        "logp_largest_fragment": Crippen.MolLogP(frag),
        "largest_ring_size": len(big),
        "macro_ring_O": sum(a.GetSymbol() == "O" for a in atoms) if len(big) >= 12 else 0,
        "macro_ring_aromatic": sum(a.GetIsAromatic() for a in atoms) if len(big) >= 12 else 0,
        "macro_ring_CH2": sum(a.GetSymbol() == "C" and not a.GetIsAromatic() and a.GetTotalNumHs() == 2
                              for a in atoms) if len(big) >= 12 else 0,
        "n_donor_sites": n_sites,
        "aliphatic_OH": len(mol.GetSubstructMatches(_patt("[OX2H1]-[CX4]"), uniquify=True)),
        "aromatic_N": census["aromatic_N_pyridine_type"] + census["triazine_N"],
        "acid_groups": acid_group_count(mol),
        "charged_or_multifragment": int(len(Chem.GetMolFrags(mol)) > 1
                                        or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0),
        "aromatic_N_chelate_sites": chelate_sites,
        "carboxyl_O_near_aromatic_N": carboxyl_near,
    }


# ----------------------------------------------------------------------------------------------
# Family classifier
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilyAssignment:
    family: str
    core_family: str
    fired: tuple[str, ...]
    candidates: tuple[str, ...]
    ambiguous: bool
    overlay: str | None
    rules_version: str

    @property
    def status(self) -> str:
        return "AMBIGUOUS" if self.ambiguous else "UNIQUE"


class FamilyClassifier:
    """Ordered SMARTS family rules read from ``family_rules.json`` (or a rules dict)."""

    def __init__(self, rules: Mapping[str, Any] | None = None):
        self.rules = copy.deepcopy(dict(rules)) if rules is not None else load_family_rules()
        if self.rules.get("schema") != RULES_SCHEMA:
            raise ValueError(f"unexpected rules schema {self.rules.get('schema')!r}")
        self.version = self.rules["version"]
        self.families = [f["name"] for f in self.rules["families"]]
        if len(set(self.families)) != len(self.families):
            raise ValueError("duplicate family names in rules")
        for fam in self.rules["families"]:
            for clause in fam["any_of"]:
                for pred in clause:
                    if "smarts" in pred:
                        _patt(pred["smarts"])
        self._cache: dict[str, FamilyAssignment] = {}

    @classmethod
    def from_json(cls, path: Path) -> "FamilyClassifier":
        return cls(load_family_rules(path))

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.families) + ("other",)

    @staticmethod
    def _clause_holds(clause: Iterable[Mapping[str, Any]], mol: Chem.Mol, desc: dict[str, float]) -> bool:
        for pred in clause:
            if "smarts" in pred:
                val = len(mol.GetSubstructMatches(_patt(pred["smarts"]), uniquify=True))
            else:
                val = desc[pred["descriptor"]]
            if "min" in pred and pred["min"] is not None and val < pred["min"]:
                return False
            if "max" in pred and pred["max"] is not None and val > pred["max"]:
                return False
        return True

    def _fires(self, fam: Mapping[str, Any], mol: Chem.Mol, desc: dict[str, float]) -> bool:
        return any(self._clause_holds(clause, mol, desc) for clause in fam["any_of"])

    def classify(self, smiles: str) -> FamilyAssignment:
        key = str(smiles)
        if key in self._cache:
            return self._cache[key]
        mol = mol_from_smiles(key)
        if mol is None:
            res = FamilyAssignment("other", "other", (), (), False, None, self.version)
            self._cache[key] = res
            return res
        desc = _rule_descriptors(mol)
        fired = [f for f in self.rules["families"] if self._fires(f, mol, desc)]
        overlays = [f for f in fired if f.get("overlay")]
        core = [f for f in fired if not f.get("overlay")]
        core_names = [f["name"] for f in core]
        subsumed: set[str] = set()
        for f in core:
            subs = f.get("subsumes", [])
            subsumed.update(n for n in core_names if n != f["name"] and (n in subs or _ALL_CORE in subs))
            for cond in f.get("subsumes_when", []):
                if cond["rule"] in core_names and cond["rule"] != f["name"] \
                        and self._clause_holds(cond["all_of"], mol, desc):
                    subsumed.add(cond["rule"])
        candidates = tuple(n for n in core_names if n not in subsumed)
        core_family = candidates[0] if candidates else "other"
        overlay = overlays[0]["name"] if overlays else None
        res = FamilyAssignment(
            family=overlay or core_family, core_family=core_family,
            fired=tuple(f["name"] for f in fired), candidates=candidates,
            ambiguous=len(candidates) > 1, overlay=overlay, rules_version=self.version)
        self._cache[key] = res
        return res


@lru_cache(maxsize=1)
def get_classifier() -> FamilyClassifier:
    return FamilyClassifier()


# ----------------------------------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------------------------------
_NAME_KEY_STRIP = re.compile(r"[\s\-_‐-―]+")


def normalize_name_key(name: object) -> str | None:
    """Case / hyphen / space / underscore-insensitive key.  Digits, brackets and locants are kept, so
    ``1-octanol`` and ``octanol`` stay different keys (the archive keeps them apart)."""
    if _is_missing(name):
        return None
    key = _NAME_KEY_STRIP.sub("", str(name).strip().casefold())
    return key or None


def name_structure_check_failures(name: object, mol: Chem.Mol | None,
                                  rules: Mapping[str, Any] | None = None) -> list[str]:
    """IDs of the plausibility checks (``name_structure_checks`` in the rules) that ``name`` fails
    for ``mol``.  Heuristic by construction: a failure is a flag, never a correction."""
    if _is_missing(name) or mol is None:
        return []
    rules = rules if rules is not None else get_classifier().rules
    census = donor_census(mol)
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    fails = []
    for chk in rules.get("name_structure_checks", []):
        m = re.search(chk["name_regex"], str(name))
        if not m:
            continue
        req = chk["requires"]
        if req.startswith("element:"):
            ok = req.split(":", 1)[1] in elements
        elif req.startswith("smarts:"):
            ok = mol.HasSubstructMatch(_patt(req.split(":", 1)[1]))
        elif req == "acid_groups>=1":
            ok = acid_group_count(mol) >= 1
        elif req == "n_carbon==alkanoic_chain":
            want = _ALKANOIC_CHAIN[m.group(1).lower()]
            ok = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms()) == want
        elif req == "triazine_N>=1":
            ok = census["triazine_N"] >= 1
        elif req == "aromatic_N>=2":
            ok = census["aromatic_N_pyridine_type"] + census["triazine_N"] >= 2
        elif req == "hydroxyl_O>=1":
            ok = census["hydroxyl_O"] >= 1
        else:
            raise ValueError(f"unknown requirement {req}")
        if not ok:
            fails.append(chk["id"])
    return fails


@dataclass
class Occurrence:
    """Aggregated (name, structure) occurrences across component entries."""
    name: str | None
    smiles: str | None
    n_entries: int = 0
    n_rows: set = field(default_factory=set)
    n_model_rows: set = field(default_factory=set)
    masking_rows: set = field(default_factory=set)
    roles: Counter = field(default_factory=Counter)


class AliasIndex:
    """Name <-> structure bookkeeping over the archive component entries.

    ``add(row_id, is_model, components)`` for every archive row, then ``finalize()``.

    * a *masking row* is a row of the ``ST*.json`` sub-source pattern: it carries an unnamed aqueous
      agent slot (every archive ``aqueous_holdback`` entry is unnamed; 777 of them are empty
      zero-concentration slots).  The caller may pass ``masking_row`` explicitly.
    * a name on structure S is a *masking-agent collision* when every one of its entries on S sits in
      a masking row and S has another name used outside such rows (TODGA's structure named
      ``SO3-Ph-BTP``, ``PHEN-dialcohol`` ...).
    * ``owner(name)`` -- the structure a name denotes: the most frequent structure among those that
      pass the name-structure checks and on which the name is not a masking-agent collision (ties
      broken by canonical SMILES; ``None`` if no structure qualifies).
    * ``canonical_name(S)`` -- most frequent MODEL-row name that S owns, passes the checks and is not
      a masking-agent collision; falls back to all rows, then to any name (status recorded).
    * ``resolve(name, smiles)`` -- the canonical name when ``normalize_name_key(name)`` equals the
      key of an accepted alias of the *same* structure; ``None`` otherwise.
    """

    def __init__(self, rules: Mapping[str, Any] | None = None):
        self.rules = rules
        self.occ: dict[tuple[str | None, str | None], Occurrence] = {}
        self._final = False

    @staticmethod
    def is_masking_row(components: Iterable[Mapping[str, Any]]) -> bool:
        return any(c.get("role") in AQUEOUS_ROLES and _is_missing(c.get("name")) for c in components)

    def add(self, row_id: object, is_model: bool, components: Iterable[Mapping[str, Any]],
            masking_row: bool | None = None) -> None:
        comps = list(components)
        if masking_row is None:
            masking_row = self.is_masking_row(comps)
        for c in comps:
            name = None if _is_missing(c.get("name")) else str(c["name"])
            smi = None if _is_missing(c.get("smiles_canonical")) else str(c["smiles_canonical"])
            o = self.occ.setdefault((name, smi), Occurrence(name, smi))
            o.n_entries += 1
            o.n_rows.add(row_id)
            if is_model:
                o.n_model_rows.add(row_id)
            if masking_row and c.get("role") == "organic_extractant":
                o.masking_rows.add(row_id)
            o.roles[c.get("role")] += 1

    def finalize(self) -> "AliasIndex":
        self.by_name: dict[str, list[Occurrence]] = defaultdict(list)
        self.by_smiles: dict[str, list[Occurrence]] = defaultdict(list)
        for (name, smi), o in sorted(self.occ.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            if name is not None:
                self.by_name[name].append(o)
            if smi is not None:
                self.by_smiles[smi].append(o)
        self.check_fail: dict[tuple[str, str], list[str]] = {}
        for (name, smi) in self.occ:
            if name is not None and smi is not None:
                self.check_fail[(name, smi)] = name_structure_check_failures(
                    name, mol_from_smiles(smi), self.rules)
        self._owner: dict[str, str | None] = {}
        for name, occs in self.by_name.items():
            ok = [o for o in occs if o.smiles is not None and not self.check_fail[(name, o.smiles)]
                  and not self.is_masking_collision(name, o.smiles)]
            ok.sort(key=lambda o: (-len(o.n_rows), o.smiles))
            self._owner[name] = ok[0].smiles if ok else None
        self._final = True
        self._canon: dict[str, tuple[str | None, str]] = {}
        for smi in self.by_smiles:
            self._canon[smi] = self._pick_canonical(smi)
        return self

    def owner(self, name: str) -> str | None:
        return self._owner.get(name)

    def is_masking_collision(self, name: str, smiles: str) -> bool:
        """All entries of ``name`` on ``smiles`` sit in masking rows, the structure carries another
        name outside masking rows, and it carries >= 2 distinct names confined to masking rows (the
        signature of the sub-source that writes agent names into the extractant slot; a structure
        with one consistent name in those rows is left alone)."""
        o = self.occ.get((name, smiles))
        if o is None or not o.masking_rows or o.masking_rows != o.n_rows:
            return False
        occs = [x for x in self.by_smiles[smiles] if x.name is not None]
        outside = any(x.name != name and (x.n_rows - x.masking_rows) for x in occs)
        confined = sum(1 for x in occs if x.masking_rows and x.masking_rows == x.n_rows)
        return outside and confined >= 2

    @staticmethod
    def same_ligand(a: str | None, b: str | None) -> bool:
        """Equal canonical SMILES, or salt / charge forms of one ligand (DTPA vs DTPA.nitrate, HEDTA vs
        HEDTA(3-)), or one stereo-unspecified drawing of it (CDTA with / without stereo).  Two
        *different* specified stereoisomers (p-TODGA-Syn vs -Anti) are different ligands."""
        if a is None or b is None:
            return False
        if a == b:
            return True
        if parent_key(a) is None or parent_key(a) != parent_key(b):
            return False
        ia, ib = parent_isomeric_smiles(a), parent_isomeric_smiles(b)
        return ia == ib or not _has_stereo(ia) or not _has_stereo(ib)

    def accepted(self, name: str, smiles: str) -> bool:
        return (self.same_ligand(self._owner.get(name), smiles) and not self.check_fail.get((name, smiles))
                and not self.is_masking_collision(name, smiles))

    def rejection_reason(self, name: str, smiles: str) -> str | None:
        if self.check_fail.get((name, smiles)):
            return "NAME_STRUCTURE_CHECK_FAILED:" + "+".join(self.check_fail[(name, smiles)])
        if self.is_masking_collision(name, smiles):
            return "MASKING_AGENT_NAME"
        own = self._owner.get(name)
        if not self.same_ligand(own, smiles):
            return "NAME_OWNED_BY_OTHER_STRUCTURE" if own else "NAME_HAS_NO_PLAUSIBLE_STRUCTURE"
        return None

    def _pick_canonical(self, smiles: str) -> tuple[str | None, str]:
        occs = [o for o in self.by_smiles[smiles] if o.name is not None]
        if not occs:
            return None, "NO_NAME"
        acc = [o for o in occs if self.accepted(o.name, smiles)]
        for pool, count, status in ((acc, lambda o: len(o.n_model_rows), "MODEL_ROWS"),
                                    (acc, lambda o: len(o.n_rows), "ALL_ROWS")):
            pool = [o for o in pool if count(o) > 0]
            if pool:
                best = max(count(o) for o in pool)
                top = sorted(o.name for o in pool if count(o) == best)
                return top[0], status + ("_TIE" if len(top) > 1 else "")
        best = max(len(o.n_rows) for o in occs)
        top = sorted(o.name for o in occs if len(o.n_rows) == best)
        return top[0], "FALLBACK_NO_ACCEPTED_NAME" + ("_TIE" if len(top) > 1 else "")

    def canonical_name(self, smiles: str) -> str | None:
        return self._canon.get(smiles, (None, "NO_NAME"))[0]

    def canonical_name_status(self, smiles: str) -> str:
        return self._canon.get(smiles, (None, "NO_NAME"))[1]

    def aliases(self, smiles: str) -> list[tuple[str, int, int]]:
        """``(name, n_rows, n_model_rows)`` for every name seen on the structure, most rows first."""
        occs = [o for o in self.by_smiles.get(smiles, []) if o.name is not None]
        return sorted(((o.name, len(o.n_rows), len(o.n_model_rows)) for o in occs),
                      key=lambda t: (-t[1], t[0]))

    def resolve(self, name: object, smiles: object) -> str | None:
        key = normalize_name_key(name)
        if key is None or _is_missing(smiles):
            return None
        smiles = str(smiles)
        if smiles not in self.by_smiles:
            smiles = canonical_smiles(smiles) or smiles
        for o in self.by_smiles.get(smiles, []):
            if o.name is not None and normalize_name_key(o.name) == key and self.accepted(o.name, smiles):
                return self.canonical_name(smiles)
        return None

    def name_implied_structure(self, name: str | None, smiles: str | None) -> str | None:
        """The structure a component *name* points to: its own structure when the name is accepted
        there, the owner structure when the name belongs elsewhere, ``None`` when unresolvable."""
        if name is None:
            return smiles
        if smiles is not None and self.accepted(name, smiles):
            return smiles
        if smiles is not None and self.is_masking_collision(name, smiles):
            return smiles
        return self._owner.get(name)


# ----------------------------------------------------------------------------------------------
# One component structure -> one descriptor record
# ----------------------------------------------------------------------------------------------
def component_id(smiles: str | None, name: str | None = None) -> str:
    if smiles:
        return "cmp_" + hashlib.sha1(smiles.encode()).hexdigest()[:10]
    if name:
        return "nam_" + hashlib.sha1(name.encode()).hexdigest()[:10]
    return "nam_empty_placeholder"


def describe_structure(smiles: str, classifier: FamilyClassifier | None = None) -> dict[str, Any]:
    """Descriptors, donor census, denticity, acidity and family for one SMILES."""
    clf = classifier or get_classifier()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {"parse_ok": False, "family": "other", "core_family": "other", "acidity_class": "UNKNOWN"}
    fa = clf.classify(smiles)
    n_sites, dent, n_units = denticity(mol)
    rec: dict[str, Any] = {"parse_ok": True}
    rec.update(rdkit_descriptors(mol))
    rec["parent_connectivity_key"] = parent_key(smiles)
    rec.update({f"donor_{k}": v for k, v in donor_census(mol).items()})
    rec.update({"n_donor_sites": n_sites, "denticity_proxy": dent, "n_chelating_units": n_units,
                "acid_groups": acid_group_count(mol), "acidity_class": acidity_class(mol),
                "family": fa.family, "core_family": fa.core_family,
                "family_status": fa.status, "family_candidates": "|".join(fa.candidates),
                "family_rules_fired": "|".join(fa.fired), "family_overlay": fa.overlay,
                "family_rules_version": fa.rules_version})
    return rec


def resolve_occurrence(index: AliasIndex, name: object, smiles: object,
                       classifier: FamilyClassifier | None = None,
                       name_corrected: bool = True,
                       row_structures: Iterable[str] = ()) -> dict[str, Any] | None:
    """The chemistry one component entry stands for.

    Returns ``{"key", "smiles", "family", "core_family", "basis"}`` or ``None`` for an empty slot.
    With ``name_corrected=False`` only the recorded structure is used (``STRUCTURE``; name-only
    entries resolve by name, since there is nothing else).  With ``name_corrected=True`` (brief
    section 5: known chemistry first) two corrections apply, both only when the entry's name *fails*
    a name-structure check on the recorded structure:

    * :data:`KNOWN_NAME_OVERRIDES` (e.g. ``Br-Cosan`` recorded with a boron-free structure);
    * ``NAME_IMPLIED_STRUCTURE`` -- the structure that owns the name elsewhere in the archive
      (``2-bromodecanoic acid`` recorded with the N-DPP SMILES in positional ``ST*.json`` rows).
      Skipped when that structure is already recorded in the same row (``row_structures``): the
      names were swapped between positions but the set of structures is right
      (basis ``STRUCTURE_POSITIONAL_SWAP``).  Also skipped when the owning structure is a
      hydrophilic aqueous agent or salt (``SO3-Ph-BTP`` written on TODGA's structure): the name
      denotes the aqueous agent, the recorded organic structure stands
      (basis ``STRUCTURE_NAME_IS_AQUEOUS_AGENT``).
    """
    clf = classifier or get_classifier()
    name = None if _is_missing(name) else str(name)
    smiles = None if _is_missing(smiles) else str(smiles)
    if smiles is not None:
        fails = index.check_fail.get((name, smiles), []) if name is not None else []
        if name_corrected and name is not None and fails:
            ov = KNOWN_NAME_OVERRIDES.get(name)
            if ov is not None and ov["requires_failed_check"] in fails:
                return {"key": f"name:{name}", "smiles": None, "family": ov["family_by_name"],
                        "core_family": ov["family_by_name"], "basis": "KNOWN_NAME_OVERRIDE"}
            own = index.owner(name)
            if own is not None and not index.same_ligand(own, smiles):
                if any(index.same_ligand(own, s) for s in row_structures):
                    fa = clf.classify(smiles)
                    return {"key": smiles, "smiles": smiles, "family": fa.family, "core_family": fa.core_family,
                            "basis": "STRUCTURE_POSITIONAL_SWAP"}
                fa = clf.classify(own)
                if fa.overlay is not None:
                    fs = clf.classify(smiles)
                    return {"key": smiles, "smiles": smiles, "family": fs.family, "core_family": fs.core_family,
                            "basis": "STRUCTURE_NAME_IS_AQUEOUS_AGENT"}
                return {"key": own, "smiles": own, "family": fa.family, "core_family": fa.core_family,
                        "basis": "NAME_IMPLIED_STRUCTURE"}
        fa = clf.classify(smiles)
        return {"key": smiles, "smiles": smiles, "family": fa.family, "core_family": fa.core_family,
                "basis": "STRUCTURE"}
    if name is None:
        return None
    look = NAME_ONLY_LOOKUP.get(name)
    if look is not None and not look.get("in_corpus"):
        if look.get("smiles"):
            fa = clf.classify(look["smiles"])
            return {"key": canonical_smiles(look["smiles"]), "smiles": canonical_smiles(look["smiles"]),
                    "family": fa.family, "core_family": fa.core_family, "basis": "NAME_ONLY_LOOKUP"}
        return {"key": f"name:{name}", "smiles": None, "family": look["family"], "core_family": look["family"],
                "basis": "NAME_ONLY_LOOKUP_NO_SMILES"}
    own = index.owner(name)
    if own is not None:
        fa = clf.classify(own)
        return {"key": own, "smiles": own, "family": fa.family, "core_family": fa.core_family,
                "basis": "NAME_OWNED_BY_CORPUS_STRUCTURE"}
    return {"key": f"name:{name}", "smiles": None, "family": "other", "core_family": "other",
            "basis": "NAME_UNRESOLVED"}


def iter_component_dicts(components: object) -> list[dict[str, Any]]:
    """The archive ``components`` cell as a list of plain dicts (``[]`` for empty/NaN)."""
    if components is None or (isinstance(components, float) and np.isnan(components)):
        return []
    return [dict(c) for c in list(components)]
