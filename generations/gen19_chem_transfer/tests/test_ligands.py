"""Tests for ``gen19ct.chemistry.ligands`` and ``gen19ct.chemistry.mechanisms`` (brief sections 4.2, 5, 27).

Fast tests use reference SMILES and small synthetic alias indexes.  The two full-archive tests
(every component SMILES gets exactly one family label; the built tables balance) are marked ``slow``.
"""
from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import ligands as L
from gen19ct.chemistry import mechanisms as M

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
TEHDGA = "CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC"
D2EHPA = "CCCCC(CC)COP(=O)(O)OCC(CC)CCCC"
PC88A = "CCCCC(CC)COP(=O)(O)CC(CC)CCCC"
CYANEX272 = "CC(CC(C)(C)C)CP(=O)(O)CC(C)CC(C)(C)C"
CYANEX301 = "CC(CC(C)(C)C)CP(=S)(S)CC(C)CC(C)(C)C"
TBP = "CCCCOP(=O)(OCCCC)OCCCC"
TOPO = "CCCCCCCCP(=O)(CCCCCCCC)CCCCCCCC"
CMPO = "CCCCCCCCP(=O)(CC(=O)N(CC(C)C)CC(C)C)c1ccccc1"
C5BTBP = "CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC"
HTTA = "O=C(CC(=O)C(F)(F)F)c1cccs1"
ALIQUAT_CATION = "C[N+](CCCCCCCC)(CCCCCCCC)CCCCCCCC"
DHHP = "CCCCCCOP(=O)(O)OCCCCCC"
DEHPA_AMIDE = "CCCCC(CC)CN(CC(CC)CCCC)C(=O)CC"
TEDGA = "CCN(CC)C(=O)COCC(=O)N(CC)CC"
SO3_BTP = ("O=S(=O)([O-])c1ccc(-c2nnc(-c3cccc(-c4nnc(-c5ccc(S(=O)(=O)[O-])cc5)c(-c5ccc(S(=O)(=O)[O-])cc5)n4)n3)"
           "nc2-c2ccc(S(=O)(=O)[O-])cc2)cc1")
DTPA = "O=C(O)CN(CCN(CC(=O)O)CC(=O)O)CCN(CC(=O)O)CC(=O)O"
DTPA_NITRATE = DTPA + ".O=[N+]([O-])[O-]"
NDPP = "c1cc(-n2cccn2)nc(-n2cccn2)c1"
BROMODECANOIC = "CCCCCCCCC(Br)C(=O)O"
TODGA_SYN = "CCCCCCCCN(CCCCCCCC)C(=O)[C@H](CCC)O[C@H](CCC)C(=O)N(CCCCCCCC)CCCCCCCC"
TODGA_ANTI = "CCCCCCCCN(CCCCCCCC)C(=O)[C@H](CCC)O[C@@H](CCC)C(=O)N(CCCCCCCC)CCCCCCCC"
CDTA_STEREO = "O=C(O)CN(CC(=O)O)[C@@H]1CCCC[C@H]1N(CC(=O)O)CC(=O)O"
CDTA_FLAT = "O=C(O)CN(CC(=O)O)C1CCCCC1N(CC(=O)O)CC(=O)O"
COSAN_WRONG = "Cc1cc(Br)ccc1NC(=O)CCl"
T8_PHEN_CAM = "CCCCCCCCN(CCCCCCCC)C(=O)c1ccc2ccc3ccc(C(=O)N(CCCCCCCC)CCCCCCCC)nc3c2n1"


@pytest.fixture(scope="module")
def clf() -> L.FamilyClassifier:
    return L.FamilyClassifier(L.FAMILY_RULES_SPEC)


# ------------------------------------------------------------------------------------------------
# families, acidity, mechanisms of known extractants
# ------------------------------------------------------------------------------------------------
KNOWN = [
    ("TODGA", TODGA, "diglycolamide", "NEUTRAL_SOLVATING", "NEUTRAL"),
    ("D2EHPA", D2EHPA, "phosphoric_acid", "ACIDIC_CATION_EXCHANGE", "ACIDIC"),
    ("PC88A", PC88A, "phosphonic_acid_monoester", "ACIDIC_CATION_EXCHANGE", "ACIDIC"),
    ("Cyanex 272", CYANEX272, "phosphinic_acid", "ACIDIC_CATION_EXCHANGE", "ACIDIC"),
    ("Cyanex 301", CYANEX301, "dithiophosphinic_acid", "ACIDIC_CATION_EXCHANGE", "ACIDIC"),
    ("TBP", TBP, "neutral_organophosphate", "NEUTRAL_SOLVATING", "NEUTRAL"),
    ("TOPO", TOPO, "phosphine_oxide", "NEUTRAL_SOLVATING", "NEUTRAL"),
    ("CMPO", CMPO, "carbamoylmethylphosphine_oxide", "NEUTRAL_SOLVATING", "NEUTRAL"),
    ("C5-BTBP", C5BTBP, "n_heterocyclic", "SOFT_N_DONOR", "NEUTRAL"),
    ("HTTA", HTTA, "beta_diketone", "CHELATING", "ACIDIC"),
    ("Aliquat 336 cation", ALIQUAT_CATION, "quaternary_ammonium", "ION_PAIR_BASIC", "IONIC"),
    ("dihexyl hydrogen phosphate", DHHP, "phosphoric_acid", "ACIDIC_CATION_EXCHANGE", "ACIDIC"),
    ("DEHPA (archive amide)", DEHPA_AMIDE, "monoamide", "NEUTRAL_SOLVATING", "NEUTRAL"),
    ("LIX 84", "CCCCCCCCCc1ccc(O)c(C(C)=NO)c1", "hydroxyoxime", "CHELATING", "ACIDIC"),
    ("Versatic 10", "CCCCCCC(C)(C)C(=O)O", "carboxylic_acid", "ACIDIC_CATION_EXCHANGE", "ACIDIC"),
    ("trioctylamine", "CCCCCCCCN(CCCCCCCC)CCCCCCCC", "amine_basic", "ION_PAIR_BASIC", "BASIC"),
    ("18-crown-6", "C1COCCOCCOCCOCCOCCO1", "crown_ether_calixarene", "NEUTRAL_SOLVATING", "NEUTRAL"),
    ("C4mim Tf2N", "CCCCn1cc[n+](C)c1.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "ionic_liquid",
     "ION_PAIR_BASIC", "IONIC"),
    ("DMDOHEMA", "CCCCCCCCN(C)C(=O)C(CCOCCCCCC)C(=O)N(C)CCCCCCCC", "malonamide", "NEUTRAL_SOLVATING", "NEUTRAL"),
    # CHEM-10: acidic / chelating classes that previously fell into neutral families
    ("HPMBP (keto)", "CC1=NN(c2ccccc2)C(=O)C1C(=O)c1ccccc1", "acylpyrazolone", "CHELATING", "ACIDIC"),
    ("HPMBP (enol)", "Cc1nn(-c2ccccc2)c(O)c1C(=O)c1ccccc1", "acylpyrazolone", "CHELATING", "ACIDIC"),
    ("8-hydroxyquinoline", "Oc1cccc2cccnc12", "hydroxyquinoline", "CHELATING", "ACIDIC"),
    ("thiodiglycolamide", "CCCCCCCCN(CCCCCCCC)C(=O)CSCC(=O)N(CCCCCCCC)CCCCCCCC", "diglycolamide",
     "NEUTRAL_SOLVATING", "NEUTRAL"),
    # CHEM-09: quercetin is an O,O chelator
    ("quercetin", "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "hydroxyketone_catechol", "CHELATING", "ACIDIC"),
    # CHEM-05: the COOH of EsPyTri is remote from the N3 pocket -> soft-N donor, not a cation exchanger
    ("EsPyTri", "CCCCCCCCn1cc(-c2cc(CC(=O)O)cc(-c3cn(CCCCCCCC)nn3)n2)nn1", "n_heterocyclic", "SOFT_N_DONOR", "ACIDIC"),
]


@pytest.mark.parametrize("label,smiles,family,mechanism,acidity", KNOWN, ids=[k[0] for k in KNOWN])
def test_known_family_mechanism_acidity(clf, label, smiles, family, mechanism, acidity):
    fa = clf.classify(smiles)
    assert fa.family == family, (label, fa)
    assert fa.status == "UNIQUE", (label, fa.candidates)
    assert M.component_mechanism(fa.family, fa.core_family)["mechanism"] == mechanism
    assert L.acidity_class(L.mol_from_smiles(smiles)) == acidity


def test_aliquat_with_chloride_is_not_a_salt(clf):
    fa = clf.classify(ALIQUAT_CATION + ".[Cl-]")
    assert fa.family == "quaternary_ammonium" and fa.overlay is None


@pytest.mark.parametrize("smiles,core,mech", [
    (TEDGA, "diglycolamide", "NEUTRAL_SOLVATING"),
    (SO3_BTP, "n_heterocyclic", "SOFT_N_DONOR"),
    (DTPA, "aminopolycarboxylic_acid", "CHELATING"),
])
def test_hydrophilic_agents_keep_core_family(clf, smiles, core, mech):
    fa = clf.classify(smiles)
    assert fa.family == "hydrophilic_aqueous_agent"
    assert fa.core_family == core
    cm = M.component_mechanism(fa.family, fa.core_family)
    assert cm["is_hydrophilic_agent"] and cm["mechanism"] == mech


def test_salts(clf):
    assert clf.classify("O=[N+]([O-])[O-]").family == "electrolyte_salt"
    assert clf.classify("O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F").family == "electrolyte_salt"


def test_reference_extractants_match_expected_family(clf):
    for name, ref in L.REFERENCE_EXTRACTANTS.items():
        assert clf.classify(ref["smiles"]).family == ref["expected_family"], name


def test_ambiguous_flag_only_for_non_nested_rules(clf):
    phen_bis_po = "O=P(c1ccccc1)(c1ccccc1)c1ccc2ccc3ccc(P(=O)(c4ccccc4)c4ccccc4)nc3c2n1"
    fa = clf.classify(phen_bis_po)
    assert fa.ambiguous and set(fa.candidates) == {"phosphine_oxide", "n_heterocyclic"}
    cmpo = clf.classify(CMPO)          # fires CMPO + phosphine_oxide + monoamide, all nested
    assert {"phosphine_oxide", "monoamide"} <= set(cmpo.fired) and not cmpo.ambiguous
    dga = clf.classify(TODGA)          # fires DGA + monoamide/polyamide, nested
    assert not dga.ambiguous


def test_every_family_has_a_mechanism(clf):
    for fam in clf.labels:
        if fam == "hydrophilic_aqueous_agent":      # overlay: the core family decides
            continue
        assert fam in M.FAMILY_MECHANISM, fam
    assert "metallacarborane_anion" in M.FAMILY_MECHANISM


# ------------------------------------------------------------------------------------------------
# donor census and denticity proxy
# ------------------------------------------------------------------------------------------------
def test_donor_census_and_denticity():
    m = L.mol_from_smiles(TODGA)
    c = L.donor_census(m)
    assert c["amide_carbonyl_O"] == 2 and c["ether_O"] == 1
    assert L.denticity(m) == (3, 3, 1)
    assert L.denticity(L.mol_from_smiles(D2EHPA)) == (1, 1, 0)   # P=O and P-OH collapse to one site
    assert L.denticity(L.mol_from_smiles(CMPO))[1] == 2
    assert L.denticity(L.mol_from_smiles(HTTA))[1] == 2
    # regression: carboxylic acid O atoms are donor sites (collapsed to one)
    assert L.denticity(L.mol_from_smiles("CCCCCCC(C)(C)C(=O)O")) == (1, 1, 0)
    assert L.donor_census(L.mol_from_smiles("CCCCCCCCCc1ccc(O)c(C(C)=NO)c1"))["oxime_N"] == 1
    btbp = L.donor_census(L.mol_from_smiles(C5BTBP))
    assert btbp["aromatic_N_pyridine_type"] == 2 and btbp["triazine_N"] == 6
    assert L.denticity(L.mol_from_smiles(C5BTBP))[1] == 4   # N4: 2 pyridine + 2 triazine rings


def test_transannular_pair_not_credited():
    # 4-pyranone-2,6-dicarboxamide: ring O chelates with both amide O; the 4-keto O points away
    pyranone = "CCCCCCCCCCCCNC(=O)c1cc(=O)cc(C(=O)NCCCCCCCCCCCC)o1"
    n_sites, dent, units = L.denticity(L.mol_from_smiles(pyranone))
    assert (n_sites, dent, units) == (4, 3, 1)


# ------------------------------------------------------------------------------------------------
# rules file and determinism
# ------------------------------------------------------------------------------------------------
def test_rules_file_agrees_with_spec():
    if not L.FAMILY_RULES_PATH.exists():
        pytest.skip("descriptors/family_rules.json not built yet")
    with open(L.FAMILY_RULES_PATH, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert L.rules_digest(on_disk) == L.rules_digest(L.FAMILY_RULES_SPEC)


def test_classifier_reads_the_rules_file(tmp_path):
    rules = copy.deepcopy(L.FAMILY_RULES_SPEC)
    rules["families"] = [f for f in rules["families"] if f["name"] != "phosphinic_acid"]
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(rules), encoding="utf-8")
    assert L.FamilyClassifier.from_json(p).classify(CYANEX272).family != "phosphinic_acid"
    q = tmp_path / "rules_full.json"
    q.write_text(json.dumps(L.FAMILY_RULES_SPEC), encoding="utf-8")
    assert L.FamilyClassifier.from_json(q).classify(CYANEX272).family == "phosphinic_acid"


def test_classifier_determinism(clf):
    fresh = L.FamilyClassifier(copy.deepcopy(L.FAMILY_RULES_SPEC))
    for _, smi, *_ in KNOWN:
        assert clf.classify(smi) == fresh.classify(smi)
        assert L.describe_structure(smi, clf) == L.describe_structure(smi, fresh)
    # a non-canonical writing of D2EHPA gets the same label
    assert fresh.classify("OP(=O)(OCC(CC)CCCC)OCC(CC)CCCC").family == "phosphoric_acid"


# ------------------------------------------------------------------------------------------------
# name / alias normalisation
# ------------------------------------------------------------------------------------------------
def _c(role, name, smiles):
    return {"role": role, "name": name, "smiles_raw": smiles, "smiles_canonical": smiles,
            "structure_source": "archive", "concentration_M": 0.1}


def _hold():
    return {"role": "aqueous_holdback", "name": None, "smiles_raw": None, "smiles_canonical": None,
            "structure_source": None, "concentration_M": 0.0}


@pytest.fixture()
def index() -> L.AliasIndex:
    idx = L.AliasIndex(L.FAMILY_RULES_SPEC)
    rid = 0

    def add(n, comps, model=True):
        nonlocal rid
        for _ in range(n):
            idx.add(rid, model, comps)
            rid += 1

    add(10, [_c("organic_extractant", "TEHDGA", TEHDGA)])
    add(3, [_c("organic_extractant", "T2EHDGA", TEHDGA)])
    add(1, [_c("organic_extractant", "TEHDGA", TODGA)])                  # one mis-named TODGA row
    add(20, [_c("organic_extractant", "TODGA", TODGA)])
    add(2, [_c("organic_extractant", "TWE-40", TODGA), _hold()])          # masking-agent names
    add(2, [_c("organic_extractant", "TWE-41", TODGA), _hold()])
    add(2, [_c("organic_extractant", "SO3-Ph-BTP", TODGA), _hold()])
    add(3, [_c("aqueous_complexant", "SO3-Ph-BTP", SO3_BTP), _c("organic_extractant", "TODGA", TODGA)])
    add(4, [_c("organic_extractant", "N-DPP", NDPP), _hold()])            # consistent name in masking rows
    add(4, [_c("organic_extractant", "2-bromodecanoic acid", BROMODECANOIC), _c("organic_extractant", "N-DPP", NDPP), _hold()])
    add(2, [_c("organic_extractant", "2-bromodecanoic acid", NDPP), _c("organic_extractant", "N-DP(DOM)P",
            "CCCCCCCCCCCCOCc1cc(-n2cccn2)nc(-n2cccn2)c1"), _hold()])
    add(5, [_c("aqueous_complexant", "DTPA", DTPA_NITRATE), _c("organic_extractant", "TODGA", TODGA)])
    add(1, [_c("aqueous_complexant", "DTPA", DTPA), _c("organic_extractant", "TODGA", TODGA)])
    add(3, [_c("organic_extractant", "p-TODGA-Syn", TODGA_SYN)])
    add(1, [_c("organic_extractant", "p-TODGA-Anti", TODGA_ANTI)])
    add(2, [_c("organic_extractant", "Br-Cosan", COSAN_WRONG), _c("organic_extractant", "T8-PHEN-CAM", T8_PHEN_CAM), _hold()])
    return idx.finalize()


def test_normalize_name_key():
    k = L.normalize_name_key
    assert k("Cy5-S-Me4-BTBP") == k("cy5s me4_btbp") == k("CY5S-ME4-BTBP")
    assert k("1-octanol") != k("octanol")          # locants kept: the archive keeps these apart
    assert k(None) is None and k("  ") is None


def test_alias_resolves_only_when_structure_agrees(index):
    assert index.canonical_name(TEHDGA) == "TEHDGA"
    assert index.resolve("tehdga", TEHDGA) == "TEHDGA"
    assert index.resolve("T-2EHDGA", TEHDGA) == "TEHDGA"      # spelling variant of an accepted alias
    assert index.resolve("TEHDGA", TODGA) is None             # same spelling, different structure
    assert index.resolve("todga", TEHDGA) is None
    assert index.rejection_reason("TEHDGA", TODGA) == "NAME_OWNED_BY_OTHER_STRUCTURE"
    # a non-canonical writing of the structure is canonicalised before lookup
    assert index.resolve("TEHDGA", "O=C(COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC)N(CC(CC)CCCC)CC(CC)CCCC") == "TEHDGA"


def test_masking_agent_names_never_become_canonical(index):
    assert index.canonical_name(TODGA) == "TODGA"
    assert index.is_masking_collision("TWE-40", TODGA)
    assert index.rejection_reason("TWE-41", TODGA) == "MASKING_AGENT_NAME"
    assert index.rejection_reason("SO3-Ph-BTP", TODGA).startswith("NAME_STRUCTURE_CHECK_FAILED")
    assert index.owner("SO3-Ph-BTP") == SO3_BTP
    # one consistent name confined to masking rows is not a masking collision
    assert not index.is_masking_collision("N-DPP", NDPP)
    assert index.canonical_name(NDPP) == "N-DPP"


def test_salt_and_stereo_forms(index):
    assert L.AliasIndex.same_ligand(DTPA, DTPA_NITRATE)
    assert L.AliasIndex.same_ligand(CDTA_STEREO, CDTA_FLAT)
    assert not L.AliasIndex.same_ligand(TODGA_SYN, TODGA_ANTI)
    assert index.accepted("DTPA", DTPA) and index.accepted("DTPA", DTPA_NITRATE)


def test_resolve_occurrence_corrections(index, clf):
    # name fails a check and its owner is absent from the row -> name-implied structure
    r = L.resolve_occurrence(index, "2-bromodecanoic acid", NDPP, clf, row_structures=["CCCCCCCCCCCCOCc1cc(-n2cccn2)nc(-n2cccn2)c1"])
    assert r["basis"] == "NAME_IMPLIED_STRUCTURE" and r["family"] == "carboxylic_acid"
    # owner already in the row -> positional swap, structure stands
    r = L.resolve_occurrence(index, "2-bromodecanoic acid", NDPP, clf, row_structures=[BROMODECANOIC])
    assert r["basis"] == "STRUCTURE_POSITIONAL_SWAP" and r["family"] == "n_heterocyclic"
    # name denotes an aqueous agent -> recorded organic structure stands
    r = L.resolve_occurrence(index, "SO3-Ph-BTP", TODGA, clf)
    assert r["basis"] == "STRUCTURE_NAME_IS_AQUEOUS_AGENT" and r["family"] == "diglycolamide"
    # known chemistry override
    r = L.resolve_occurrence(index, "Br-Cosan", COSAN_WRONG, clf)
    assert r["basis"] == "KNOWN_NAME_OVERRIDE" and r["family"] == "metallacarborane_anion"
    assert L.resolve_occurrence(index, "Br-Cosan", COSAN_WRONG, clf, name_corrected=False)["family"] == "monoamide"
    # name-only lookup
    r = L.resolve_occurrence(index, "HDEHP", None, clf)
    assert r["family"] == "phosphoric_acid" and r["smiles"] == L.canonical_smiles(D2EHPA)
    assert L.resolve_occurrence(index, None, None, clf) is None


# ------------------------------------------------------------------------------------------------
# system mechanisms
# ------------------------------------------------------------------------------------------------
def _entry(clf, smiles, role="organic_extractant"):
    fa = clf.classify(smiles)
    cm = M.component_mechanism(fa.family, fa.core_family)
    return M.Entry(key=smiles, mechanism=cm["mechanism"], family=fa.core_family,
                   is_hydrophilic_agent=cm["is_hydrophilic_agent"], role=role)


def test_system_mechanisms(clf):
    e = lambda s, role="organic_extractant": _entry(clf, s, role)  # noqa: E731
    assert M.system_mechanism([]).mechanism == "UNKNOWN"
    assert M.system_mechanism([e(TODGA)]).mechanism == "NEUTRAL_SOLVATING"
    assert M.system_mechanism([e(BROMODECANOIC), e(C5BTBP)]).mechanism == "SYNERGISTIC"
    assert M.system_mechanism([e(HTTA), e(TOPO)]).mechanism == "SYNERGISTIC"
    assert M.system_mechanism([e(TODGA), e(TBP)]).mechanism == "MIXED_NEUTRAL"
    assert M.system_mechanism([e(C5BTBP), e(TODGA)]).mechanism == "MIXED_NEUTRAL"
    assert M.system_mechanism([e(D2EHPA), e(CYANEX272)]).mechanism == "MIXED_ACIDIC"
    assert M.system_mechanism([e(ALIQUAT_CATION), e(D2EHPA)]).mechanism == "UNKNOWN"
    sm = M.system_mechanism([e(TODGA), e(TEDGA)])
    assert sm.mechanism == "NEUTRAL_SOLVATING" and "HYDROPHILIC_AGENT_IN_EXTRACTANT_SLOT" in sm.flags
    assert sm.system_family == "diglycolamide"
    sm = M.system_mechanism([e(TEDGA)])
    assert sm.mechanism == "NEUTRAL_SOLVATING" and "HYDROPHILIC_ONLY_EXTRACTANT" in sm.flags
    sm = M.system_mechanism([e(TODGA)], [e(D2EHPA, "phase_modifier")])
    assert sm.mechanism == "SYNERGISTIC" and "ACIDIC_MODIFIER_AS_COEXTRACTANT" in sm.flags
    assert M.system_mechanism([e(TODGA)], [e(TBP, "phase_modifier")]).mechanism == "NEUTRAL_SOLVATING"
    # the same structure twice is one extractant
    assert M.system_mechanism([e(TODGA), e(TODGA)]).mechanism == "NEUTRAL_SOLVATING"


def test_remote_acid_does_not_override_n_pocket_but_chelating_acid_does(clf):
    """CHEM-05: the conditional subsumption needs a >= 2-site aromatic-N pocket AND an acid that cannot chelate
    with a ring N; a picolinic-type acid (COOH next to the ring N) keeps the carboxylic-acid label."""
    esp = clf.classify("CCCCCCCCn1cc(-c2cc(CC(=O)O)cc(-c3cn(CCCCCCCC)nn3)n2)nn1")
    assert esp.core_family == "n_heterocyclic" and not esp.ambiguous
    assert L.aromatic_n_chelation(L.mol_from_smiles("CCCCCCCCn1cc(-c2cc(CC(=O)O)cc(-c3cn(CCCCCCCC)nn3)n2)nn1")) == (3, 0)
    pic = clf.classify("CCCCCCCCc1ccnc(C(=O)O)c1")           # lipophilic picolinic acid: one ring, chelating acid
    assert pic.core_family == "carboxylic_acid"
    assert L.aromatic_n_chelation(L.mol_from_smiles("CCCCCCCCc1ccnc(C(=O)O)c1"))[1] == 2


def test_name_structure_checks_for_dga_and_dipic():
    """CHEM-02 / CHEM-09: 'DGA' needs a diglycolAMIDE (thiodiglycolic acid under 'TDGA' fails); 'DIPIC' needs a
    pyridine-2,6-dicarboxamide."""
    fails = lambda name, smi: L.name_structure_check_failures(name, L.mol_from_smiles(smi), L.FAMILY_RULES_SPEC)  # noqa: E731
    assert "DGA_NAME_REQUIRES_DIGLYCOLYL" in fails("TDGA", "O=C(O)CSCC(=O)O")
    assert "DGA_NAME_REQUIRES_DIGLYCOLYL" in fails("TPDGA", "CCCN(CCC)C(=O)CC(=O)N(CCC)CCC")
    assert fails("TODGA", TODGA) == []
    assert fails("TOTDGA", "CCCCCCCCN(CCCCCCCC)C(=O)CSCC(=O)N(CCCCCCCC)CCCCCCCC") == []
    tba = "CC(C)(C)c1ccc(NCC(=O)c2cccc(C(=O)CNc3ccc(C(C)(C)C)cc3)n2)cc1"
    assert "DIPIC_NAME_REQUIRES_PYRIDINE_DICARBOXAMIDE" in fails("TBADIPIC", tba)
    assert fails("TEDIPIC", "CCN(CC)C(=O)c1cccc(C(=O)N(CC)CC)n1") == []
    assert fails("dipicolinic acid", "O=C(O)c1cccc(C(=O)O)n1") == []


def test_aqueous_agent_recorded_as_sole_extractant_is_unknown(clf):
    """CHEM-01 / CHEM-02: an ionisable hydrophilic acid or chelator as the only recorded extractant is an aqueous
    agent in the extractant slot (the real solvent is not recorded); a neutral hydrophilic DGA stays neutral."""
    e = lambda s: _entry(clf, s)  # noqa: E731
    hedta = "O=C(O)CN(CCO)CCN(CC(=O)O)CC(=O)O"
    for smi in (hedta, CDTA_STEREO, "O=C(O)CSCC(=O)O"):
        sm = M.system_mechanism([e(smi)])
        assert sm.mechanism == "UNKNOWN" and "AQUEOUS_AGENT_RECORDED_AS_EXTRACTANT" in sm.flags, smi
    assert M.system_mechanism([e(TEDGA)]).mechanism == "NEUTRAL_SOLVATING"


@pytest.mark.slow
def test_built_components_carry_corrections():
    """CHEM-07 / CHEM-09 / CHEM-04 on the built tables."""
    cp = paths.DESCRIPTORS_DIR / "extractant_components.csv"
    sp_ = paths.DESCRIPTORS_DIR / "extractant_systems.csv"
    if not (cp.exists() and sp_.exists()):
        pytest.skip("outputs not built")
    comp = pd.read_csv(cp)
    cos = comp[comp["smiles_canonical"] == COSAN_WRONG].iloc[0]
    assert (cos["family"], cos["mechanism"], cos["acidity_class"]) == ("metallacarborane_anion", "ACIDIC_CATION_EXCHANGE",
                                                                       "ACIDIC")
    assert cos["family_structural"] == "monoamide" and cos["family_status"] == "NAME_STRUCTURE_CONFLICT"
    systems = pd.read_csv(sp_)
    names = systems.set_index("component_canonical_names")
    assert names.loc["T8-PHEN-CAM|Br-Cosan", "system_family"] == "metallacarborane_anion+pyridine_carboxamide"
    assert names.loc["N-DP(DOM)P|N-DPP", "system_family"] == "carboxylic_acid+n_heterocyclic"
    # the name corrections reach the per-component labels too; the structure-only labels are kept beside them
    corrected = systems[systems["system_family"] != systems["system_family_structural"]]
    assert len(corrected) >= 5          # the Br-Cosan and N-DPP mixtures and the malonamide recorded as TPDGA
    for _, row in corrected.iterrows():
        assert "+".join(sorted(set(str(row["component_core_families"]).split("|")))) == row["system_family"], row["system_id"]
        assert "+".join(sorted(set(str(row["component_core_families_structural"]).split("|")))) \
            == row["system_family_structural"], row["system_id"]
    assert set(str(names.loc["T8-PHEN-TAM|Br-Cosan", "component_families"]).split("|")) == {
        "metallacarborane_anion", "pyridine_carboxamide"}
    tpdga = systems[(systems["component_canonical_names"] == "TPDGA") & systems["has_name_structure_conflict"].astype(bool)]
    assert tpdga["component_families_structural"].tolist() == ["malonamide"]
    assert tpdga["component_families"].tolist() == ["diglycolamide"]
    cov = pd.read_csv(paths.DATA_AUDIT_DIR / "family_coverage.csv")
    levels = set(cov["level"])
    assert {"system_family", "component_family", "mechanism", "system_family_structural",
            "component_family_structural"} <= levels
    lab = cov.set_index(["level", "label"])
    assert ("system_family", "metallacarborane_anion+pyridine_carboxamide") in lab.index
    assert ("system_family_structural", "metallacarborane_anion+pyridine_carboxamide") not in lab.index
    assert ("component_family", "metallacarborane_anion") in lab.index
    for nm in ("HEDTA", "TDGA"):
        assert names.loc[nm, "mechanism"] == "UNKNOWN"
    assert names.loc["TDGA", "has_name_structure_conflict"]
    assert names.loc["TODGA", "n_model_rows_acidic_coextractant_modifier"] >= 1
    assert names.loc["TODGA", "mechanism"] == "NEUTRAL_SOLVATING"
    assert names.loc["TODGA", "mechanism_with_acidic_coextractant"] == "SYNERGISTIC"
    assert "_TIE" not in str(names.loc["DOHyA", "mechanism_row_votes"])


def test_vote():
    assert M.vote({"A": 3, "B": 1}) == ("A", "A:3;B:1")
    assert M.vote({"B": 2, "A": 2})[1].endswith("_TIE")


# ------------------------------------------------------------------------------------------------
# full archive
# ------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_every_archive_component_smiles_gets_exactly_one_family(clf):
    from gen19ct.data.load import load_archive
    df = load_archive(copy=False)
    smiles = {str(c["smiles_canonical"]) for comps in df["components"] for c in comps
              if c.get("smiles_canonical") is not None and str(c["smiles_canonical"]).strip()}
    assert len(smiles) > 0
    allowed = set(clf.labels) | {"hydrophilic_aqueous_agent", "electrolyte_salt"}
    for s in smiles:
        fa = clf.classify(s)
        assert isinstance(fa.family, str) and fa.family in allowed, s
        assert L.mol_from_smiles(s) is not None, s
    if paths.DESCRIPTORS_DIR.joinpath("extractant_components.csv").exists():
        comp = pd.read_csv(paths.DESCRIPTORS_DIR / "extractant_components.csv")
        st = comp[comp["record_type"] == "STRUCTURE"]
        assert set(st["smiles_canonical"]) == smiles
        assert st["smiles_canonical"].is_unique
        assert st["family"].notna().all()
        assert all(st["family_structural"] == [clf.classify(s).family for s in st["smiles_canonical"]])
        # the component family differs from the structural one only where a known name override applies
        differs = st[st["family"] != st["family_structural"]]
        assert differs["flags"].fillna("").str.contains("KNOWN_NAME_OVERRIDE_APPLIED_TO_COMPONENT").all()


@pytest.mark.slow
def test_built_tables_balance():
    sysp = paths.DESCRIPTORS_DIR / "extractant_systems.csv"
    covp = paths.DATA_AUDIT_DIR / "family_coverage.csv"
    presp = paths.DATA_AUDIT_DIR / "named_extractant_presence.csv"
    if not (sysp.exists() and covp.exists() and presp.exists()):
        pytest.skip("outputs not built")
    from gen19ct.data.load import load_archive
    df = load_archive(copy=False)
    s = pd.read_csv(sysp, keep_default_na=False)
    assert s["system_id"].is_unique and s["extractant_system_key"].is_unique
    assert s["n_rows"].sum() == len(df)
    for t in ("MODEL", "TARGET_ONLY", "NO_TARGET"):
        assert s[f"n_rows_{t}"].sum() == int((df["g19_tier"] == t).sum())
    assert s["extractant_system_key"].nunique() == df["extractant_system_key"].nunique(dropna=False)
    cov = pd.read_csv(covp)
    for level in ("system_family", "mechanism"):
        assert cov.loc[cov["level"] == level, "n_rows_all"].sum() == len(df)
    pres = pd.read_csv(presp).set_index("term")
    assert pres.loc["DEHPA", "verdict"] == "NAME_TRAP_DIFFERENT_STRUCTURE_FAMILY"
    for term in ("PC88A", "P507", "Cyanex 272", "D2EHPA", "P204"):
        assert pres.loc[term, "verdict"] == "ABSENT"
    # CHEM-04: HDEHP sits in the modifier slot only; titles name bis(2-ethylhexyl)phosphoric acid on rows whose
    # recorded structures are other extractants or aqueous agents
    assert pres.loc["HDEHP", "roles_on_matching_components"].startswith("phase_modifier")
    assert pres.loc["D2EHPA:FULL_NAME_OR_TITLE_TEXT", "n_model_rows"] > 0
    for term in ("PC88A:FULL_NAME_OR_TITLE_TEXT", "Cyanex 272:FULL_NAME_OR_TITLE_TEXT"):
        assert pres.loc[term, "verdict"] == "ABSENT"
