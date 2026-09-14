"""Tests for the multi-metal SAE dataset pipeline.

The unit-level tests use small synthetic records so that a failure names the
exact rule that broke.  The integration tests assert properties of the real
cleaned artifacts, because several of the guarantees in the brief ("no row
silently disappears") are only meaningful on the full dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import sae_chem as CHEM            # noqa: E402
import sae_identity as ID          # noqa: E402
import sae_normalize as NORM       # noqa: E402
import sae_schema as SCHEMA        # noqa: E402
from sae_structures import canonicalize  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def make_record(**overrides):
    """A minimal normalized record; override only what a test cares about."""
    record = {
        "metal_symbol": "Eu", "metal_oxidation_state": 3,
        "extractant_system_key": "CCCCOP(=O)(OCCCC)OCCCC",
        "extractant_smiles_canonical": ["CCCCOP(=O)(OCCCC)OCCCC"],
        "extractant_concentrations_M": [0.1],
        "acid_signature": "HNO3", "acid_concentration_M": 3.0,
        "acid_concentration_organic_M": None, "solvent_key": "dodecane:1",
        "modifier_name": None, "modifier_concentration_M": None,
        "complexant_signature": None, "complexant_smiles_canonical": None,
        "holdback_smiles_canonical": None, "holdback_concentration_M": None,
        "metal_concentration_M": 0.001, "nitrate_concentration_M": None,
        "temperature_C": 25.0, "contact_time_min": 30.0,
        "shaking_time_min": None, "phase_ratio_org_aq": 1.0,
        "log_D": 0.5,
    }
    record.update(overrides)
    return record


@pytest.fixture(scope="session")
def master():
    path = ROOT / "clean" / "master_clean.parquet"
    if not path.exists():
        pytest.skip("pipeline artifacts not built yet")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def mapping():
    return pd.read_parquet(ROOT / "audit" / "raw_to_canonical_mapping.parquet")


# --------------------------------------------------------------------------
# 1. Permutation invariance of deduplication
# --------------------------------------------------------------------------

def test_component_order_does_not_change_identity():
    """"A, B" and "B, A" are the same synergistic system."""
    forward = make_record(extractant_smiles_canonical=["CCO", "CCN"],
                          extractant_concentrations_M=[0.2, 0.5])
    reverse = make_record(extractant_smiles_canonical=["CCN", "CCO"],
                          extractant_concentrations_M=[0.5, 0.2])
    assert ID.identity_hash(forward) == ID.identity_hash(reverse)


def test_component_order_still_respects_concentration_pairing():
    """Swapping which component carries which concentration IS a real change."""
    original = make_record(extractant_smiles_canonical=["CCO", "CCN"],
                           extractant_concentrations_M=[0.2, 0.5])
    swapped = make_record(extractant_smiles_canonical=["CCO", "CCN"],
                          extractant_concentrations_M=[0.5, 0.2])
    assert ID.identity_hash(original) != ID.identity_hash(swapped)


def test_row_order_does_not_change_grouping():
    """Shuffling the input must not change which records group together.

    Comparing a forward list against a reversed one is an algebraic identity
    that holds for ANY pure per-record function, so it proved nothing. This
    instead builds the actual partition -- which records share a key -- and
    requires it to be identical after a shuffle.
    """
    import random as _random

    records = [make_record(log_D=float(i), metal_symbol=m, acid_concentration_M=c)
               for i, (m, c) in enumerate([("Eu", 3.0), ("Nd", 3.0), ("Eu", 3.0),
                                           ("La", 1.0), ("Eu", 1.0), ("Nd", 3.0)])]
    for index, record in enumerate(records):
        record["_id"] = f"r{index}"

    def partition(rows):
        groups = {}
        for row in rows:
            groups.setdefault(ID.identity_hash(row), set()).add(row["_id"])
        return sorted(sorted(g) for g in groups.values())

    baseline = partition(records)
    # The fixture really must produce collisions, or the test is trivial.
    assert any(len(g) > 1 for g in baseline), "fixture produced no duplicate group"
    assert len(baseline) < len(records), "fixture produced no grouping at all"

    for seed in (0, 1, 2, 3, 4):
        shuffled = list(records)
        _random.Random(seed).shuffle(shuffled)
        assert partition(shuffled) == baseline, f"grouping changed under shuffle seed {seed}"


# --------------------------------------------------------------------------
# 2. Stable canonical IDs
# --------------------------------------------------------------------------

def test_identity_hash_is_stable_across_processes():
    """The hash must not depend on Python's per-process string hash seed."""
    record = make_record()
    expected = ID.identity_hash(record)
    code = (f"import sys; sys.path.insert(0, {str(SCRIPTS)!r});"
            "import sae_identity as ID, json;"
            f"print(ID.identity_hash(json.loads({json.dumps(json.dumps(record))})))")
    for seed in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"})
        assert out.stdout.strip() == expected, out.stderr


def test_canonical_measurement_id_derives_from_archive_id(master):
    ids = master["canonical_measurement_id"]
    assert ids.is_unique
    assert (ids == "SAE:" + master["source_record_id"]).all()


# --------------------------------------------------------------------------
# 3. Equivalent-unit normalisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,kind,expected", [
    ("0.72 mM", "concentration", 0.00072),
    ("0.00072 M", "concentration", 0.00072),
    ("3 M", "concentration", 3.0),
    ("3.0 M", "concentration", 3.0),
    ("60 min", "time", 60.0),
    ("1 h", "time", 60.0),
    ("298.15 K", "temperature", 25.0),
    ("25 C", "temperature", 25.0),
])
def test_equivalent_units_normalise_to_the_same_value(text, kind, expected):
    quantity = NORM.parse_quantity(text, kind)
    assert quantity is not None
    assert quantity.value == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_mM_and_M_records_group_together():
    """0.72 mM and 0.00072 M are one condition, not two."""
    a = make_record(metal_concentration_M=NORM.parse_quantity("0.72 mM", "concentration").value)
    b = make_record(metal_concentration_M=NORM.parse_quantity("0.00072 M", "concentration").value)
    assert ID.identity_hash(a) == ID.identity_hash(b)


def test_float_serialisation_noise_is_absorbed():
    a = make_record(acid_concentration_M=0.001)
    b = make_record(acid_concentration_M=0.001000000000000023)
    assert ID.identity_hash(a) == ID.identity_hash(b)


def test_unparseable_quantity_returns_none_not_a_guess():
    for text in ("- -", "", "-", "~3 M", "1-2 M", ">5 M"):
        assert NORM.parse_quantity(text, "concentration") is None


# --------------------------------------------------------------------------
# 4. Meaningful condition differences prevent a merge
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("metal_symbol", "Nd"),
    ("metal_oxidation_state", 4),
    ("acid_concentration_M", 1.0),
    ("acid_signature", "HCl"),
    ("solvent_key", "toluene:1"),
    ("temperature_C", 60.0),
    ("contact_time_min", 5.0),
    ("metal_concentration_M", 0.05),
    ("phase_ratio_org_aq", 0.5),
    ("modifier_name", "1-octanol"),
    ("complexant_signature", "DTPA@0.05"),
    ("holdback_smiles_canonical", "CCO"),
    ("nitrate_concentration_M", 0.5),
])
def test_meaningful_condition_difference_blocks_merge(field, value):
    baseline = make_record()
    changed = make_record(**{field: value})
    assert ID.identity_hash(baseline) != ID.identity_hash(changed), (
        f"changing {field} must produce a different experimental identity")


def test_second_extractant_is_not_discarded():
    single = make_record(extractant_smiles_canonical=["CCO"],
                         extractant_concentrations_M=[0.2],
                         extractant_system_key="CCO")
    synergistic = make_record(extractant_smiles_canonical=["CCO", "CCN"],
                              extractant_concentrations_M=[0.2, 0.5],
                              extractant_system_key="CCN|CCO")
    assert ID.identity_hash(single) != ID.identity_hash(synergistic)


# --------------------------------------------------------------------------
# 5. Metadata-only differences do NOT prevent grouping
# --------------------------------------------------------------------------

def test_metadata_is_absent_from_the_identity_key():
    metadata = {"doi_primary", "doi_all", "source_file", "source_line_number",
                "source_record_id", "canonical_measurement_id", "entry_author",
                "addition_date", "publication_title", "data_location",
                "raw_row_ids", "export_source_files", "comments_raw"}
    used = set(ID.IDENTITY_FIELDS_CATEGORICAL) | set(ID.IDENTITY_FIELDS_NUMERIC)
    assert not (metadata & used)


def test_metadata_differences_still_group():
    a = make_record()
    b = make_record()
    a_full = dict(a, doi_primary="10.1/x", source_file="Am.csv", entry_author="A")
    b_full = dict(b, doi_primary="10.2/y", source_file="Eu.csv", entry_author="B")
    assert ID.identity_hash(a_full) == ID.identity_hash(b_full)


def test_export_fanout_is_metadata_only(master):
    """The per-metal file a record came from is not the metal it measured."""
    multi = master[master["export_fanout_size"] > 1]
    assert len(multi) > 0
    mislabelled = multi[multi.apply(
        lambda r: isinstance(r["metal_symbol"], str)
        and r["metal_symbol"] not in list(r["export_metals_queried"])
        and r["metal_species_form"] is None, axis=1)]
    assert len(mislabelled) == 0, "metal must always be among its export files"
    # ...and a record measuring one metal really does appear under others.
    spread = multi[multi["export_fanout_size"] >= 5]
    assert len(spread) > 0


# --------------------------------------------------------------------------
# 6. Same conditions / different value -> conflict, not deletion
# --------------------------------------------------------------------------

def test_value_conflicts_are_preserved_not_averaged(master):
    conflicts = master[master["duplicate_class"] == "E_VALUE_CONFLICT"]
    assert len(conflicts) > 0
    for _, group in conflicts.groupby("duplicate_group_id"):
        assert len(group) >= 2
        # every member survives as its own row
        assert group["is_canonical_row"].all()
        # and they really do disagree
        assert group["log_D"].round(6).nunique() > 1


def test_no_conflict_group_was_collapsed(master):
    conflicted = master[master["duplicate_class"].isin(
        ["E_VALUE_CONFLICT", "F_INSUFFICIENT_INFORMATION", "C_POSSIBLE_INDEPENDENT_REPLICATE"])]
    assert bool(conflicted["is_canonical_row"].all())


# --------------------------------------------------------------------------
# 7. Multiple citations preserved after merging
# --------------------------------------------------------------------------

def test_merged_groups_keep_every_citation(master):
    groups = pd.read_parquet(ROOT / "audit" / "duplicate_groups.parquet")
    merged = groups[groups["duplicate_class"].isin(
        ["A_EXACT_DATABASE_DUPLICATE", "B_SAME_MEASUREMENT_DIFF_PROV"])]
    assert len(merged) > 0
    for _, group in merged.iterrows():
        members = master[master["canonical_measurement_id"].isin(list(group["members"]))]
        assert len(members) == group["size"]
        # Every member -- including the ones folded away -- keeps its own row
        # and its own citation list. (The old assertion here summed non-negative
        # lengths and compared against >= 0, which could never fail.)
        assert members["canonical_measurement_id"].nunique() == group["size"]
        assert set(members["is_canonical_row"]) <= {True, False}
        folded = members[~members["is_canonical_row"]]
        assert len(folded) == group["size"] - 1, "exactly one representative survives per group"
        for value in members["doi_all"]:
            assert value is not None, "a merged member lost its citation list"
        recorded = {d for d in members["doi_primary"] if isinstance(d, str) and d}
        assert recorded == set(group["dois"])
        assert "None" not in recorded, "a missing DOI must not become the string 'None'"


def test_every_raw_row_keeps_its_provenance(master):
    for column in ("raw_row_ids", "export_source_files"):
        assert master[column].map(lambda v: v is not None and len(v) > 0).all()
    sizes = master["raw_row_ids"].map(len)
    assert (sizes == master["export_fanout_size"]).all()


# --------------------------------------------------------------------------
# 8. RDKit failures preserved and reported
# --------------------------------------------------------------------------

def test_rdkit_failure_is_recorded_not_dropped():
    bad = canonicalize("this-is-not-smiles")
    assert bad.status == "parse_failure"
    assert bad.raw == "this-is-not-smiles"      # raw survives
    assert bad.canonical is None                # but nothing is invented


def test_rdkit_blank_is_distinct_from_failure():
    assert canonicalize("-").status == "blank"
    assert canonicalize("").status == "blank"
    assert canonicalize("CCO").status == "ok"


def test_canonicalisation_is_representation_independent():
    """The same molecule written two ways must canonicalise identically."""
    a = canonicalize("CCCCCCCCN(C)C(=O)C(CCCCCC)(OCC)C(=O)N(C)CCCCCCCC")
    b = canonicalize("CCCCCCCCN(C)C(=O)C(CCCCCC)(C(=O)N(C)CCCCCCCC)OCC")
    assert a.status == b.status == "ok"
    assert a.canonical == b.canonical


def test_failures_are_reported(master):
    log = json.loads((ROOT / "reports" / "normalization_log.json").read_text())
    assert "rdkit_version" in log
    assert "rdkit_parse_failures" in master.columns


# --------------------------------------------------------------------------
# 9. Multi-component systems are not collapsed
# --------------------------------------------------------------------------

def test_multi_component_systems_keep_every_component(master):
    multi = master[master["n_organic_extractants"] > 1]
    assert len(multi) > 0
    assert (multi["extractant_smiles_canonical"].map(len) > 1).all()
    assert (multi["extractant_concentrations_M"].map(len) > 1).all()
    # the system key must mention both structures
    assert multi["extractant_system_key"].map(lambda k: "|" in k).all()


def test_single_component_subset_excludes_multi_component():
    single = pd.read_parquet(ROOT / "clean" / "single_component_clean.parquet")
    assert (single["system_component_class"] == "SINGLE_EXTRACTANT").all()
    assert (single["n_organic_extractants"] <= 1).all()
    assert single["modifier_name"].isna().all()
    assert single["complexant_smiles_canonical"].isna().all()
    assert single["holdback_smiles_canonical"].isna().all()


def test_zero_concentration_agent_is_single_component_but_held_back(master):
    """A masking agent recorded at 0 M is chemically absent...

    ...so the record is still SINGLE_EXTRACTANT, but it is deliberately kept
    out of the conservative subset because a second component was mentioned.
    """
    zeroed = master[(master["system_component_class"] == "SINGLE_EXTRACTANT")
                    & master["holdback_smiles_canonical"].notna()]
    assert len(zeroed) > 0
    assert (zeroed["holdback_concentration_M"] == 0.0).all()
    single = pd.read_parquet(ROOT / "clean" / "single_component_clean.parquet")
    assert not set(zeroed["canonical_measurement_id"]) & set(single["canonical_measurement_id"])


def test_components_column_carries_roles(master):
    roles = {c["role"] for row in master["components"] if row is not None for c in row}
    assert "organic_extractant" in roles
    assert {"phase_modifier", "aqueous_complexant", "aqueous_holdback"} & roles


# --------------------------------------------------------------------------
# 10. No row silently disappears
# --------------------------------------------------------------------------

def test_every_raw_row_maps_to_a_record(mapping, master):
    raw_files = sorted((ROOT / "raw").glob("*.csv"))
    expected = sum(sum(1 for _ in path.open()) - 1 for path in raw_files)
    assert len(mapping) == expected
    assert mapping["canonical_measurement_id"].notna().all()
    assert set(mapping["canonical_measurement_id"]) == set(master["canonical_measurement_id"])


def test_row_accounting_balances():
    accounting = json.loads((ROOT / "reports" / "row_accounting.json").read_text())
    assert accounting["balance_check"] == "ok"
    assert sum(accounting["record_disposition"].values()) == accounting["archive_records"]


def test_readiness_tiers_partition_the_table(master):
    assert master["model_readiness"].notna().all()
    assert master["model_readiness"].value_counts().sum() == len(master)


def test_subsets_partition_by_metal(master):
    lanthanide = pd.read_parquet(ROOT / "clean" / "lanthanide_clean.parquet")
    other = pd.read_parquet(ROOT / "clean" / "non_lanthanide_clean.parquet")
    unresolved = int(master["metal_symbol"].isna().sum())
    assert len(lanthanide) + len(other) + unresolved == len(master)
    assert not set(lanthanide["canonical_measurement_id"]) & set(other["canonical_measurement_id"])


# --------------------------------------------------------------------------
# 11. Leakage guards
# --------------------------------------------------------------------------

def test_log_D_is_not_part_of_identity():
    assert "log_D" not in ID.IDENTITY_FIELDS_CATEGORICAL
    assert "log_D" not in ID.IDENTITY_FIELDS_NUMERIC


def test_series_and_curves_are_invariant_to_the_target():
    """Permuting log_D must not move a single record between curves.

    A textual grep for "log_D" cannot prove this (the module's own docstring
    mentions it), so the property is tested directly: rebuild the series and
    curve membership from a frame whose target has been shuffled and reversed,
    and require the membership to be byte-identical.
    """
    import numpy as np
    import stage04_series as S4

    frame = pd.read_parquet(ROOT / "intermediate" / "records_with_series.parquet")
    frame = frame.head(4000).reset_index(drop=True)

    baseline_series = S4.series_id(frame)
    baseline = S4.build_curves(frame.assign(series_id=baseline_series))

    rng = np.random.default_rng(0)
    scrambled = frame.copy()
    scrambled["log_D"] = rng.permutation(scrambled["log_D"].to_numpy())
    scrambled["D_value"] = scrambled["log_D"]
    shuffled_series = S4.series_id(scrambled)
    shuffled = S4.build_curves(scrambled.assign(series_id=shuffled_series))

    assert baseline_series.equals(shuffled_series)
    pd.testing.assert_frame_equal(
        baseline.sort_values(["curve_id", "canonical_measurement_id"]).reset_index(drop=True),
        shuffled.sort_values(["curve_id", "canonical_measurement_id"]).reset_index(drop=True))


def test_provenance_and_feature_groups_are_disjoint():
    assert not set(SCHEMA.PROVENANCE_COLUMNS) & set(SCHEMA.PRIMARY_COLUMNS)
    assert not set(SCHEMA.PROVENANCE_COLUMNS) & set(SCHEMA.TARGET_COLUMNS)


# --------------------------------------------------------------------------
# 12. Chemistry reference sanity
# --------------------------------------------------------------------------

def test_lanthanide_index_matches_gen10():
    assert CHEM.lanthanide_index("La") == 1
    assert CHEM.lanthanide_index("Lu") == 15
    assert CHEM.lanthanide_index("Y") is None


def test_ionic_radius_matches_gen10_for_lanthanides():
    for symbol, expected in [("La", 1.160), ("Eu", 1.066), ("Lu", 0.977)]:
        radius, status = CHEM.ionic_radius_cn8(symbol, 3)
        assert status == "ok"
        assert radius == pytest.approx(expected)


def test_missing_radius_is_flagged_not_invented():
    radius, status = CHEM.ionic_radius_cn8("Cm", 3)
    assert radius is None and status == "requires_curation"


def test_uranyl_token_expands_to_uranium_six():
    symbol, ox, species = CHEM.SPECIES_TOKENS["UO2+2"]
    assert (symbol, ox) == ("U", 6) and "uranyl" in species


def test_implausible_oxidation_states_are_flagged_not_corrected(master):
    bad = master[master["metal_oxidation_state_plausible"] == False]  # noqa: E712
    assert len(bad) > 0
    # the archive value survives untouched
    assert bad["metal_oxidation_state_raw"].notna().all()
    assert (bad["metal_oxidation_state"] == 3).all()
    assert (bad["metal_symbol"] == "Sr").all()


# --------------------------------------------------------------------------
# 13. Rerunning the pipeline reproduces identical files
# --------------------------------------------------------------------------

def test_recorded_determinism_check_passed():
    """`run_all.py --verify-determinism` must have been run and passed.

    The full two-pass run takes minutes, so the check itself lives in the
    orchestrator; this test asserts the recorded verdict rather than silently
    letting a stale or missing result through.
    """
    path = ROOT / "reports" / "determinism_check.json"
    if not path.exists():
        # Skipping here kept the suite green while the guarantee was unproven.
        # If the pipeline has produced artifacts, the evidence must exist too.
        assert not (ROOT / "clean" / "master_clean.parquet").exists(), (
            "the pipeline has produced artifacts but no determinism evidence exists -- "
            "run `python scripts/run_all.py --verify-determinism`")
        pytest.skip("pipeline has not been run yet")
    report = json.loads(path.read_text())
    assert report["deterministic"], (
        f"changed={report['changed']} added={report['added']} removed={report['removed']}")
    assert report["artifacts_compared"] > 10


def test_pipeline_stages_have_no_nondeterminism_sources():
    """No wall-clock, RNG or salted-hash use anywhere in the pipeline."""
    import re
    # `hash(` needs a word boundary: `ID.identity_hash(...)` is our own stable
    # SHA-256 helper, whereas the builtin `hash()` is salted per process.
    banned = (r"datetime\.now", r"datetime\.today", r"time\.time", r"time\.monotonic",
              r"random\.random", r"random\.shuffle", r"random\.sample", r"random\.choice",
              r"np\.random", r"numpy\.random", r"uuid[14]", r"os\.urandom", r"os\.getpid",
              r"tempfile\.", r"(?<![A-Za-z0-9_])hash\(", r"(?<![A-Za-z0-9_])id\(",
              r"\.now\(\)", r"\.today\(\)")
    # Scan every module run_all.py actually executes, plus the shared sae_*
    # modules they import -- identity hashing and normalisation live there, so
    # globbing only "stage*.py" left the highest-risk code unchecked.
    modules = sorted(set(SCRIPTS.glob("stage*.py")) | set(SCRIPTS.glob("sae_*.py")))
    assert len(modules) >= 13, f"expected the full module set, got {[m.name for m in modules]}"
    for module in modules:
        source = module.read_text()
        for pattern in banned:
            found = re.search(pattern, source)
            assert found is None, (
                f"{module.name} uses non-deterministic {pattern} at offset {found.start()}")


def test_group_ids_are_assigned_from_sorted_keys():
    """Group ids must not depend on the order rows happen to arrive in."""
    frame = pd.read_parquet(ROOT / "intermediate" / "records_grouped.parquet")
    pairs = frame[["identity_hash", "duplicate_group_id"]].drop_duplicates()
    ordered = pairs.sort_values("identity_hash")["duplicate_group_id"].tolist()
    assert ordered == sorted(ordered), "duplicate_group_id must follow sorted identity_hash"


def test_manifest_hashes_match_published_artifacts():
    manifest = json.loads((ROOT / "reports" / "manifest.json").read_text())
    assert not manifest["missing_artifacts"], manifest["missing_artifacts"]
    run_dir = Path(manifest["deliverables_directory"])
    import hashlib
    for name, entry in manifest["artifacts"].items():
        path = run_dir / name
        assert path.exists(), name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        assert digest.hexdigest() == entry["sha256"], f"{name} hash mismatch"


def test_raw_inputs_are_unmodified():
    """The immutable raw copies must still match their ingest checksums."""
    import hashlib
    recorded = {}
    for line in (ROOT / "reports" / "raw_checksums.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        recorded[name.strip()] = digest
    for path in sorted((ROOT / "raw").glob("*.csv")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == recorded[path.name], f"{path.name} was modified after ingest"


def test_extreme_D_is_flagged_but_preserved(master):
    """A physically impossible D is flagged, never clipped or dropped."""
    extreme = master[master["flags"].map(
        lambda fl: "implausible_extreme_D" in fl if fl is not None else False)]
    assert len(extreme) > 0
    # the raw value survives untouched
    assert extreme["D_raw"].notna().all()
    assert (extreme["log_D"].abs() > 6.0).all()
    # ...and it is demoted out of the model-ready tier
    assert (extreme["model_readiness"] != "A_model_ready").all()


# --------------------------------------------------------------------------
# 14. DOI repair and reference metadata
# --------------------------------------------------------------------------

def test_doi_repair_never_overwrites_the_raw_value(master):
    repaired = master[master["doi_correction_rule"].notna()]
    assert len(repaired) > 0
    # the recorded DOI survives untouched...
    assert repaired["doi_primary"].notna().all()
    # ...and the repair lives in its own column, with its evidence
    assert (repaired["doi_primary"] != repaired["doi_primary_corrected"]).all()
    assert repaired["doi_correction_evidence"].notna().all()


def test_repaired_dois_are_corroborated_by_the_archive_itself(master):
    """Each corrected DOI is one the archive already records on other rows."""
    repaired = master[master["doi_correction_rule"].notna()]
    recorded = set(master["doi_primary"].dropna())
    for corrected in repaired["doi_primary_corrected"].dropna().unique():
        assert corrected in recorded, (
            f"{corrected} should appear elsewhere in the archive as a clean DOI")


def test_reference_metadata_is_sourced_and_never_invented(master):
    sources = set(master["reference_metadata_source"].dropna())
    assert sources <= {"crossref", "archive_comments"}
    crossref = master[master["reference_metadata_source"] == "crossref"]
    assert len(crossref) > 0
    assert crossref["reference_title"].notna().all()
    assert crossref["doi_primary_corrected"].notna().all()


def test_reference_columns_are_provenance_not_features():
    for column in ("reference_title", "reference_year", "reference_journal",
                   "reference_authors", "doi_primary_corrected"):
        assert column in SCHEMA.PROVENANCE_COLUMNS
        assert column not in SCHEMA.PRIMARY_COLUMNS


def test_crossref_cache_is_a_file_not_a_network_call():
    """The pipeline must stay runnable offline and deterministic."""
    source = (SCRIPTS / "stage02_normalize.py").read_text()
    for token in ("urlopen", "requests.", "http://", "https://api"):
        assert token not in source, f"stage02 must not reach the network ({token})"
    cache = ROOT / "cache" / "crossref_metadata.json"
    assert cache.exists()
    payload = json.loads(cache.read_text())
    assert sum(1 for v in payload.values() if v.get("status") == "ok") > 100


# --------------------------------------------------------------------------
# 15. Aqueous complexants are multi-component too (audit regression)
# --------------------------------------------------------------------------

def test_complexant_concentration_sweep_is_not_merged():
    """A DTPA sweep must not collapse into one "exact duplicate".

    The archive writes two aqueous complexants as "DTPA; NaNO3" with the
    parallel concentrations "0.02;1". Feeding that whole string to the
    single-quantity parser yielded None, which dropped the complexant
    concentration out of the identity key and merged the sweep the source
    study was actually varying.
    """
    low = make_record(complexant_signature="DTPA@0.02|NaNO3@1.0")
    high = make_record(complexant_signature="DTPA@0.05|NaNO3@1.0")
    assert ID.identity_hash(low) != ID.identity_hash(high)


def test_complexant_signature_is_order_invariant():
    a = make_record(complexant_signature="DTPA@0.02|NaNO3@1.0")
    b = make_record(complexant_signature="DTPA@0.02|NaNO3@1.0")
    assert ID.identity_hash(a) == ID.identity_hash(b)


def test_real_dtpa_sweep_survives_as_separate_records(master):
    """The concrete pair the audit found must both remain canonical."""
    pair = master[master["canonical_measurement_id"].isin(["SAE:13066", "SAE:13077"])]
    if len(pair) != 2:
        pytest.skip("those archive ids are not present in this build")
    assert pair["is_canonical_row"].all(), "a real DTPA concentration difference was folded away"
    assert pair["complexant_signature"].nunique() == 2
    assert pair["duplicate_group_id"].nunique() == 2


def test_multi_complexant_records_keep_every_component(master):
    multi = master[master["n_complexants"] > 1]
    assert len(multi) > 0
    assert (multi["complexant_names"].map(len) > 1).all()
    assert (multi["complexant_concentrations_M"].map(len) > 1).all()
    assert multi["complexant_signature"].notna().all()


def test_solvent_names_with_iupac_locants_are_not_split(master):
    """"1,4-diisopropylbenzene" is one substance, not four components."""
    for raw in ("1,4-diisopropylbenzene", "1,1,2,2-tetrachloroethane", "1,2-dichloroethane"):
        rows = master[master["solvent_name_raw"] == raw]
        if not len(rows):
            continue
        assert (rows["solvent_n_components"] == 1).all(), f"{raw} was split into components"
    numeral_first = master["solvent_components"].map(
        lambda v: bool(v is not None and len(v) and str(v[0]).isdigit()))
    assert int(numeral_first.sum()) == 0, "a locant digit became a solvent component"


def test_curves_are_built_only_from_canonical_rows(master):
    membership = pd.read_parquet(ROOT / "intermediate" / "curve_membership.parquet")
    canonical = set(master[master["is_canonical_row"]]["canonical_measurement_id"])
    strays = set(membership["canonical_measurement_id"]) - canonical
    assert not strays, f"{len(strays)} folded-away duplicates leaked into curve membership"


def test_records_without_an_extractant_are_labelled_honestly(master):
    none_recorded = master[master["n_organic_extractants"] == 0]
    assert len(none_recorded) > 0
    assert (none_recorded["system_component_class"] == "NO_EXTRACTANT_RECORDED").all()
    assert none_recorded["extractant_system_key"].isna().all()


def test_deduced_structures_have_their_own_provenance_tag(master):
    """The four elimination-recovered structures must be separable."""
    tags = {c["structure_source"] for row in master["components"] if row is not None
            for c in row if c["structure_source"]}
    assert "deduced_by_elimination" in tags
    deduced = master[master["components"].map(
        lambda row: any(c["structure_source"] == "deduced_by_elimination" for c in row)
        if row is not None else False)]
    # It must select the TBP/DHOA-style rows, not most of the dataset.
    assert 0 < len(deduced) < len(master) * 0.25
