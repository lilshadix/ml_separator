"""Tests for the two gen6 provenance scripts.

``scripts/build_chemistry_map.py`` freezes the description of chemical space that
every later gen6 run hashes into its manifest, and ``scripts/aggregate_gen6.py``
is the thing that refuses to put two incomparable runs in one table.  Both are
scripts rather than package modules, so they are loaded here through
``importlib`` (there is no ``scripts`` package and importing one must not depend
on the working directory).

What these tests are actually protecting:

* the freezer writes **every** promised artifact and the map it wrote can be read
  back byte-for-byte — a frozen artifact that does not round-trip is not frozen;
* the freezer answers the Phase 0 question (who enters at ``min_cells = 3`` but
  not at 10) from a synthetic cohort where the answer is known by construction;
* the aggregator **refuses** two runs whose dataset hashes differ, and succeeds
  on two that agree.  The refusal is the load-bearing behaviour: two earlier
  generations of this repo silently mixed runs, and a test that only checked the
  happy path would not have caught it.

The synthetic source table is built in-file (the convention of
``tests/test_levels.py``) and carries the minimum the gen5 harness needs:
identity, metal, fingerprint and condition columns.  Ligand 0..3 get many cells
and ligand 4..5 get few, so the 3-vs-10 entrant analysis has a known answer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_script(name: str) -> ModuleType:
    """Import a file under ``scripts/`` as a module without a package."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gen6_script_{name}", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_chemistry_map_script = _load_script("build_chemistry_map")
aggregate_gen6 = _load_script("aggregate_gen6")


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #

#: (extractant index -> number of distinct conditions).  Two ligands are
#: deliberately sparse so that the 3-vs-10 entrant table is non-empty and its
#: expected content is known before the script runs.
CELLS_PER_LIGAND: dict[int, int] = {0: 8, 1: 8, 2: 8, 3: 8, 4: 2, 5: 2}
N_METALS = 2
N_BITS = 64


def _synthetic_source(seed: int = 11) -> pd.DataFrame:
    """A source table with every column ``build_level_dataset`` insists on."""
    rng = np.random.default_rng(seed)
    metals = [("Nd", 60.0), ("Eu", 63.0)]
    rows: list[dict] = []
    fingerprints: dict[int, np.ndarray] = {}
    for ligand, n_conditions in CELLS_PER_LIGAND.items():
        bits = np.zeros(N_BITS, dtype=int)
        bits[rng.choice(N_BITS, size=12, replace=False)] = 1
        if ligand == 1:
            # a near-copy of ligand 0, so the two share a super-cluster and the
            # frozen partition has something non-trivial to be stable about
            bits = fingerprints[0].copy()
            bits[int(rng.integers(N_BITS))] ^= 1
        fingerprints[ligand] = bits
        for condition in range(n_conditions):
            for symbol, atomic_number in metals[:N_METALS]:
                row = {
                    "canonical_smiles": f"SMILES_{ligand}",
                    "extractant_name": f"LIGAND-{ligand}",
                    "metal_symbol": symbol,
                    "Atomic Number_metal": atomic_number,
                    "lanthanide_index": atomic_number - 57.0,
                    "Ionic Radius_metal": 1.0 + 0.01 * atomic_number,
                    "log_D": float(rng.normal(loc=ligand * 0.5, scale=0.4)),
                    "cond__acid_concentration_M": 0.5 + condition,
                    "cond__extractant_concentration_M": 0.05 * (condition + 1),
                    "cond__diluent": "kerosene" if ligand % 2 == 0 else "dodecane",
                    "DONOR_TYPES": json.dumps(["O(amide_carbonyl)", "O(ether)"]),
                    "DENTATE": 3, "coreCN": 9, "n_ligs": 3, "n_fill": 0,
                    "MolWt": 400.0 + ligand, "TPSA": 50.0, "NumHDonors": 0.0,
                    "NumHAcceptors": 4.0, "NumRotatableBonds": 12.0, "NumAromaticRings": 0.0,
                    "NumAliphaticRings": 0.0, "RingCount": 0.0, "FractionCSP3": 0.9,
                    "MolLogP": 6.0,
                }
                row.update({f"ecfp_{b}": int(bits[b]) for b in range(N_BITS)})
                rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_descriptors() -> pd.DataFrame:
    """The rdkit-derived motif columns the family assignment prefers."""
    records = []
    for ligand in CELLS_PER_LIGAND:
        records.append({
            "canonical_smiles": f"SMILES_{ligand}",
            "lig2d__hc__n_dga_motifs": 1.0 if ligand < 2 else 0.0,
            "lig2d__hc__n_malonamide_motifs": 1.0 if ligand == 2 else 0.0,
            "lig2d__hc__n_phosphoryl_O": 1.0 if ligand == 3 else 0.0,
            "lig2d__hc__n_aromatic_N": 3.0 if ligand == 4 else 0.0,
            "lig2d__hc__n_ether_O": 2.0 if ligand == 5 else 0.0,
            "lig2d__hc__n_amide_N": 2.0 if ligand < 3 else 0.0,
            "lig2d__hc__n_carbonyl_O": 2.0 if ligand < 3 else 0.0,
            "lig2d__hc__n_hydroxyl_O": 0.0,
            "lig2d__mw": 400.0 + ligand,
        })
    return pd.DataFrame(records)


@pytest.fixture()
def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "source.parquet"
    descriptors = tmp_path / "descriptors.parquet"
    _synthetic_source().to_parquet(dataset, index=False)
    _synthetic_descriptors().to_parquet(descriptors, index=False)
    return dataset, descriptors


# --------------------------------------------------------------------------- #
# build_chemistry_map.py
# --------------------------------------------------------------------------- #

def test_build_chemistry_map_writes_every_promised_artifact(synthetic_inputs, tmp_path):
    dataset, descriptors = synthetic_inputs
    out = tmp_path / "chem"
    code = build_chemistry_map_script.main(
        ["--dataset", str(dataset), "--descriptors", str(descriptors), "--output-dir", str(out)])
    assert code == 0

    for name in build_chemistry_map_script.REQUIRED_ARTIFACTS:
        path = out / name
        assert path.is_file(), f"missing artifact {name}"
        assert path.stat().st_size > 0, f"empty artifact {name}"
    for name in ("validation.json", "artifact_hashes.json", "_SUCCESS.json", "log.txt"):
        assert (out / name).is_file(), f"missing {name}"
    assert not (out / "_FAILED.json").exists()

    validation = json.loads((out / "validation.json").read_text())
    assert validation["ok"] is True
    assert validation["missing_manifest_keys"] == []
    assert validation["failed_checks"] == []


def test_build_chemistry_map_round_trips_the_frozen_map(synthetic_inputs, tmp_path):
    """A frozen artifact that does not read back identically is not frozen."""
    from lanthanide_separation.gen6.chemistry import ChemistryMap
    from lanthanide_separation.gen6.manifest import sha256_frame

    dataset, descriptors = synthetic_inputs
    out = tmp_path / "chem"
    assert build_chemistry_map_script.main(
        ["--dataset", str(dataset), "--descriptors", str(descriptors), "--output-dir", str(out)]) == 0

    reloaded = ChemistryMap.from_parquet(out / "chemistry_map.parquet")
    assert len(reloaded) == len(CELLS_PER_LIGAND)
    assert set(reloaded.table["extractant"]) == {f"SMILES_{i}" for i in CELLS_PER_LIGAND}
    assert reloaded.similarity.shape == (len(CELLS_PER_LIGAND), len(CELLS_PER_LIGAND))
    assert np.allclose(np.diag(reloaded.similarity), 1.0)
    assert np.allclose(reloaded.similarity, reloaded.similarity.T)

    # the standalone npz deliverable must agree with the sidecar
    with np.load(out / "nearest_neighbor_matrix.npz", allow_pickle=True) as payload:
        assert np.array_equal(payload["similarity"], reloaded.similarity)
        assert [str(x) for x in payload["extractants"]] == list(reloaded.extractants)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["chemistry_cluster_definition"]["chemistry_map_sha256"] == sha256_frame(
        reloaded.table, sort_rows_by=["extractant"])

    cluster_manifest = json.loads((out / "chemistry_cluster_manifest.json").read_text())
    members = {m for group in cluster_manifest["superclusters"].values() for m in group}
    assert members == set(reloaded.table["extractant"])


def test_build_chemistry_map_reports_the_min_cells_entrants(synthetic_inputs, tmp_path):
    """The Phase 0 question, on a cohort whose answer is known by construction."""
    dataset, descriptors = synthetic_inputs
    out = tmp_path / "chem"
    assert build_chemistry_map_script.main(
        ["--dataset", str(dataset), "--descriptors", str(descriptors), "--output-dir", str(out)]) == 0

    summary = json.loads((out / "summary.json").read_text())
    entrants = summary["entrants"]
    assert entrants["status"] == "OK"
    # ligands 4 and 5 have 2 conditions x 2 metals = 4 cells: eligible at 3, not at 10
    assert entrants["n_entrants"] == 2
    assert entrants["n_base_extractants"] == 4
    assert entrants["n_expanded_extractants"] == 6
    assert entrants["base_is_subset_of_expanded"] is True

    table = pd.read_csv(out / "cohort_entrants.csv")
    assert set(table["extractant"]) == {"SMILES_4", "SMILES_5"}
    assert (table["cohort_n_cells"] == 4).all()
    assert table["max_tanimoto_to_base"].between(0.0, 1.0).all()

    # the partition-stability claim must be present, checked, and true here
    stability = summary["partition_stability"]
    assert set(stability) == {"3", "10"}
    for check in stability.values():
        assert check["ok"] is True
        assert check["ecfp"]["local_groups_spanning_multiple_frozen"] == 0
        assert check["supercluster"]["frozen_groups_spanning_multiple_local"] == 0

    report = (out / "chemistry_report.md").read_text()
    assert "Phase 0 question" in report
    assert "LIGAND-4" in report and "LIGAND-5" in report
    assert "Partition stability" in report


def test_build_chemistry_map_ignores_the_target(synthetic_inputs, tmp_path):
    """Blanking ``log_D`` must not move a single cell of the map.

    The cohorts cannot be built without a target, so this compares the map
    artifact only — which is exactly the claim: the frozen chemistry is
    target-independent.
    """
    from lanthanide_separation.gen6.chemistry import ChemistryMap
    from lanthanide_separation.gen6.manifest import sha256_frame

    dataset, descriptors = synthetic_inputs
    scrambled = tmp_path / "scrambled.parquet"
    frame = pd.read_parquet(dataset)
    rng = np.random.default_rng(7)
    frame["log_D"] = rng.normal(size=len(frame)) * 10.0
    frame.to_parquet(scrambled, index=False)

    first, second = tmp_path / "a", tmp_path / "b"
    assert build_chemistry_map_script.main(
        ["--dataset", str(dataset), "--descriptors", str(descriptors), "--output-dir", str(first)]) == 0
    assert build_chemistry_map_script.main(
        ["--dataset", str(scrambled), "--descriptors", str(descriptors),
         "--output-dir", str(second)]) == 0

    a = ChemistryMap.from_parquet(first / "chemistry_map.parquet")
    b = ChemistryMap.from_parquet(second / "chemistry_map.parquet")
    assert sha256_frame(a.table, sort_rows_by=["extractant"]) == sha256_frame(
        b.table, sort_rows_by=["extractant"])
    assert np.array_equal(a.similarity, b.similarity)


def test_build_chemistry_map_survives_a_missing_descriptor_table(synthetic_inputs, tmp_path):
    """No descriptor parquet: families fall back, and the fallback is declared."""
    dataset, _ = synthetic_inputs
    out = tmp_path / "chem"
    assert build_chemistry_map_script.main(
        ["--dataset", str(dataset), "--descriptors", "", "--output-dir", str(out)]) == 0
    table = pd.read_parquet(out / "chemistry_map.parquet")
    # without the motif columns the map falls back to the DONOR_TYPES census first
    # and to the declared SMILES heuristic last; both are labelled per row
    assert set(table["chem__family_source"]) <= {
        "smiles_heuristic", "donor_census", "donor_census_fallback"}


# --------------------------------------------------------------------------- #
# aggregate_gen6.py — fake run directories
# --------------------------------------------------------------------------- #

def _fake_run(
    directory: Path,
    *,
    run_id: str,
    dataset_sha: str = "a" * 64,
    source_sha: str = "b" * 64,
    feature_sha: str = "c" * 64,
    cohort_min_cells: int = 3,
    leaderboard: pd.DataFrame | None = None,
    bootstrap: pd.DataFrame | None = None,
    validated: bool = True,
) -> Path:
    """A minimal but structurally honest gen6 run directory."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layer": "gen6",
        "run_id": run_id,
        "created_utc": "2026-08-19T00:00:00+00:00",
        "dataset_path": "dataset.parquet",
        "dataset_file_sha256": dataset_sha,
        "source_table_sha256": source_sha,
        "feature_registry_sha256": feature_sha,
        "code_sha256": {"scripts/fake.py": "d" * 64},
        "split_definition": {"kind": "supercluster_holdout", "n_splits": 5},
        "chemistry_cluster_definition": {"supercluster": "single-linkage Tanimoto >= 0.7"},
        "provenance_state": {"variant": "legacy_compatible"},
        "model_seed": 42,
        "split_seeds": [104729],
        "preprocessing": [{"step": "build_level_dataset"}],
        "folds": [],
        "cohort_definition": {"shared_cohort_min_cells": cohort_min_cells,
                              "replicate_policy": "mean"},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (directory / "summary.json").write_text(json.dumps({"run_id": run_id}, indent=2))
    (directory / "validation.json").write_text(json.dumps({"ok": validated}, indent=2))
    if validated:
        (directory / "_SUCCESS.json").write_text(json.dumps({"run_id": run_id}, indent=2))
    if leaderboard is not None:
        leaderboard.to_csv(directory / "leaderboard.csv", index=False)
    if bootstrap is not None:
        bootstrap.to_csv(directory / "paired_bootstrap.csv", index=False)
    return directory


def _leaderboard(offset: float) -> pd.DataFrame:
    return pd.DataFrame({
        "arm": ["BASE", "EXPANDED", "EXPANDED_SHUFFLED"],
        "macro_mae": [0.90 + offset, 0.85 + offset, 0.91 + offset],
        "offset_mae": [0.70 + offset, 0.62 + offset, 0.71 + offset],
        "n_ligands": [152, 152, 152],
    })


def _bootstrap(low: float) -> pd.DataFrame:
    return pd.DataFrame({
        "comparison": ["BASE_vs_EXPANDED"],
        "reference": ["BASE"],
        "candidate": ["EXPANDED"],
        "statistic": ["mae"],
        "point_delta": [0.05],
        "ci95_low": [low],
        "ci95_high": [0.09],
    })


def test_aggregate_refuses_runs_with_different_dataset_hashes(tmp_path):
    """The load-bearing behaviour: incomparable runs must not be combined."""
    first = _fake_run(tmp_path / "run_a", run_id="gen6_a", dataset_sha="a" * 64,
                      leaderboard=_leaderboard(0.0))
    second = _fake_run(tmp_path / "run_b", run_id="gen6_b", dataset_sha="f" * 64,
                       leaderboard=_leaderboard(0.1))
    out = tmp_path / "agg"
    code = aggregate_gen6.main([str(first), str(second), "--output-dir", str(out)])
    assert code != 0

    payload = json.loads((out / "compatibility.json").read_text())
    entry = payload["compatibility"]["keys"]["dataset_file_sha256"]
    assert entry["status"] == "MISMATCH"
    assert entry["n_distinct_values"] == 2
    assert payload["compatibility"]["ok"] is False
    # the refusal must be discoverable from the directory listing, not only stderr
    assert (out / "_FAILED.json").is_file()
    assert not (out / "_SUCCESS.json").exists()
    assert "FATAL" in (out / "aggregate_decision_report.md").read_text()


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_sha", "e" * 64), ("feature_sha", "e" * 64), ("cohort_min_cells", 10)],
)
def test_aggregate_refuses_every_other_fingerprint_mismatch(tmp_path, field, value):
    """Dataset hash is not a special case — every fingerprint is fatal."""
    first = _fake_run(tmp_path / "run_a", run_id="gen6_a", leaderboard=_leaderboard(0.0))
    second = _fake_run(tmp_path / "run_b", run_id="gen6_b", leaderboard=_leaderboard(0.1),
                       **{field: value})
    code = aggregate_gen6.main([str(first), str(second), "--output-dir", str(tmp_path / "agg")])
    assert code != 0


def test_aggregate_succeeds_on_two_compatible_runs(tmp_path):
    first = _fake_run(tmp_path / "run_a", run_id="gen6_a",
                      leaderboard=_leaderboard(0.0), bootstrap=_bootstrap(0.01))
    second = _fake_run(tmp_path / "run_b", run_id="gen6_b",
                       leaderboard=_leaderboard(0.1), bootstrap=_bootstrap(-0.02))
    out = tmp_path / "agg"
    assert aggregate_gen6.main([str(first), str(second), "--output-dir", str(out)]) == 0

    for name in aggregate_gen6.REQUIRED_ARTIFACTS + ("_SUCCESS.json", "validation.json",
                                                     "artifact_hashes.json"):
        assert (out / name).is_file(), f"missing {name}"
    assert json.loads((out / "validation.json").read_text())["ok"] is True

    combined = pd.read_csv(out / "combined_leaderboard.csv")
    assert len(combined) == 6
    assert set(combined["run_id"]) == {"gen6_a", "gen6_b"}

    by_arm = pd.read_csv(out / "leaderboard_by_arm.csv")
    assert set(by_arm["arm"]) == {"BASE", "EXPANDED", "EXPANDED_SHUFFLED"}
    assert (by_arm["n_runs"] == 2).all()
    # EXPANDED is the lowest macro MAE in both runs, so it must rank first
    assert by_arm.sort_values("rank").iloc[0]["arm"] == "EXPANDED"

    consistency = pd.read_csv(out / "bootstrap_consistency.csv")
    row = consistency.iloc[0]
    assert row["n_runs"] == 2
    # the contrast points the same way in both runs but only one CI clears zero:
    # exactly the "not established" state the report warns about
    assert row["n_delta_positive"] == 2
    assert row["n_ci95_low_above_zero"] == 1


def test_aggregate_never_averages_across_a_facet(tmp_path):
    """An 'overall' row and a 'hard chemistry' row are not the same measurement.

    Both the leaderboard and the bootstrap view must keep them apart; collapsing
    them would produce a number that describes neither, which is exactly the
    silent-mixing failure this script exists to prevent.
    """
    leaderboard = pd.DataFrame({
        "policy": ["diversity", "diversity", "depth", "depth"],
        "budget": [500, 500, 500, 500],
        "endpoint": ["overall", "nn<0.4", "overall", "nn<0.4"],
        "macro_mae": [1.0, 2.0, 1.4, 3.0],
    })
    bootstrap = pd.DataFrame({
        "comparison": ["diversity_vs_depth"] * 2,
        "reference": ["depth"] * 2,
        "candidate": ["diversity"] * 2,
        "statistic": ["mae"] * 2,
        "endpoint": ["overall", "nn<0.4"],
        "point_delta": [0.4, 1.0],
        "ci95_low": [0.1, -0.2],
        "ci95_high": [0.7, 2.1],
    })
    run = _fake_run(tmp_path / "run_a", run_id="gen6_a",
                    leaderboard=leaderboard, bootstrap=bootstrap)
    out = tmp_path / "agg"
    assert aggregate_gen6.main([str(run), "--output-dir", str(out)]) == 0

    # arm column resolves to `policy` (there is no `arm` column here)
    by_arm = pd.read_csv(out / "leaderboard_by_arm.csv")
    assert json.loads((out / "summary.json").read_text())["arm_column"] == "policy"
    assert len(by_arm) == 4
    assert set(by_arm["endpoint"]) == {"overall", "nn<0.4"}
    assert (by_arm["n_rows"] == 1).all()

    consistency = pd.read_csv(out / "bootstrap_consistency.csv")
    assert len(consistency) == 2
    assert set(consistency["endpoint"]) == {"overall", "nn<0.4"}
    assert consistency.set_index("endpoint").loc["overall", "n_ci95_low_above_zero"] == 1
    assert consistency.set_index("endpoint").loc["nn<0.4", "n_ci95_low_above_zero"] == 0


def test_aggregate_works_on_a_single_run_directory(tmp_path):
    """Phase 0 calls the aggregator with one directory."""
    only = _fake_run(tmp_path / "run_a", run_id="gen6_a", leaderboard=_leaderboard(0.0))
    out = tmp_path / "agg"
    assert aggregate_gen6.main([str(only), "--output-dir", str(out)]) == 0
    inventory = pd.read_csv(out / "runs_inventory.csv")
    assert len(inventory) == 1
    assert inventory.iloc[0]["run_id"] == "gen6_a"


def test_aggregate_tolerates_missing_optional_artifacts(tmp_path):
    """A run with no leaderboard and no bootstrap is inventoried, not rejected."""
    rich = _fake_run(tmp_path / "run_a", run_id="gen6_a", leaderboard=_leaderboard(0.0))
    bare = _fake_run(tmp_path / "run_b", run_id="gen6_b")
    for optional in ("summary.json", "validation.json", "_SUCCESS.json"):
        (bare / optional).unlink()
    out = tmp_path / "agg"
    assert aggregate_gen6.main([str(rich), str(bare), "--output-dir", str(out)]) == 0
    inventory = pd.read_csv(out / "runs_inventory.csv").set_index("run_id")
    assert inventory.loc["gen6_b", "leaderboard_rows"] == 0
    assert "no leaderboard CSV" in inventory.loc["gen6_b", "problems"]
    assert len(pd.read_csv(out / "combined_leaderboard.csv")) == 3


def test_aggregate_strict_keys_turns_an_absent_fingerprint_into_a_failure(tmp_path):
    first = _fake_run(tmp_path / "run_a", run_id="gen6_a", leaderboard=_leaderboard(0.0))
    second = _fake_run(tmp_path / "run_b", run_id="gen6_b", leaderboard=_leaderboard(0.1))
    manifest = json.loads((second / "manifest.json").read_text())
    del manifest["cohort_definition"]
    del manifest["split_definition"]
    del manifest["preprocessing"]
    (second / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lenient = tmp_path / "agg_lenient"
    assert aggregate_gen6.main([str(first), str(second), "--output-dir", str(lenient)]) == 0
    entry = json.loads((lenient / "compatibility.json").read_text())
    assert entry["compatibility"]["keys"]["cohort_definition"]["status"] == "ABSENT (tolerated)"

    strict = tmp_path / "agg_strict"
    assert aggregate_gen6.main(
        [str(first), str(second), "--output-dir", str(strict), "--strict-keys"]) != 0


def test_aggregate_refuses_a_directory_without_a_manifest(tmp_path):
    good = _fake_run(tmp_path / "run_a", run_id="gen6_a", leaderboard=_leaderboard(0.0))
    empty = tmp_path / "not_a_run"
    empty.mkdir()
    assert aggregate_gen6.main([str(good), str(empty), "--output-dir", str(tmp_path / "agg")]) != 0


def test_aggregate_refuses_duplicate_run_ids(tmp_path):
    """Two directories claiming one run_id would double-count every metric row."""
    first = _fake_run(tmp_path / "run_a", run_id="gen6_same", leaderboard=_leaderboard(0.0))
    second = _fake_run(tmp_path / "run_b", run_id="gen6_same", leaderboard=_leaderboard(0.1))
    assert aggregate_gen6.main([str(first), str(second), "--output-dir", str(tmp_path / "agg")]) != 0


def test_aggregate_accepts_a_real_chemistry_map_run(synthetic_inputs, tmp_path):
    """End to end: the freezer's output is a valid input to the aggregator."""
    dataset, descriptors = synthetic_inputs
    chemistry_dir = tmp_path / "chem"
    assert build_chemistry_map_script.main(
        ["--dataset", str(dataset), "--descriptors", str(descriptors),
         "--output-dir", str(chemistry_dir)]) == 0
    out = tmp_path / "agg"
    assert aggregate_gen6.main([str(chemistry_dir), "--output-dir", str(out)]) == 0
    report = (out / "aggregate_decision_report.md").read_text()
    assert "No leaderboard CSV was found" in report
    assert json.loads((out / "validation.json").read_text())["ok"] is True
