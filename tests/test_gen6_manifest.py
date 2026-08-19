"""Tests for the gen6 run manifest: hashing, fail-closed validation, success markers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen6.manifest import (
    REQUIRED_FOLD_KEYS, REQUIRED_MANIFEST_KEYS, RunManifest, artifact_hashes, code_hashes,
    sha256_file, sha256_frame, sha256_json, validate_run, write_success,
)


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "row_id": [f"r{i:03d}" for i in range(20)],
        "extractant": ["A", "B"] * 10,
        "log_D": rng.normal(size=20),
        "count": rng.integers(0, 5, size=20),
        "flag": rng.integers(0, 2, size=20).astype(bool),
    })


def _fold_record(fold: int = 0) -> dict:
    return {
        "fold": fold, "split_seed": 104729, "test_row_ids_sha256": "deadbeef",
        "test_extractants": ["A"], "test_superclusters": ["tan001"],
        "train_extractants_by_arm": {"BASE": ["B"], "EXPANDED": ["B", "C"]},
        "train_superclusters_by_arm": {"BASE": ["tan002"], "EXPANDED": ["tan002", "tan003"]},
        "n_test_rows": 10, "n_train_rows_by_arm": {"BASE": 10, "EXPANDED": 14},
    }


def _complete_manifest(tmp_path) -> RunManifest:
    dataset = tmp_path / "dataset.parquet"
    frame = _frame()
    frame.to_parquet(dataset)
    manifest = RunManifest(layer="gen6_diversity", run_id="test_run")
    manifest.record_dataset(dataset_path=dataset, source_frame=frame)
    manifest.record_code([dataset])  # any existing file works as a stand-in
    manifest.record_features(feature_sets={"MC": ["metal_Z", "cond__a"]})
    manifest.record_split(definition={"algorithm": "round_robin", "group": "tanimoto_cluster"},
                          folds=[_fold_record(0), _fold_record(1)])
    manifest.record_chemistry(definition={"threshold": 0.7, "n_extractants": 190})
    manifest.record_provenance(state={"publication_id": "unavailable"})
    manifest.record_preprocessing([{"step": "median_impute", "fitted_on": "training fold only"}])
    manifest.record("model_seed", 42)
    manifest.record("split_seeds", [104729])
    return manifest


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #

def test_frame_hash_ignores_column_order_and_is_content_sensitive():
    frame = _frame()
    permuted = frame[list(reversed(frame.columns))]
    assert sha256_frame(frame) == sha256_frame(permuted)

    changed = frame.copy()
    changed.loc[0, "log_D"] = changed.loc[0, "log_D"] + 1e-6
    assert sha256_frame(frame) != sha256_frame(changed)


def test_frame_hash_survives_a_parquet_round_trip(tmp_path):
    frame = _frame()
    path = tmp_path / "f.parquet"
    frame.to_parquet(path)
    assert sha256_frame(pd.read_parquet(path)) == sha256_frame(frame)


def test_frame_hash_row_order_invariance_is_opt_in():
    frame = _frame()
    shuffled = frame.sample(frac=1.0, random_state=3)
    assert sha256_frame(frame) != sha256_frame(shuffled)
    assert sha256_frame(frame, sort_rows_by=["row_id"]) == sha256_frame(shuffled, sort_rows_by=["row_id"])


def test_frame_hash_rejects_unknown_sort_columns():
    with pytest.raises(KeyError):
        sha256_frame(_frame(), sort_rows_by=["nope"])


def test_file_hash_matches_hashlib(tmp_path):
    import hashlib
    path = tmp_path / "x.bin"
    path.write_bytes(b"lanthanide" * 1000)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_code_hashes_marks_missing_paths(tmp_path):
    present = tmp_path / "a.py"
    present.write_text("x = 1\n")
    out = code_hashes([present, tmp_path / "absent.py"])
    assert out[str(present)] != "MISSING"
    assert out[str(tmp_path / "absent.py")] == "MISSING"


def test_code_hashes_walks_directories_and_skips_pycache(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "m.py").write_text("y = 2\n")
    (pkg / "__pycache__" / "m.cpython.py").write_text("cached\n")
    out = code_hashes([pkg])
    assert list(out) == [str(pkg / "m.py")]


def test_json_hash_is_key_order_independent():
    assert sha256_json({"a": 1, "b": [1, 2]}) == sha256_json({"b": [1, 2], "a": 1})


def test_feature_registry_hash_is_column_order_sensitive():
    """max_features samples columns, so a permuted column list is a different model."""
    a = RunManifest(layer="l", run_id="r").record_features(feature_sets={"MC": ["x", "y"]})
    b = RunManifest(layer="l", run_id="r").record_features(feature_sets={"MC": ["y", "x"]})
    assert a.entries["feature_registry_sha256"] != b.entries["feature_registry_sha256"]


# --------------------------------------------------------------------------- #
# Fail-closed behaviour
# --------------------------------------------------------------------------- #

def test_missing_required_keys_are_reported():
    manifest = RunManifest(layer="gen6", run_id="r")
    assert set(manifest.missing_required()) == set(REQUIRED_MANIFEST_KEYS)


def test_complete_manifest_has_no_missing_keys(tmp_path):
    assert _complete_manifest(tmp_path).missing_required() == []


def test_split_record_rejects_an_incomplete_fold():
    manifest = RunManifest(layer="gen6", run_id="r")
    broken = _fold_record()
    del broken["test_superclusters"]
    with pytest.raises(ValueError, match="missing required keys"):
        manifest.record_split(definition={}, folds=[broken])
    assert set(REQUIRED_FOLD_KEYS) >= {"test_row_ids_sha256", "test_extractants"}


def test_validate_run_fails_on_missing_artifact(tmp_path):
    manifest = _complete_manifest(tmp_path).write(tmp_path)
    record = validate_run(tmp_path, manifest=manifest, required_artifacts=["absent.csv"])
    assert record["ok"] is False
    assert record["missing_artifacts"] == ["absent.csv"]


def test_validate_run_fails_on_empty_artifact(tmp_path):
    (tmp_path / "empty.csv").write_text("")
    manifest = _complete_manifest(tmp_path).write(tmp_path)
    record = validate_run(tmp_path, manifest=manifest, required_artifacts=["empty.csv"])
    assert record["ok"] is False
    assert record["empty_artifacts"] == ["empty.csv"]


def test_validate_run_fails_on_a_failed_check(tmp_path):
    manifest = _complete_manifest(tmp_path).write(tmp_path)
    record = validate_run(tmp_path, manifest=manifest,
                          checks={"identical_test_rows": True,
                                  "no_leakage": {"ok": False, "detail": "cluster tan007 on both sides"}})
    assert record["ok"] is False
    assert record["failed_checks"] == ["no_leakage"]


def test_validate_run_fails_on_missing_manifest_key(tmp_path):
    manifest = _complete_manifest(tmp_path)
    del manifest.entries["model_seed"]
    payload = manifest.write(tmp_path)
    record = validate_run(tmp_path, manifest=payload)
    assert record["ok"] is False
    assert record["missing_manifest_keys"] == ["model_seed"]


def test_success_marker_only_on_a_valid_run(tmp_path):
    manifest = _complete_manifest(tmp_path).write(tmp_path)
    (tmp_path / "result.csv").write_text("a,b\n1,2\n")
    good = validate_run(tmp_path, manifest=manifest, required_artifacts=["result.csv"],
                        checks={"identical_test_rows": True})
    assert good["ok"] is True
    marker = write_success(tmp_path, manifest=manifest, validation=good)
    assert marker is not None and marker.name == "_SUCCESS.json"
    payload = json.loads(marker.read_text())
    assert payload["validation_ok"] is True
    assert payload["dataset_file_sha256"] == manifest["dataset_file_sha256"]
    assert not (tmp_path / "_FAILED.json").exists()


def test_failed_run_writes_a_failure_marker_instead(tmp_path):
    manifest = _complete_manifest(tmp_path).write(tmp_path)
    bad = validate_run(tmp_path, manifest=manifest, required_artifacts=["nope.csv"])
    assert write_success(tmp_path, manifest=manifest, validation=bad) is None
    assert (tmp_path / "_FAILED.json").exists()
    assert not (tmp_path / "_SUCCESS.json").exists()


# --------------------------------------------------------------------------- #
# Written artifacts
# --------------------------------------------------------------------------- #

def test_write_emits_manifest_and_artifact_hashes(tmp_path):
    (tmp_path / "table.csv").write_text("x\n1\n")
    payload = _complete_manifest(tmp_path).write(tmp_path)
    assert (tmp_path / "manifest.json").is_file()
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text())
    assert "table.csv" in hashes["files"]
    assert hashes["files"]["table.csv"]["bytes"] > 0
    # the hash file must not try to hash itself
    assert "artifact_hashes.json" not in hashes["files"]
    assert payload["layer"] == "gen6_diversity"
    assert json.loads((tmp_path / "manifest.json").read_text())["run_id"] == "test_run"


def test_artifact_hashes_are_content_sensitive(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    first = artifact_hashes(tmp_path)
    (tmp_path / "a.txt").write_text("two")
    assert artifact_hashes(tmp_path)["files"]["a.txt"]["sha256"] != first["files"]["a.txt"]["sha256"]


def test_manifest_json_serialises_numpy(tmp_path):
    manifest = _complete_manifest(tmp_path)
    manifest.record("numpy_scalars", {"i": np.int64(3), "f": np.float32(1.5), "b": np.bool_(True),
                                      "arr": np.arange(3)})
    payload = manifest.write(tmp_path)
    restored = json.loads((tmp_path / "manifest.json").read_text())
    assert restored["numpy_scalars"] == {"i": 3, "f": 1.5, "b": True, "arr": [0, 1, 2]}
    assert payload["numpy_scalars"]["arr"].tolist() == [0, 1, 2]


def test_validate_run_rejects_non_boolean_check_values(tmp_path):
    """`bool("FAIL")` is True and `bool(0.0001)` is True — a check must be a real bool."""
    manifest = _complete_manifest(tmp_path).write(tmp_path)
    for bad in ("FAIL", 0.0001, [None], {"ok": "false"}, {"ok": 1}):
        record = validate_run(tmp_path, manifest=manifest, checks={"h1": bad})
        assert record["ok"] is False, bad
        assert record["failed_checks"] == ["h1"], bad
    good = validate_run(tmp_path, manifest=manifest, checks={"h1": True, "h2": {"ok": True}})
    assert good["ok"] is True


def test_frame_hash_refuses_duplicate_column_labels():
    """With duplicate labels `frame[col]` is a DataFrame and the hash would describe
    the schema, not the data. Fail closed rather than hash nothing."""
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate column labels"):
        sha256_frame(frame)


def test_artifact_hashes_cover_validation_and_late_writes(tmp_path):
    """artifact_hashes.json used to be written before validation.json existed."""
    manifest = _complete_manifest(tmp_path).write(tmp_path)
    (tmp_path / "result.csv").write_text("a\n1\n")
    validation = validate_run(tmp_path, manifest=manifest, required_artifacts=["result.csv"],
                              checks={"ok_check": True})
    write_success(tmp_path, manifest=manifest, validation=validation)
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text())
    assert "validation.json" in hashes["files"]
    assert "_SUCCESS.json" in hashes["files"]
    assert "result.csv" in hashes["files"]
    assert "artifact_hashes.json" not in hashes["files"]


def test_code_hashes_flag_files_modified_after_the_run_started(tmp_path):
    """A file edited mid-run would otherwise be certified in a state nothing executed."""
    import time
    module = tmp_path / "m.py"
    module.write_text("x = 1\n")
    manifest = RunManifest(layer="gen6", run_id="r")
    manifest.record_code([module])
    assert manifest.entries["code_modified_after_start"] == []
    time.sleep(0.01)
    module.write_text("x = 2\n")
    later = RunManifest(layer="gen6", run_id="r")
    later.created_utc = "2000-01-01T00:00:00+00:00"      # a run that started long ago
    later.record_code([module])
    assert later.entries["code_modified_after_start"] == [str(module)]
    assert "code_hash_warning" in later.entries
