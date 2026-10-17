"""LT-09: the cohort equals the frozen manifest (521 cells), NaN-aware bench.Y equals the frame's logD__ columns,
and the bundle / cohort-parquet digests equal the pinned values (IMPLEMENTATION_PLAN.md §1.1).  Only NaN masks
and equality assertions touch the target."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gen201 import bootstrap
from gen201 import data as D
from gen20k import data as GD


def test_cohort_assertions_pass(bench):
    rec = D.assert_cohort(bench)
    assert rec["n_cells"] == 521 and rec["n_entries"] == 3359
    assert rec["Y_equals_frame_logD"] is True
    assert rec["bundle_sha256"] == D.BUNDLE_SHA256 == "fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd"
    assert rec["cohort_parquet_sha256"] == D.COHORT_PARQUET_SHA256 == \
        "3f11ac4ccb7c9f6b1fa768f4406e1f249881cfd76689ba49a9115efc08bb7fd6"
    assert len(rec["bench_pkl_sha256"]) == 64 and rec["bench_pkl_bytes"] == bootstrap.BENCH_PKL.stat().st_size
    assert rec["assert_matches_manifest"]["n_cells"] == 521


def test_identity_counts(bench):
    f = bench.frame
    assert len(f) == 521 and f["cell_id"].is_unique
    assert f["extractant"].nunique() == 90 and f["chemotype"].nunique() == 45 and f["publication_id"].nunique() == 58
    assert int(D.observed(bench).sum()) == 3359
    assert (D.observed(bench).sum(1) >= 2).all()


def test_y_equality_assertion_catches_a_mask_change(bench):
    """Dropping one measured entry from bench.Y (a NaN-mask change, no value read) must fail the assertion."""
    Y = np.array(bench.Y, dtype=float, copy=True)
    ci, im = map(int, np.argwhere(~np.isnan(Y))[0])
    Y[ci, im] = np.nan
    with pytest.raises(AssertionError):
        D.assert_cohort(replace(bench, Y=Y))


def test_y_equality_assertion_catches_an_added_entry(bench):
    Y = np.array(bench.Y, dtype=float, copy=True)
    ci, im = map(int, np.argwhere(np.isnan(Y))[0])
    Y[ci, im] = 0.0
    with pytest.raises(AssertionError):
        D.assert_cohort(replace(bench, Y=Y))


def test_manifest_assertion_catches_a_frame_change(bench):
    f = bench.frame.copy()
    f.loc[0, "publication_id"] = "pub_tampered"
    with pytest.raises(AssertionError):
        GD.assert_matches_manifest(f)


def test_load_record_is_filled(bench):
    assert D.LOAD_RECORD.get("n_cells") == 521
    assert D.LOAD_RECORD.get("fingerprint_gen13_recorded") == "4c3c6628ea0be949"
