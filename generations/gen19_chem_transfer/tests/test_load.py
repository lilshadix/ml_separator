"""Tests for the orchestrator's loader ``gen19ct.data.load`` (read-only use; the loader is not edited here).

Synthetic tests pin the publication / study key rules; archive tests pin the tier rule and the claim that
the g19 publication partition reproduces gen6/gen18's on the frozen bundle rows.
"""
from __future__ import annotations

import hashlib
import re
import time

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.data import load as ld

SELF = ld.SAFE_DATABASE_SELF_CITATION


# --------------------------------------------------------------------------------------------- #
# canonical_publication
# --------------------------------------------------------------------------------------------- #

def test_canonical_publication_strips_prefixes_and_self_citation():
    pid, refs = ld.canonical_publication(["https://doi.org/10.1021/ABC.123", SELF])
    pid2, refs2 = ld.canonical_publication(["10.1021/abc.123"])
    assert refs == ("10.1021/abc.123",)
    assert (pid, refs) == (pid2, refs2)


@pytest.mark.parametrize("raw", ["https://dx.doi.org/10.1/x", "http://www.doi.org/10.1/x", "doi:10.1/x",
                                 "DOI.ORG/10.1/X", "10.1/x.", "10.1/x/", "'10.1/x'", " 10.1/x ; "])
def test_canonical_publication_prefix_and_punctuation_variants(raw):
    assert ld.canonical_publication([raw])[1] == ("10.1/x",)


def test_canonical_publication_is_order_invariant_and_deduplicated():
    a = ld.canonical_publication(["10.2/b", "10.1/a"])
    b = ld.canonical_publication(["10.1/a", "10.2/b", "10.1/a"])
    c = ld.canonical_publication(["10.1/a; 10.2/b"])
    assert a == b == c
    assert a[1] == ("10.1/a", "10.2/b")
    assert a[0] == "pub_" + hashlib.sha1("10.1/a|10.2/b".encode()).hexdigest()[:10]


def test_canonical_publication_extracts_embedded_doi():
    _, refs = ld.canonical_publication(["see https://pubs.acs.org/doi/10.1021/ic5001234 for details"])
    assert refs == ("10.1021/ic5001234",)


@pytest.mark.parametrize("refs", [[], [SELF], ["https://doi.org/" + SELF], ["", "  "]])
def test_canonical_publication_nothing_left(refs):
    assert ld.canonical_publication(refs) == (None, ())


# --------------------------------------------------------------------------------------------- #
# publication_key / study_key
# --------------------------------------------------------------------------------------------- #

CORDIS = "https://cordis.europa.eu/project/id/211267"


def test_publication_key_doi_wins_over_reports():
    pid, status, refs = ld.publication_key(np.array(["10.1/a", SELF], dtype=object), np.array([CORDIS]), "./ST1.json")
    assert status == "DOI" and refs == ("10.1/a",)
    assert pid == ld.canonical_publication(["10.1/a"])[0]


def test_publication_key_report_subsource_groups_all_studies_of_one_project():
    p1 = ld.publication_key(np.array([SELF], dtype=object), np.array([CORDIS], dtype=object), "./ST1.json")
    p2 = ld.publication_key(np.array([], dtype=object), np.array([CORDIS], dtype=object), "./ST99.json")
    assert p1[1] == p2[1] == "REPORT_SUBSOURCE"
    assert p1[0] == p2[0]


@pytest.mark.parametrize("sub", [None, float("nan")])
def test_publication_key_report_without_subsource(sub):
    pid, status, refs = ld.publication_key(np.array([], dtype=object), np.array([CORDIS], dtype=object), sub)
    assert status == "REPORT"
    # keyed over the references only, so the same project gets the same id with or without sub-source file
    assert pid == ld.publication_key(np.array([], dtype=object), np.array([CORDIS], dtype=object), "./ST1.json")[0]


@pytest.mark.parametrize("doi_all,refs", [(np.array([], dtype=object), np.array([], dtype=object)),
                                          (np.array([SELF], dtype=object), np.array(["  "], dtype=object)),
                                          (None, None), (float("nan"), float("nan"))])
def test_publication_key_unresolved(doi_all, refs):
    assert ld.publication_key(doi_all, refs, "./ST1.json") == (ld.UNRESOLVED_PUBLICATION, "UNRESOLVED", ())


def test_study_key_rules():
    assert ld.study_key("pub_x", None) == "pub_x"
    assert ld.study_key("pub_x", float("nan")) == "pub_x"
    s1, s2 = ld.study_key("pub_x", "./ST1.json"), ld.study_key("pub_x", "./ST2.json")
    assert s1.startswith("stu_") and s2.startswith("stu_") and s1 != s2
    assert s1 == ld.study_key("pub_x", "./ST1.json")
    assert s1 != ld.study_key("pub_y", "./ST1.json")
    assert s1 == "stu_" + hashlib.sha1("pub_x@@./ST1.json".encode()).hexdigest()[:10]


# --------------------------------------------------------------------------------------------- #
# the archive
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def archive() -> pd.DataFrame:
    return ld.load_archive(copy=False)


def test_archive_digest_is_pinned():
    assert ld.assert_archive_unchanged() == paths.ARCHIVE_MASTER_SHA256


def test_loader_adds_columns_without_dropping_rows(archive):
    raw = pd.read_parquet(paths.ARCHIVE_MASTER, columns=["canonical_measurement_id"])
    assert len(archive) == len(raw)
    assert archive["canonical_measurement_id"].tolist() == raw["canonical_measurement_id"].tolist()
    for c in ("g19_publication_id", "g19_publication_status", "g19_publication_refs", "g19_study_id", "g19_metal",
              "g19_ox", "g19_metal_state", "g19_tier", "g19_bundle_exp_id"):
        assert c in archive.columns
    assert set(archive["g19_publication_status"]) <= set(ld.PUBLICATION_STATUSES)
    assert set(archive["g19_tier"]) == {"MODEL", "TARGET_ONLY", "NO_TARGET"}


def test_tier_rule(archive):
    logd = pd.to_numeric(archive["log_D"], errors="coerce")
    has_target = logd.notna() & np.isfinite(logd) & archive["metal_symbol"].notna()
    model = (has_target & archive["is_canonical_row"].astype(bool)
             & archive["model_readiness"].isin(["A_model_ready", "B_usable_with_caveats"]))
    expected = np.where(model, "MODEL", np.where(has_target, "TARGET_ONLY", "NO_TARGET"))
    assert (archive["g19_tier"].to_numpy() == expected).all()
    rows = ld.load_model_rows(copy=False)
    assert len(rows) == int(model.sum())
    assert (rows["g19_tier"] == "MODEL").all()
    assert rows.index.equals(archive.index[model.to_numpy()])


def test_load_returns_copies_by_default(archive):
    a = ld.load_archive()
    a.loc[a.index[0], "g19_tier"] = "MUTATED"
    m = ld.load_model_rows()
    m.loc[m.index[0], "g19_tier"] = "MUTATED"
    assert "MUTATED" not in set(ld.load_archive(copy=False)["g19_tier"])


def test_metal_state_label(archive):
    st = archive["g19_metal_state"]
    known = st.notna()
    assert st[known].map(lambda s: re.fullmatch(r"[A-Z][a-z]?\((?:I|II|III|IV|V|VI|VII)\)", s) is not None).all()
    expected_known = archive["metal_symbol"].notna() & archive["metal_oxidation_state"].notna()
    assert (known == expected_known).all()
    u6 = archive[(archive["metal_symbol"] == "U") & (archive["metal_oxidation_state"] == 6)]
    assert len(u6) and (u6["g19_metal_state"] == "U(VI)").all()


def test_publication_columns_reproduce_publication_key(archive):
    sample = archive.iloc[:: 97]
    for (_, r) in sample.iterrows():
        pid, status, refs = ld.publication_key(r["doi_all"], r["reference_other"], r["sub_source_file"])
        assert (pid, status, " | ".join(refs)) == (r["g19_publication_id"], r["g19_publication_status"],
                                                   r["g19_publication_refs"])
        assert r["g19_study_id"] == ld.study_key(pid, r["sub_source_file"])


def test_publication_partition_equals_gen6_on_bundle_rows(archive):
    t0 = time.perf_counter()
    g6 = ld.bundle_publication_map()
    assert len(g6) == g6["safe_exp_id"].nunique()
    j = g6.merge(archive[["g19_bundle_exp_id", "g19_publication_id", "metal_symbol"]], left_on="exp_id",
                 right_on="g19_bundle_exp_id", how="left", suffixes=("_gen6", ""), validate="one_to_one")
    assert j["g19_publication_id"].notna().all(), "every bundle row joins an archive record"
    assert (j["metal_symbol_gen6"] == j["metal_symbol"]).all()
    # no gen6 publication split across two g19 ids, and no g19 id merging two gen6 ids
    assert j.groupby("publication_id")["g19_publication_id"].nunique().max() == 1
    assert j.groupby("g19_publication_id")["publication_id"].nunique().max() == 1
    assert j["publication_id"].nunique() == j["g19_publication_id"].nunique()
    # ids are mostly literally identical; the exceptions are relabellings, not re-partitions
    assert (j["publication_id"] == j["g19_publication_id"]).mean() > 0.99
    assert time.perf_counter() - t0 < 10.0
