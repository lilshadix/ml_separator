"""Tests for provenance reconstruction.

Two paths must both be right: with the upstream tables (publication identity is
real) and without them (publication identity is *unavailable* and must never be
invented). The tests below do not require the sibling dataset-builder project;
the one test that does is skipped when it is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen6.provenance import (
    DEFAULT_UPSTREAM_DIRECTORIES, PROVENANCE_STATUS, SAFE_DATABASE_SELF_CITATION,
    canonical_publication, find_upstream_directory, load_upstream_tables,
    provenance_audit_report, reconstruct_provenance,
)


def _source(n_extractants: int = 3, n_conditions: int = 3, n_metals: int = 2) -> pd.DataFrame:
    rows = []
    counter = 1000
    for i in range(n_extractants):
        for c in range(n_conditions):
            for m in range(n_metals):
                counter += 1
                rows.append({
                    "safe_exp_id": f"Eu_SAFE:{counter}",
                    "canonical_smiles": f"SMILES{i}",
                    "metal_symbol": ["Nd", "Eu"][m],
                    "log_D": float(i + 0.1 * c + 0.01 * m),
                    "cond__acid_concentration_M": float(c),
                    "cond__diluent__kerosene": 1.0,
                    "cond__diluent__toluene": 0.0,
                })
    return pd.DataFrame(rows)


def _upstream(source: pd.DataFrame, *, doi_of=lambda i: "10.1000/paper-a") -> pd.DataFrame:
    return pd.DataFrame({
        "safe_exp_id": source["safe_exp_id"],
        "exp_id": range(len(source)),
        "DOI": [f"https://doi.org/{doi_of(i)}, https://doi.org/{SAFE_DATABASE_SELF_CITATION}"
                for i in range(len(source))],
        "entry_author": ["A Curator"] * len(source),
        "addition_date": ["2025-01-01"] * len(source),
        "upstream_source_file": ["Eu_SAFE.csv"] * len(source),
        "comments_description": [f"Fig {i % 3}" for i in range(len(source))],
        "Solvent_Name": ["kerosene"] * len(source),
    })


# --------------------------------------------------------------------------- #
# Reference canonicalisation
# --------------------------------------------------------------------------- #

def test_self_citation_is_removed_and_notation_is_normalised():
    a, refs_a = canonical_publication(
        f"https://doi.org/10.1000/x, https://doi.org/{SAFE_DATABASE_SELF_CITATION}")
    b, refs_b = canonical_publication("10.1000/X")
    assert a == b and refs_a == refs_b == ("10.1000/x",)


def test_a_row_citing_only_the_database_has_no_identifiable_study():
    identifier, references = canonical_publication(f"https://doi.org/{SAFE_DATABASE_SELF_CITATION}")
    assert identifier is None and references == ()


def test_missing_reference_is_none_not_an_exception():
    assert canonical_publication(None) == (None, ())
    assert canonical_publication(float("nan")) == (None, ())


def test_reference_order_does_not_change_the_identifier():
    a, _ = canonical_publication("10.1/a, 10.2/b")
    b, _ = canonical_publication("10.2/b, 10.1/a")
    assert a == b


def test_distinct_papers_get_distinct_identifiers():
    a, _ = canonical_publication("10.1/a")
    b, _ = canonical_publication("10.1/b")
    assert a != b and a.startswith("pub_") and b.startswith("pub_")


# --------------------------------------------------------------------------- #
# Without upstream: honest unavailability
# --------------------------------------------------------------------------- #

def test_without_upstream_publication_is_unavailable_not_guessed():
    audit = reconstruct_provenance(_source(), upstream=None,
                                   upstream_directory="/nonexistent/path")
    a = audit.audit
    assert a["publication_id_status"] == "unavailable"
    assert a["n_publications"] is None
    assert a["fraction_ambiguous_provenance"] == 1.0
    assert "ARI" in a["publication_id_unavailable_reason"]
    assert audit.table["publication_id"].isna().all()
    # and nothing downstream pretends otherwise
    assert a["merged_across_publication_boundaries"]["level_cell"]["status"] == "unavailable"


def test_without_upstream_series_is_labelled_a_surrogate():
    audit = reconstruct_provenance(_source(), upstream_directory="/nonexistent/path")
    assert audit.audit["experiment_series_id_status"] == "surrogate"
    assert "SURROGATE" in audit.audit["experiment_series_id_definition"]
    assert set(audit.table["experiment_series_id_status"]) == {"surrogate"}


def test_without_upstream_strict_equals_legacy():
    audit = reconstruct_provenance(_source(), upstream_directory="/nonexistent/path")
    assert audit.audit["provenance_strict_is_legacy"] is True
    assert audit.table["provenance_strict"].all()
    assert audit.audit["provenance_strict_rows_dropped"] == 0


def test_experiment_id_is_always_available():
    audit = reconstruct_provenance(_source(), upstream_directory="/nonexistent/path")
    assert audit.audit["experiment_id_status"] == "reconstructed"
    assert audit.audit["n_experiment_ids"] == audit.audit["n_rows"]


def test_source_without_safe_exp_id_is_rejected():
    with pytest.raises(KeyError, match="safe_exp_id"):
        reconstruct_provenance(_source().drop(columns=["safe_exp_id"]))


# --------------------------------------------------------------------------- #
# With upstream
# --------------------------------------------------------------------------- #

def test_with_upstream_publication_is_reconstructed():
    source = _source()
    audit = reconstruct_provenance(source, upstream=_upstream(source))
    a = audit.audit
    assert a["publication_id_status"] == "reconstructed"
    assert a["upstream_join_fraction"] == pytest.approx(1.0)
    assert a["n_publications"] == 1
    assert a["fraction_ambiguous_provenance"] == 0.0
    assert set(audit.table["publication_id_status"]) == {"reconstructed"}


def test_two_papers_are_two_publications_and_purity_is_measured():
    source = _source(n_extractants=2, n_conditions=2, n_metals=2)
    # first ligand from paper A, second from paper B
    upstream = _upstream(source)
    upstream.loc[4:, "DOI"] = "https://doi.org/10.1000/paper-b"
    audit = reconstruct_provenance(source, upstream=upstream)
    assert audit.audit["n_publications"] == 2
    purity = audit.audit["merged_across_publication_boundaries"]["extractant"]
    assert purity["n_groups_multi_publication"] == 0     # each ligand sits in one paper


def test_a_cell_spanning_two_publications_is_counted():
    """Two studies measuring the same ligand/condition/metal are a merged average."""
    source = _source(n_extractants=1, n_conditions=1, n_metals=1)
    source = pd.concat([source, source.assign(safe_exp_id="Eu_SAFE:9999")], ignore_index=True)
    upstream = _upstream(source)
    upstream.loc[1, "DOI"] = "https://doi.org/10.1000/paper-b"
    audit = reconstruct_provenance(source, upstream=upstream)
    cell = audit.audit["merged_across_publication_boundaries"]["level_cell"]
    assert cell["n_groups_multi_publication"] == 1
    assert cell["n_rows_in_multi_publication_groups"] == 2
    assert audit.audit["fraction_merged_across_publication_boundaries"] == pytest.approx(1.0)
    assert audit.audit["provenance_strict_is_legacy"] is False


def test_repeated_cells_with_a_hidden_axis_are_not_called_replicates():
    source = _source(n_extractants=1, n_conditions=1, n_metals=1)
    source = pd.concat([source, source.assign(safe_exp_id="Eu_SAFE:8888", log_D=3.0)],
                       ignore_index=True)
    upstream = _upstream(source)
    upstream.loc[1, "Solvent_Name"] = "toluene"        # the hidden axis
    audit = reconstruct_provenance(source, upstream=upstream)
    a = audit.audit
    assert a["replicate_id_status"] == "surrogate"
    assert a["n_cells_repeated"] == 1
    assert a["repeated_cells_with_a_hidden_axis"] == 1
    assert a["repeated_cells_that_look_like_true_replicates"] == 0
    assert a["hidden_axes_inside_repeated_cells"]["Solvent_Name"]["cells_varying"] == 1
    assert a["repeated_cell_log_d_range"]["max"] == pytest.approx(3.0)


def test_status_values_are_from_the_declared_vocabulary():
    source = _source()
    audit = reconstruct_provenance(source, upstream=_upstream(source))
    for column in ("experiment_id_status", "publication_id_status",
                   "experiment_series_id_status", "replicate_id_status"):
        assert set(audit.table[column]) <= set(PROVENANCE_STATUS)


def test_audit_carries_every_field_the_protocol_requires():
    source = _source()
    audit = reconstruct_provenance(source, upstream=_upstream(source)).audit
    for key in ("n_publications", "n_series", "n_cells", "rows_per_publication",
                "extractants_per_publication", "conditions_per_series", "metals_per_series",
                "fraction_ambiguous_provenance", "fraction_merged_across_publication_boundaries"):
        assert key in audit, key


def test_state_and_json_round_trip(tmp_path):
    source = _source()
    audit = reconstruct_provenance(source, upstream=_upstream(source))
    state = audit.state
    assert state["publication_status"] == "reconstructed"
    path = audit.to_json(tmp_path / "provenance_audit.json")
    restored = json.loads(path.read_text())
    assert restored["n_publications"] == audit.audit["n_publications"]


def test_report_is_readable_and_names_the_status():
    source = _source()
    text = provenance_audit_report(reconstruct_provenance(source, upstream=_upstream(source)))
    assert "# Provenance audit" in text
    assert "publication_id" in text and "reconstructed" in text
    assert "Dataset variants" in text


def test_report_works_when_provenance_is_unavailable():
    text = provenance_audit_report(
        reconstruct_provenance(_source(), upstream_directory="/nonexistent/path"))
    assert "NOT REACHABLE" in text
    assert "unavailable" in text


# --------------------------------------------------------------------------- #
# Integration with the real upstream tables (skipped when absent)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(find_upstream_directory() is None,
                    reason="the sibling lanthanide_dataset_builder checkout is not present")
def test_real_upstream_tables_join_and_carry_dois():
    directory = find_upstream_directory()
    raw = load_upstream_tables(directory)
    assert "DOI" in raw.columns
    assert raw["safe_exp_id"].str.contains("_SAFE:").all()
    assert raw["safe_exp_id"].is_unique
