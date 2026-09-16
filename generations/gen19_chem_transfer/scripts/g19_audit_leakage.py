"""``g19_audit_leakage.py`` -- provenance completeness and the automated leakage audit (brief 1.4, 12, 27).

Runs every detector in :mod:`gen19ct.data.leakage` and the field registry in
:mod:`gen19ct.data.provenance` on the full archive and on ``g19_tier == "MODEL"`` rows.  Nothing is
trained or fitted.  Outputs (all in ``generations/gen19_chem_transfer/data_audit/``):

* ``metadata_availability.csv``          brief 1.4 fields x status x fill (all / MODEL / MODEL by metal
                                          group), free-text regex scan with examples, D_raw precision
* ``leakage_exact_duplicates.csv``       archive duplicate groups with > 1 row (checks in the summary)
* ``leakage_near_duplicates.csv``        pairs sharing the near-duplicate key at 6 and/or 3 s.f.
* ``leakage_cross_publication_copies.csv`` near-duplicate pairs, |delta log D| <= 0.005, different publications
* ``leakage_double_digitisation.csv``    same publication, 3 s.f. key, |delta log D| <= 0.02, not one A/B group
* ``leakage_doi_multiplicity.csv``       DOI / report / title multiplicity across fold groups and contexts
* ``leakage_publication_components.csv`` per publication: merged fold group under cumulative link rules
* ``leakage_metal_alias_risk.csv``       element vs element+state labels, export file vs metal
* ``leakage_metadata_target.csv``        target in text, provenance group statistics, serial correlation
* ``leakage_bundle_overlap.csv``         frozen lanthanide bundle rows vs their archive records
* ``leakage_summary.json``               counts of every class over all rows and MODEL rows

Run from the repository root::

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_audit_leakage.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.data import leakage as L  # noqa: E402
from gen19ct.data import provenance as P  # noqa: E402
from gen19ct.data.load import bundle_publication_map, load_archive  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json  # noqa: E402

NAME = "g19_audit_leakage"
BUNDLE_COLUMNS = ["safe_exp_id", "metal_symbol", "metal_ox", "log_D", "canonical_smiles", "extractant_name",
                  "split", "cond__acid_concentration_M", "cond__extractant_concentration_M", "cond__temperature_C"]
MAX_CSV_BYTES = 20_000_000


def _vc(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}


def _cap(df: pd.DataFrame, max_rows: int, name: str, notes: dict[str, Any]) -> pd.DataFrame:
    """Keep the first ``max_rows`` rows (callers sort by importance first); record any truncation."""
    notes[name] = {"n_rows_total": int(len(df)), "n_rows_written": int(min(len(df), max_rows)),
                   "truncated": bool(len(df) > max_rows), "max_rows": int(max_rows)}
    return df.head(max_rows)


def _pair_counts(p: pd.DataFrame, sig: int) -> dict[str, Any]:
    f = p[p[f"near_dup_sig{sig}"].astype(bool)]
    strict = f[f"strict_key_equal_sig{sig}"].astype("boolean").fillna(False).astype(bool)
    ids = set(f["id_a"]).union(f["id_b"])
    return {
        "n_pairs": int(len(f)),
        "n_rows_in_pairs": int(len(ids)),
        "n_keys_with_pairs": int(f[f"key_id_sig{sig}"].nunique()),
        "n_pairs_same_publication": int(f["same_publication"].sum()),
        "n_pairs_cross_publication": int((~f["same_publication"].astype(bool)).sum()),
        "n_pairs_within_one_AB_group": int(f["both_ab_group"].sum()),
        "n_pairs_log_D_both_finite": int(f["abs_delta_log_D"].notna().sum()),
        "n_pairs_delta_le_0.005": int(f["abs_delta_log_D"].le(0.005 + 1e-12).sum()),
        "n_pairs_delta_le_0.02": int(f["abs_delta_log_D"].le(0.02 + 1e-12).sum()),
        "n_pairs_delta_gt_0.3": int(f["abs_delta_log_D"].gt(0.3).sum()),
        "n_pairs_strict_key_equal": int(strict.sum()),
        "n_pairs_strict_key_equal_cross_publication": int((strict & ~f["same_publication"].astype(bool)).sum()),
        "n_publications_in_cross_publication_pairs": int(len(set(f.loc[~f["same_publication"].astype(bool), "pub_a"])
                                                             .union(f.loc[~f["same_publication"].astype(bool), "pub_b"]))),
    }


def _attach_context(pairs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Human triage columns for pair listings (metal state, extractant, conditions, source)."""
    ctx = df.set_index(L.ID_COL)
    loc = L.location_token(df)
    loc.index = df[L.ID_COL].to_numpy()
    out = pairs.copy()
    for side in ("a", "b"):
        ids = out[f"id_{side}"]
        out[f"metal_state_{side}"] = ids.map(L._state_label(df).set_axis(df[L.ID_COL].to_numpy()))
        out[f"extractant_name_{side}"] = ids.map(ctx["extractant_primary_name"])
        out[f"entry_author_{side}"] = ids.map(ctx["entry_author"])
        out[f"location_{side}"] = ids.map(loc)
        out[f"publication_refs_{side}"] = ids.map(ctx["g19_publication_refs"]).map(lambda s: L._short(s, 120))
    first = out["id_a"]
    out["acid_primary"] = first.map(ctx["acid_primary"])
    out["acid_M"] = first.map(ctx["acid_concentration_M"])
    out["extractant_M"] = first.map(ctx["extractant_primary_concentration_M"])
    out["temperature_C"] = first.map(ctx["temperature_C"])
    out["solvent_key"] = first.map(ctx["solvent_key"])
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=paths.DATA_AUDIT_DIR)
    ap.add_argument("--max-pair-rows", type=int, default=150_000,
                    help="cap on rows of each pair listing (truncation is recorded in the summary)")
    ap.add_argument("--seed", type=int, default=0, help="permutation seed of the target-in-text null")
    ns = ap.parse_args(argv)
    out_dir = paths.ensure_dir(ns.out_dir)
    trunc: dict[str, Any] = {}
    summary: dict[str, Any] = {"schema": "gen19.leakage_summary.v1", "script": NAME}

    with Run(NAME, args={"out_dir": paths.rel(out_dir), "max_pair_rows": ns.max_pair_rows, "seed": ns.seed},
             seed=ns.seed) as run:
        here = Path(__file__).resolve()
        run.inputs(paths.ARCHIVE_MASTER, paths.BUNDLE_DATASET, paths.GEN6_PROVENANCE, here,
                   paths.G19_ROOT / "gen19ct" / "data" / "leakage.py", paths.G19_ROOT / "gen19ct" / "data" / "provenance.py",
                   paths.G19_ROOT / "gen19ct" / "data" / "load.py")
        df = load_archive(copy=False)
        model = df["g19_tier"].eq("MODEL")
        mdf = df[model]
        summary["archive"] = {"n_rows": int(len(df)), "n_model_rows": int(model.sum()),
                              "sha256": paths.ARCHIVE_MASTER_SHA256, "tier_counts": _vc(df["g19_tier"]),
                              "n_publications_all": int(df[L.PUB_COL].nunique()),
                              "n_publications_model": int(mdf[L.PUB_COL].nunique())}
        outputs: list[Path] = []

        # ---- 1. provenance completeness ---------------------------------------------------------
        avail = P.availability_table(df, model)
        outputs.append(write_csv(avail, out_dir / "metadata_availability.csv"))
        fields = avail[avail["section"] == "field"]
        prim = fields[fields["source"] == "primary"]
        summary["metadata_availability"] = {
            "n_brief_fields": int(prim["brief_field"].nunique()),
            "status_counts_primary": _vc(prim["status"]),
            "fields_by_status": {s: sorted(prim.loc[prim["status"] == s, "brief_field"]) for s in P.STATUSES},
            "fill_model_primary": {r.brief_field: round(float(r.fill_model), 6) for r in prim.itertuples()},
            "text_scan_free_text_rows": {
                r.evidence_pattern: {"all": int(r.n_filled_all), "model": int(r.n_filled_model)}
                for r in avail[(avail["section"] == "text_scan")
                               & (avail["text_column"] == P.FREE_TEXT_COLUMN)].itertuples()},
        }

        # ---- 2. exact duplicates ----------------------------------------------------------------
        groups, checks = L.exact_duplicate_groups(df)
        outputs.append(write_csv(groups, out_dir / "leakage_exact_duplicates.csv"))
        mgroups, _ = L.exact_duplicate_groups(mdf)
        summary["exact_duplicates"] = {
            "checks": checks,
            "all_rows": {"n_groups_multi_row_by_class": _vc(groups["duplicate_class"]),
                         "n_rows_in_multi_row_groups_by_class":
                             {str(k): int(v) for k, v in groups.groupby("duplicate_class")["n_rows"].sum().items()},
                         "n_noncanonical_rows": int((~df["is_canonical_row"].astype(bool)).sum()),
                         "n_multi_row_groups_spanning_publications": int((groups["n_publications"] > 1).sum())},
            "model_rows": {"n_groups_multi_row_by_class": _vc(mgroups["duplicate_class"]),
                           "n_rows_in_multi_row_groups_by_class":
                               {str(k): int(v) for k, v in mgroups.groupby("duplicate_class")["n_rows"].sum().items()},
                           "n_multi_row_groups_spanning_publications": int((mgroups["n_publications"] > 1).sum()),
                           "n_multi_row_groups_spanning_publications_by_class":
                               {str(k): int((g["n_publications"] > 1).sum()) for k, g in mgroups.groupby("duplicate_class")},
                           "max_log_D_spread_in_group": float(mgroups["log_D_spread"].max()) if len(mgroups) else 0.0},
        }

        # ---- 3. near duplicates -----------------------------------------------------------------
        pairs = L.near_duplicate_pair_table(df, sigs=(6, 3))
        for s in (6, 3):
            pairs[f"strict_sig{s}"] = (pairs[f"near_dup_sig{s}"].astype(bool)
                                       & pairs[f"strict_key_equal_sig{s}"].astype("boolean").fillna(False).astype(bool))
        mpairs = pairs[pairs["both_model"]]
        summary["near_duplicates"] = {
            "key_fields_base": list(L.KEY_CATEGORICAL + L.KEY_NUMERIC),
            "key_fields_strict_extra": list(L.STRICT_CATEGORICAL + L.STRICT_NUMERIC),
            "na_matches_na": True,
            "all_rows": {f"sig{s}": _pair_counts(pairs, s) for s in (6, 3)},
            "model_rows": {f"sig{s}": _pair_counts(mpairs, s) for s in (6, 3)},
        }
        listing = pairs.assign(_x=~pairs["same_publication"].astype(bool))
        listing = listing.sort_values(["_x", "both_model", "abs_delta_log_D", "id_a", "id_b"],
                                      ascending=[False, False, True, True, True], kind="stable", na_position="last")
        listing = listing.drop(columns=["_x", "idx_a", "idx_b"])
        outputs.append(write_csv(_cap(listing, ns.max_pair_rows, "leakage_near_duplicates.csv", trunc),
                                 out_dir / "leakage_near_duplicates.csv"))

        # ---- 4. cross-publication copies --------------------------------------------------------
        copies = pairs[(~pairs["same_publication"].astype(bool))
                       & pairs["abs_delta_log_D"].le(L.COPY_TOLERANCE_LOG_D + 1e-12)].copy()
        copies = _attach_context(copies, df).sort_values(["pub_a", "pub_b", "id_a", "id_b"], kind="stable")
        pub_pairs = copies.assign(pp=[" & ".join(sorted((str(a), str(b)))) for a, b in zip(copies["pub_a"],
                                                                                           copies["pub_b"])])
        summary["cross_publication_copies"] = {
            "tolerance_log_D": L.COPY_TOLERANCE_LOG_D,
            **{f"{pop}_sig{s}": {
                "n_pairs": int(c[f"near_dup_sig{s}"].sum()),
                "n_pairs_strict_key_equal": int(c[f"strict_sig{s}"].sum()),
                "n_rows": int(len(set(c.loc[c[f"near_dup_sig{s}"], "id_a"]).union(c.loc[c[f"near_dup_sig{s}"], "id_b"]))),
                "n_publications": int(len(set(c.loc[c[f"near_dup_sig{s}"], "pub_a"]).union(
                    c.loc[c[f"near_dup_sig{s}"], "pub_b"]))),
                "n_publication_pairs": int(pub_pairs.loc[c.index][c[f"near_dup_sig{s}"]]["pp"].nunique()),
            } for pop, c in (("all_rows", copies), ("model_rows", copies[copies["both_model"]])) for s in (6, 3)},
            "top_publication_pairs_model_rows": {
                k: int(v) for k, v in pub_pairs[pub_pairs["both_model"]]["pp"].value_counts().head(10).items()},
        }
        outputs.append(write_csv(_cap(copies.drop(columns=["idx_a", "idx_b"]), ns.max_pair_rows,
                                      "leakage_cross_publication_copies.csv", trunc),
                                 out_dir / "leakage_cross_publication_copies.csv"))

        # ---- 5. double digitisation -------------------------------------------------------------
        p3 = L.near_duplicate_pairs(df, sig=3)
        dd = L.double_digitisation(df, sig=3, pairs=p3)
        dd["both_model"] = (dd["tier_a"] == "MODEL") & (dd["tier_b"] == "MODEL")
        mdd = dd[dd["both_model"]]
        tags = L.digitiser_tag(df)
        summary["double_digitisation"] = {
            "tolerance_log_D": L.DOUBLE_DIGITISATION_TOLERANCE_LOG_D, "sig_figs": 3,
            **{pop: {"n_pairs": int(len(x)), "n_rows": int(len(set(x["id_a"]).union(x["id_b"]))),
                     "n_publications": int(x["pub_a"].nunique()),
                     "by_location_relation": _vc(x["location_relation"]),
                     "n_pairs_strict_key_equal": int(x["strict_key_equal"].astype(bool).sum()),
                     "n_pairs_strict_key_equal_same_location": int((x["strict_key_equal"].astype(bool)
                                                                    & x["location_relation"].eq("SAME_LOCATION")).sum()),
                     "n_pairs_delta_exactly_0": int(x["abs_delta_log_D"].le(1e-12).sum()),
                     "n_pairs_delta_exactly_0_strict_key_equal": int((x["abs_delta_log_D"].le(1e-12)
                                                                      & x["strict_key_equal"].astype(bool)).sum()),
                     "n_rows_delta_exactly_0_strict_key_equal": int(len(
                         set(x.loc[x["abs_delta_log_D"].le(1e-12) & x["strict_key_equal"].astype(bool), "id_a"]).union(
                             x.loc[x["abs_delta_log_D"].le(1e-12) & x["strict_key_equal"].astype(bool), "id_b"]))),
                     "duplicate_class_pairs": {f"{a}|{b}": int(v) for (a, b), v in
                                               x.groupby(["duplicate_class_a", "duplicate_class_b"]).size().items()},
                     "n_pairs_both_canonical": int(x["both_canonical"].sum()),
                     "n_pairs_different_digitiser_tags": int((x["digitiser_a"].notna() & x["digitiser_b"].notna()
                                                              & (x["digitiser_a"] != x["digitiser_b"])).sum())}
               for pop, x in (("all_rows", dd), ("model_rows", mdd))},
            "explicit_digitiser_tag_rows": {"n_rows": int(tags.notna().sum()),
                                            "tier_counts": _vc(df.loc[tags.notna(), "g19_tier"]),
                                            "n_publications": int(df.loc[tags.notna(), L.PUB_COL].nunique())},
        }
        dd_out = dd.assign(_o=dd["location_relation"].map({"SAME_LOCATION": 0, "LOCATION_UNKNOWN": 1,
                                                           "DIFFERENT_LOCATION": 2}))
        dd_out = dd_out.sort_values(["both_model", "_o", "abs_delta_log_D", "id_a", "id_b"],
                                    ascending=[False, True, True, True, True], kind="stable")
        dd_out = dd_out.drop(columns=["_o", "idx_a", "idx_b"])
        outputs.append(write_csv(_cap(dd_out, ns.max_pair_rows, "leakage_double_digitisation.csv", trunc),
                                 out_dir / "leakage_double_digitisation.csv"))

        # ---- 6. DOI multiplicity and publication groups -----------------------------------------
        doi = L.doi_source_multiplicity(df)
        outputs.append(write_csv(doi, out_dir / "leakage_doi_multiplicity.csv"))
        risky = doi[doi["risk"] != "NONE"]
        corr = L.corrected_publication_id(df)
        summary["doi_multiplicity"] = {
            "risk_by_kind": {k: _vc(g["risk"]) for k, g in doi.groupby("kind")},
            "non_none": [{"kind": r.kind, "source": L._short(r.source, 120), "risk": r.risk, "n_rows": int(r.n_rows),
                          "n_model_rows": int(r.n_model_rows), "n_g19_publication_ids": int(r.n_g19_publication_ids),
                          "entry_authors": r.entry_authors} for r in risky.itertuples()],
            "n_g19_publication_ids": int(df[L.PUB_COL].nunique()),
            "n_corrected_publication_ids": int(corr.nunique()),
            "n_corrected_ids_merging_several_g19_ids": int((df.groupby(corr)[L.PUB_COL].nunique() > 1).sum()),
            "g19_ids_split_only_by_doi_spelling": sorted(
                df.loc[corr.isin(df.groupby(corr)[L.PUB_COL].nunique().loc[lambda s: s > 1].index), L.PUB_COL].unique()),
            "n_rows_in_g19_ids_split_only_by_doi_spelling": int(corr.isin(
                df.groupby(corr)[L.PUB_COL].nunique().loc[lambda s: s > 1].index).sum()),
            "n_model_rows_in_g19_ids_split_only_by_doi_spelling": int((corr.isin(
                df.groupby(corr)[L.PUB_COL].nunique().loc[lambda s: s > 1].index) & model).sum()),
        }
        mcopy6 = pairs[pairs["both_model"] & pairs["near_dup_sig6"] & ~pairs["same_publication"].astype(bool)
                       & pairs["abs_delta_log_D"].le(L.COPY_TOLERANCE_LOG_D + 1e-12)]
        mkey6 = pairs[pairs["both_model"] & pairs["near_dup_sig6"] & ~pairs["same_publication"].astype(bool)]
        comps = L.publication_link_components(df, copy_pairs=mcopy6, key_pairs=mkey6, duplicate_group_mask=model)
        outputs.append(write_csv(comps, out_dir / "leakage_publication_components.csv"))
        mpubs = comps[comps["n_model_rows"] > 0]
        summary["publication_groups"] = {
            "rules_cumulative": list(L.LINK_RULES),
            "copy_and_key_edges_population": "MODEL rows, sig6 key; copy = |delta log D| <= 0.005",
            "archive_duplicate_group_edges_population": "MODEL rows sharing a duplicate_group_id (any class)",
            **{rule: {"n_groups_all_publications": int(comps[f"group_{rule}"].nunique()),
                      "n_groups_with_model_rows": int(mpubs[f"group_{rule}"].nunique()),
                      "n_merged_groups": int(comps.loc[comps[f"n_publications_in_group_{rule}"] > 1,
                                                       f"group_{rule}"].nunique()),
                      "n_model_publications_in_merged_groups": int((mpubs[f"n_publications_in_group_{rule}"] > 1).sum()),
                      "largest_merged_group_n_publications": int(comps[f"n_publications_in_group_{rule}"].max()),
                      "largest_merged_group_n_model_rows": int(comps.loc[comps[f"n_publications_in_group_{rule}"] > 1,
                                                                         f"n_model_rows_in_group_{rule}"].max())
                      if (comps[f"n_publications_in_group_{rule}"] > 1).any() else 0,
                      "n_model_rows_in_merged_groups": int(mpubs.loc[mpubs[f"n_publications_in_group_{rule}"] > 1,
                                                                     "n_model_rows"].sum())}
               for rule in L.LINK_RULES},
            "largest_single_publication_n_model_rows": int(comps["n_model_rows"].max()),
        }

        # ---- 7. V1 burden of near duplicates (MODEL rows) ---------------------------------------
        criteria = {
            "key_sig6_any_value": ("near_dup_sig6", None), "key_sig3_any_value": ("near_dup_sig3", None),
            "key_sig6_delta_le_0.005": ("near_dup_sig6", 0.005), "key_sig3_delta_le_0.005": ("near_dup_sig3", 0.005),
            "key_sig3_delta_le_0.02": ("near_dup_sig3", 0.02),
            "strict_key_sig6_any_value": ("strict_sig6", None), "strict_key_sig6_delta_le_0.005": ("strict_sig6", 0.005),
        }
        summary["v1_near_duplicate_burden_model_rows"] = {
            name: L.cross_boundary_burden(mdf, mpairs, flag=flag, tol=tol, key_col="key_id_sig" + flag[-1])
            for name, (flag, tol) in criteria.items()}

        # ---- 8. metal aliases -------------------------------------------------------------------
        alias = L.metal_alias_split_risk(df)
        outputs.append(write_csv(alias, out_dir / "leakage_metal_alias_risk.csv"))
        el = alias[alias["scope"] == "element"]
        pe = alias[alias["scope"] == "publication_element"]
        ef = alias[alias["scope"] == "export_file"]
        summary["metal_alias"] = {
            "element_risk_counts": _vc(el["risk"]),
            "n_model_rows_unknown_state": int(el["n_model_rows_unknown_state"].sum()),
            "n_model_rows": int(el["n_model_rows"].sum()),
            "elements_with_unknown_and_known_state_model_rows": sorted(
                el.loc[(el["n_model_rows_unknown_state"] > 0) & (el["n_model_rows"] > el["n_model_rows_unknown_state"]),
                       "g19_metal"]),
            "publication_element_risk_counts": _vc(pe["risk"]),
            "n_publication_elements_same_conditions_across_labels_model": int(
                (pe["n_model_condition_keys_across_labels"] > 0).sum()),
            "n_model_rows_in_condition_keys_shared_across_state_labels": int(pe["n_model_rows_in_shared_condition_keys"].sum()),
            "n_rows_export_file_is_other_metal": int(ef["n_rows_file_is_other_metal"].sum()),
            "n_model_rows_export_file_is_other_metal": int(ef["n_model_rows_file_is_other_metal"].sum()),
        }

        # ---- 9. metadata -> target --------------------------------------------------------------
        meta_all = L.metadata_target_leak(df, seed=ns.seed, population="all")
        meta_model = pd.concat([L.provenance_group_predictability(mdf, population="model"),
                                L.serial_target_correlation(mdf, population="model")], ignore_index=True, sort=False)
        meta = pd.concat([meta_all, meta_model], ignore_index=True, sort=False)
        outputs.append(write_csv(meta, out_dir / "leakage_metadata_target.csv"))
        tit = meta[meta["check"] == "target_in_text"]
        pg = meta[meta["check"] == "provenance_group_predictability"]
        sc = meta[meta["check"] == "serial_target_correlation"]
        summary["metadata_target"] = {
            "target_in_text": {f"{r.grouping}:{r.match_type}": {"matched": int(r.n_rows_matched),
                                                                "matched_model": int(r.n_model_rows_matched),
                                                                "matched_permuted_null": int(r.n_rows_matched_permuted)}
                               for r in tit.itertuples()},
            "n_text_checks_above_permuted_null": int((tit["n_rows_matched"] > tit["n_rows_matched_permuted"]).sum()),
            "max_rows_matched_any_text_check": int(tit["n_rows_matched"].max()),
            **{f"group_predictability_{pop}": {
                "best_provenance_grouping": str(g.loc[g["role"] == "provenance"].sort_values(
                    "loo_r2", ascending=False).iloc[0]["grouping"]),
                "best_provenance_loo_r2": float(g.loc[g["role"] == "provenance", "loo_r2"].max()),
                "best_chemistry_reference_grouping": str(g.loc[g["role"] == "chemistry_reference"].sort_values(
                    "loo_r2", ascending=False).iloc[0]["grouping"]),
                "best_chemistry_reference_loo_r2": float(g.loc[g["role"] == "chemistry_reference", "loo_r2"].max()),
                "publication_loo_r2": float(g.loc[g["grouping"] == "g19_publication_id", "loo_r2"].iloc[0]),
                "n_provenance_groupings_exceeding_best_chemistry_by_0.02": int(
                    (g.loc[g["role"] == "provenance", "excess_over_best_chemistry"] > 0.02).sum()),
            } for pop, g in pg.groupby("population")},
            "serial_lag1_r": {f"{r.population}:{r.grouping}": {"r": float(r.lag1_pearson_r), "n": int(r.n_rows),
                                                               "same_cell_fraction": float(r.same_cell_fraction)}
                              for r in sc.itertuples()},
        }

        # ---- 10. bundle overlap -----------------------------------------------------------------
        bundle = pd.read_parquet(paths.BUNDLE_DATASET, columns=BUNDLE_COLUMNS)
        bo = L.bundle_overlap(bundle, df, gen6=bundle_publication_map())
        bo["bundle_split"] = bundle["split"].to_numpy()
        del bundle
        outputs.append(write_csv(bo, out_dir / "leakage_bundle_overlap.csv"))
        j = bo[bo["joined"]]
        g6 = j.dropna(subset=["gen6_publication_id"])
        pub_arch = df.set_index(L.ID_COL)[L.PUB_COL]
        lan_model = df[model & df["metal_category"].eq("lanthanide")]
        summary["bundle_overlap"] = {
            "n_bundle_rows": int(len(bo)), "n_joined": int(bo["joined"].sum()),
            "join_rate": float(bo["joined"].mean()),
            "n_metal_agree": int(j["metal_agree"].sum()), "n_ox_agree": int(j["ox_agree"].sum()),
            "n_archive_ox_unknown": int(pd.to_numeric(j["archive_ox"], errors="coerce").isna().sum()),
            "bundle_ox_where_archive_ox_unknown": _vc(j.loc[pd.to_numeric(j["archive_ox"], errors="coerce").isna(),
                                                        "bundle_metal_ox"]),
            "tier_where_archive_ox_unknown": _vc(j.loc[pd.to_numeric(j["archive_ox"], errors="coerce").isna(),
                                                   "g19_tier"]),
            "n_log_D_both_finite": int(j["abs_delta_log_D"].notna().sum()),
            "max_abs_delta_log_D": float(j["abs_delta_log_D"].max()),
            "n_abs_delta_log_D_gt_1e-6": int(j["abs_delta_log_D"].gt(1e-6).sum()),
            "n_abs_delta_log_D_gt_0.005": int(j["abs_delta_log_D"].gt(0.005).sum()),
            "n_acid_M_agree": int(j["acid_M_agree"].sum()), "n_extractant_M_agree": int(j["extractant_M_agree"].sum()),
            "n_temperature_agree": int(j["temperature_agree"].sum()),
            "n_smiles_equals_archive_primary": int(j["smiles_equals_archive_primary"].sum()),
            "n_smiles_in_archive_system": int(j["smiles_in_archive_system"].sum()),
            "tier_counts": _vc(j["g19_tier"]), "duplicate_class_counts": _vc(j["duplicate_class"]),
            "model_readiness_counts": _vc(j["model_readiness"]),
            "n_noncanonical_bundle_rows": int((~j["is_canonical_row"].astype(bool)).sum()),
            "n_duplicate_groups_with_2plus_bundle_rows": int(
                j.loc[j["n_bundle_rows_in_duplicate_group"] > 1, "duplicate_group_id"].nunique()),
            "n_bundle_rows_in_groups_with_2plus_bundle_rows": int((j["n_bundle_rows_in_duplicate_group"] > 1).sum()),
            "bundle_split_values": _vc(bo["bundle_split"]),
            "gen6_partition": {
                "n_rows_compared": int(len(g6)),
                "n_publication_id_identical": int(g6["publication_id_identical"].sum()),
                "max_g19_ids_per_gen6_id": int(g6.groupby("gen6_publication_id")[L.PUB_COL].nunique().max()),
                "max_gen6_ids_per_g19_id": int(g6.groupby(L.PUB_COL)["gen6_publication_id"].nunique().max()),
            },
            "archive_model_lanthanide_rows": int(len(lan_model)),
            "archive_model_lanthanide_rows_not_in_bundle": int((~lan_model["g19_bundle_exp_id"].isin(
                bo["exp_id"].dropna())).sum()),
            "n_archive_publications_of_bundle_rows": int(pub_arch.reindex(j[L.ID_COL]).nunique()),
        }

        # ---- summary ----------------------------------------------------------------------------
        summary["pair_listing_truncation"] = trunc
        summary["truncated_any"] = any(v["truncated"] for v in trunc.values())
        summary_path = write_json(out_dir / "leakage_summary.json", summary)
        outputs.append(summary_path)
        for p in outputs:
            size = Path(p).stat().st_size
            if size > MAX_CSV_BYTES:
                raise RuntimeError(f"{paths.rel(p)} is {size} bytes (> {MAX_CSV_BYTES}); lower --max-pair-rows")
        run.outputs(outputs)
    return summary


if __name__ == "__main__":
    s = main()
    print(f"archive rows {s['archive']['n_rows']}, MODEL {s['archive']['n_model_rows']}")
    for pop in ("all_rows", "model_rows"):
        for sig in ("sig6", "sig3"):
            c = s["near_duplicates"][pop][sig]
            print(f"near-duplicate pairs {pop} {sig}: {c['n_pairs']} (cross-publication {c['n_pairs_cross_publication']})")
    print("V1 burden (MODEL, key sig6 any value):", s["v1_near_duplicate_burden_model_rows"]["key_sig6_any_value"])
    print("truncated:", s["truncated_any"])
