"""Stage 3 -- hierarchical duplicate detection and classification.

The archive duplicates records in several different ways, and they do not all
mean the same thing scientifically.  This stage separates them:

``A_EXACT_DATABASE_DUPLICATE``      same conditions, same value, same citation
``B_SAME_MEASUREMENT_DIFF_PROV``    same conditions and value under >1 citation
``C_POSSIBLE_INDEPENDENT_REPLICATE`` same conditions, low-precision equal value,
                                     different citation -- may be a real repeat
``D_CONDITION_CONFLICT``            near-identical but one condition differs
``E_VALUE_CONFLICT``                identical conditions, different log D
``F_INSUFFICIENT_INFORMATION``      a field needed for the decision is missing

Nothing is averaged and nothing is deleted: every record keeps its own row in
the audit tables, and only class A/B collapse to a shared canonical id.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_identity as ID                                        # noqa: E402
from sae_paths import INTERMEDIATE_DIR, AUDIT_DIR, ensure_dirs    # noqa: E402


# --------------------------------------------------------------------------
# Derived signatures
# --------------------------------------------------------------------------

def _string_set(series) -> list[str]:
    """Sorted unique non-null strings.

    ``{x for x in series if x}`` is not a null test: pandas' missing value is
    truthy, so it used to leak into the DOI lists and serialise as the literal
    string ``"None"`` -- a citation that does not exist.
    """
    return sorted({str(x) for x in series if isinstance(x, str) and x})


def add_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["acid_signature"] = frame["acid_names"].apply(
        lambda names: "|".join(sorted(names)) if len(names) else "")
    return frame


# --------------------------------------------------------------------------
# The seven detection levels required by the brief
# --------------------------------------------------------------------------

# The one field set the whole ladder compares.  Every level describes the SAME
# experimental variables; the levels differ only in how canonical the
# representation is, so each step can only merge rows, never split them.
LADDER_FIELDS = (
    "metal", "oxidation_state", "extractant_identity", "extractant_concentration",
    "acid_identity", "acid_concentration", "acid_concentration_organic",
    "solvent", "modifier_identity", "modifier_concentration",
    "complexant_identity", "complexant_concentration",
    "holdback_identity", "holdback_concentration", "nitrate_concentration",
    "metal_concentration", "temperature", "contact_time", "shaking_time",
    "phase_ratio", "measured_value",
)


def level_ladder(records: pd.DataFrame, raw_rows: pd.DataFrame) -> list[dict]:
    """Count duplicate groups found by each successively weaker rule.

    Levels 2-6 all describe the same 21 experimental variables and differ only
    in how canonical each variable's representation is.  Every step therefore
    merges rows and can never split them, which is asserted at the end: a
    non-monotone ladder would mean a normalisation step is *creating*
    distinctions instead of removing them.
    """
    ladder = []

    def text(frame, column):
        return frame[column].astype("string").fillna("").str.strip().str.casefold()

    def combine(columns_by_field, population, level, name, description):
        parts = [columns_by_field[field] for field in LADDER_FIELDS]
        out = parts[0]
        for part in parts[1:]:
            out = out + "\x1f" + part
        counts = out.astype(str).value_counts()
        ladder.append({
            "level": level, "name": name, "description": description,
            "population": population,
            "groups": int((counts > 1).sum()),
            "rows_in_groups": int(counts[counts > 1].sum()),
            "redundant_rows": int(counts[counts > 1].sum() - (counts > 1).sum()),
        })

    # ---- L1: the raw export, before the per-metal fan-out is collapsed -----
    payload = [c for c in raw_rows.columns
               if c not in ("source_file", "source_line_number", "raw_row_id", "source_record_id")]
    blob = raw_rows[payload].astype("string").fillna("")
    joined = blob[payload[0]]
    for column in payload[1:]:
        joined = joined + "\x1f" + blob[column]
    sizes = joined.astype(str).value_counts()
    ladder.append({
        "level": 1, "name": "exact_raw_duplicate_rows",
        "description": "byte-identical raw CSV payloads; this is the per-metal export fan-out",
        "population": f"{len(raw_rows):,} raw rows",
        "groups": int((sizes > 1).sum()),
        "rows_in_groups": int(sizes[sizes > 1].sum()),
        "redundant_rows": int(sizes[sizes > 1].sum() - (sizes > 1).sum()),
    })

    frame = records
    population = f"{len(records):,} records"

    def numeric(column, level_name):
        values = frame[column]
        if level_name == "exact":
            return values.map(lambda v: "" if pd.isna(v) else repr(float(v))).astype("string")
        return values.map(lambda v: "" if pd.isna(v) else str(ID.quantize(v, level_name))).astype("string")

    def structure_list(column):
        return frame[column].map(
            lambda v: "" if v is None else "|".join(sorted("" if x is None else str(x) for x in v))
        ).astype("string")

    # ---- L2: raw text, whitespace/case folded only ------------------------
    l2 = {
        "metal": text(frame, "metal_raw"),
        "oxidation_state": text(frame, "metal_oxidation_state_raw"),
        "extractant_identity": text(frame, "extractant_name_raw") + "\x1e" + text(frame, "extractant_smiles_raw"),
        "extractant_concentration": text(frame, "extractant_concentration_raw"),
        "acid_identity": text(frame, "acid_name_raw"),
        "acid_concentration": text(frame, "acid_concentration_raw"),
        "acid_concentration_organic": text(frame, "acid_concentration_organic_raw"),
        "solvent": text(frame, "solvent_name_raw"),
        "modifier_identity": text(frame, "modifier_name_raw"),
        "modifier_concentration": text(frame, "modifier_concentration_raw"),
        "complexant_identity": text(frame, "complexant_name_raw") + "\x1e" + text(frame, "complexant_smiles_raw"),
        "complexant_concentration": text(frame, "complexant_concentration_raw"),
        "holdback_identity": text(frame, "holdback_smiles_raw"),
        "holdback_concentration": text(frame, "holdback_concentration_raw"),
        "nitrate_concentration": text(frame, "nitrate_concentration_raw"),
        "metal_concentration": text(frame, "metal_concentration_raw"),
        "temperature": text(frame, "temperature_raw"),
        "contact_time": text(frame, "contact_time_raw"),
        "shaking_time": text(frame, "shaking_time_raw"),
        "phase_ratio": text(frame, "phase_ratio_raw"),
        "measured_value": text(frame, "D_raw"),
    }
    combine(l2, population, 2, "whitespace_case_normalised",
            "the 21 experimental variables as raw text, whitespace/case folded")

    # ---- L3: canonical metal / acid / solvent names -----------------------
    l3 = dict(l2)
    l3["metal"] = frame["metal_symbol"].astype("string").fillna(text(frame, "metal_raw"))
    l3["oxidation_state"] = frame["metal_oxidation_state"].astype("string").fillna("")
    l3["acid_identity"] = frame["acid_signature"].astype("string").fillna("")
    l3["solvent"] = frame["solvent_key"].astype("string").fillna("")
    combine(l3, population, 3, "canonical_chemical_names",
            "L2 plus canonical metal symbols, acid identities and diluent keys")

    # ---- L4: canonical structures, order-invariant -------------------------
    l4 = dict(l3)
    l4["extractant_identity"] = frame["extractant_system_key"].astype("string").fillna("")
    l4["complexant_identity"] = frame["complexant_smiles_canonical"].astype("string").fillna("")
    l4["holdback_identity"] = frame["holdback_smiles_canonical"].astype("string").fillna("")
    combine(l4, population, 4, "canonical_structures",
            "L3 plus RDKit canonicalisation and order-invariant component ordering")

    # ---- L5: numerics parsed into canonical units, compared bit-exactly ----
    l5 = dict(l4)
    l5["extractant_concentration"] = frame["extractant_concentrations_M"].map(
        lambda v: "" if v is None else "|".join(
            sorted("" if x is None or pd.isna(x) else repr(float(x)) for x in v))
    ).astype("string")
    for field, column in (("acid_concentration", "acid_concentration_M"),
                          ("acid_concentration_organic", "acid_concentration_organic_M"),
                          ("modifier_concentration", "modifier_concentration_M"),
                          ("complexant_concentration", "complexant_concentration_M"),
                          ("holdback_concentration", "holdback_concentration_M"),
                          ("nitrate_concentration", "nitrate_concentration_M"),
                          ("metal_concentration", "metal_concentration_M"),
                          ("temperature", "temperature_C"),
                          ("contact_time", "contact_time_min"),
                          ("shaking_time", "shaking_time_min"),
                          ("phase_ratio", "phase_ratio_org_aq"),
                          ("measured_value", "log_D")):
        l5[field] = numeric(column, "exact")
    l5["modifier_identity"] = frame["modifier_name"].astype("string").fillna("")
    combine(l5, population, 5, "scientific_identity_exact_numerics",
            "L4 with every numeric parsed into its canonical unit (M / min / C)")

    # ---- L6: quantised numerics -------------------------------------------
    l6 = dict(l5)
    l6["extractant_concentration"] = frame["extractant_concentrations_M"].map(
        lambda v: "" if v is None else "|".join(
            sorted("" if x is None or pd.isna(x) else str(ID.quantize(x, ID.DEFAULT_TOLERANCE))
                   for x in v))
    ).astype("string")
    for field, column in (("acid_concentration", "acid_concentration_M"),
                          ("acid_concentration_organic", "acid_concentration_organic_M"),
                          ("modifier_concentration", "modifier_concentration_M"),
                          ("complexant_concentration", "complexant_concentration_M"),
                          ("holdback_concentration", "holdback_concentration_M"),
                          ("nitrate_concentration", "nitrate_concentration_M"),
                          ("metal_concentration", "metal_concentration_M"),
                          ("temperature", "temperature_C"),
                          ("contact_time", "contact_time_min"),
                          ("shaking_time", "shaking_time_min"),
                          ("phase_ratio", "phase_ratio_org_aq"),
                          ("measured_value", "log_D")):
        l6[field] = numeric(column, ID.DEFAULT_TOLERANCE)
    combine(l6, population, 6, f"scientific_identity_{ID.DEFAULT_TOLERANCE}",
            "L5 with significant-figure quantisation of every numeric field")

    for previous, current in zip(ladder[1:-1], ladder[2:]):
        if current["redundant_rows"] < previous["redundant_rows"]:
            raise AssertionError(
                f"level ladder is not monotone: L{current['level']} finds fewer "
                f"redundant rows ({current['redundant_rows']}) than "
                f"L{previous['level']} ({previous['redundant_rows']}); a "
                "normalisation step is creating distinctions instead of removing them")

    # ---- L7: same conditions, different value -----------------------------
    conditions_only = [ID.identity_hash(row, ID.DEFAULT_TOLERANCE)
                       for row in records.to_dict("records")]
    scratch = records.assign(_k=conditions_only)
    conflicts = conflict_rows = 0
    for _, group in scratch.groupby("_k"):
        if len(group) > 1:
            distinct = {ID.quantize(v, ID.DEFAULT_TOLERANCE) for v in group["log_D"].dropna()}
            if len(distinct) > 1:
                conflicts += 1
                conflict_rows += len(group)
    ladder.append({
        "level": 7, "name": "same_conditions_different_value",
        "description": "identity groups whose members disagree on the measured log D",
        "population": population,
        "groups": conflicts, "rows_in_groups": conflict_rows, "redundant_rows": 0,
    })
    return ladder


# --------------------------------------------------------------------------
# Group classification
# --------------------------------------------------------------------------

def classify_group(group: pd.DataFrame) -> tuple[str, str]:
    """Return ``(class, reason)`` for one identity group."""
    missing = [f for f in ID.REQUIRED_FOR_DECISION
               if group[f].isna().any() or (group[f].astype(str) == "None").any()]
    if missing:
        return "F_INSUFFICIENT_INFORMATION", f"missing required field(s): {','.join(sorted(set(missing)))}"

    values = group["log_D"].dropna()
    quantised = {ID.quantize(v, ID.DEFAULT_TOLERANCE) for v in values}
    dois = _string_set(group["doi_primary"])
    doi_sets = {tuple(sorted(v)) if v is not None else ()
                for v in group["doi_source_all"]}
    # 1,104 records carry only the archive's self-citation, so the DOI alone
    # cannot separate their provenance.  The sub-source file and the figure or
    # table the point was digitised from stand in for it.
    provenance = {
        (d or "", f or "", loc or "")
        for d, f, loc in zip(group["doi_primary"].fillna(""),
                             group["sub_source_file"].fillna(""),
                             group["data_location"].fillna(""))
    }

    if len(quantised) > 1:
        spread = float(values.max() - values.min())
        return "E_VALUE_CONFLICT", (
            f"{len(quantised)} distinct log D values under identical conditions "
            f"(spread {spread:.4g} log units)")

    # From here the measured value agrees.
    precision = max(ID.significant_digits(x) for x in group["D_raw"].fillna(""))
    if len(dois) <= 1 and len(doi_sets) <= 1 and len(provenance) <= 1:
        return "A_EXACT_DATABASE_DUPLICATE", (
            "identical conditions, identical value and identical provenance -- "
            "the archive stores the same record more than once")
    if precision >= 5:
        return "B_SAME_MEASUREMENT_DIFF_PROV", (
            f"identical value agreeing to {precision} significant digits under "
            f"{len(provenance)} distinct provenances ({len(dois)} citation(s)) -- "
            "copied between sources rather than re-measured")
    return "C_POSSIBLE_INDEPENDENT_REPLICATE", (
        f"value agrees but only to {precision} significant digit(s) while provenance "
        f"differs ({len(provenance)} distinct provenances) -- may be an independent repeat")


def detect_condition_conflicts(records: pd.DataFrame, level: str) -> pd.DataFrame:
    """Find records that report the *same* value under a differing condition.

    A concentration sweep also has "identical conditions except one" -- that is
    the whole point of a titration curve, and those points have different log D.
    What is suspicious is the opposite: two records that agree on the measured
    value to full precision while disagreeing about a condition.  That pattern
    usually means one of the two conditions was transcribed wrongly, so it is
    reported for adjudication and never merged.
    """
    fields = list(ID.IDENTITY_FIELDS_CATEGORICAL) + list(ID.IDENTITY_FIELDS_NUMERIC)
    rows = []
    full = [ID.identity_tuple(row, level) for row in records.to_dict("records")]
    frame = records.assign(_full=full)

    for position, field in enumerate(fields):
        reduced = [t[:position] + t[position + 1:] for t in full]
        scratch = frame.assign(_red=["\x1f".join(t) for t in reduced])
        for key, group in scratch.groupby("_red", sort=True):
            if len(group) < 2:
                continue
            variants = {t[position] for t in group["_full"]}
            if len(variants) < 2:
                continue
            values = {ID.quantize(v, level) for v in group["log_D"].dropna()}
            if len(values) != 1 or group["log_D"].isna().any():
                # Different values under a different condition is ordinary
                # experimental design, not a conflict.
                continue
            rows.append({
                "conflict_group_id": "CC" + hashlib.sha256(
                    (field + "\x1e" + key).encode("utf-8")).hexdigest()[:16],
                "differing_field": field,
                "n_records": int(len(group)),
                "distinct_values_of_field": sorted(str(v) for v in variants),
                "canonical_measurement_ids": sorted(group["canonical_measurement_id"]),
                "shared_log_D": float(group["log_D"].iloc[0]),
                "dois": sorted({d for d in group["doi_primary"].fillna("") if d}),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    records = add_signatures(pd.read_parquet(INTERMEDIATE_DIR / "records_normalized.parquet"))
    raw_rows = pd.read_parquet(INTERMEDIATE_DIR / "raw_rows.parquet")

    ladder = level_ladder(records, raw_rows)

    # ---- tolerance sensitivity -------------------------------------------
    sensitivity = []
    for level in ID.TOLERANCE_LEVELS:
        rows_as_dicts = records.to_dict("records")
        keys = pd.Series([ID.identity_hash(r, level) for r in rows_as_dicts],
                         index=records.index)
        counts = keys.value_counts()
        # The value tolerance is swept together with the condition tolerance,
        # otherwise the conflict count would be an artefact of one fixed choice.
        quantised_value = pd.Series([ID.quantize(r.get("log_D"), level) for r in rows_as_dicts],
                                    index=records.index)
        with_value = (records.assign(_k=keys, _v=quantised_value)
                      .dropna(subset=["_v"]).groupby("_k")["_v"].nunique())
        sensitivity.append({
            "tolerance_level": level,
            "significant_digits": ID.TOLERANCE_LEVELS[level],
            "identity_groups": int(len(counts)),
            "groups_with_duplicates": int((counts > 1).sum()),
            "records_in_duplicate_groups": int(counts[counts > 1].sum()),
            "groups_with_conflicting_values": int((with_value > 1).sum()),
        })

    # ---- final grouping at the documented default -------------------------
    level = ID.DEFAULT_TOLERANCE
    records["identity_hash"] = [ID.identity_hash(r, level) for r in records.to_dict("records")]

    # Deterministic group ids: sort by the identity hash, not by row order.
    ordered = sorted(records["identity_hash"].unique())
    group_ids = {h: f"DG{index:06d}" for index, h in enumerate(ordered)}
    records["duplicate_group_id"] = records["identity_hash"].map(group_ids)

    classes, reasons = {}, {}
    for key, group in records.groupby("duplicate_group_id", sort=True):
        if len(group) == 1:
            missing = [f for f in ID.REQUIRED_FOR_DECISION
                       if group[f].isna().any() or (group[f].astype(str) == "None").any()]
            classes[key] = "F_INSUFFICIENT_INFORMATION" if missing else "UNIQUE"
            reasons[key] = (f"singleton missing {','.join(sorted(set(missing)))}"
                            if missing else "singleton scientific identity")
        else:
            classes[key], reasons[key] = classify_group(group)

    records["duplicate_class"] = records["duplicate_group_id"].map(classes)
    records["duplicate_class_reason"] = records["duplicate_group_id"].map(reasons)
    records["duplicate_group_size"] = records.groupby("duplicate_group_id")["duplicate_group_id"].transform("size")

    # Canonical representative: lowest numeric archive id, so the choice is
    # reproducible and independent of row order.
    records["_numeric_id"] = pd.to_numeric(records["source_record_id"], errors="coerce")
    representative = (records.sort_values(["duplicate_group_id", "_numeric_id", "source_record_id"])
                      .groupby("duplicate_group_id").head(1)
                      .set_index("duplicate_group_id")["canonical_measurement_id"])
    records["group_representative_id"] = records["duplicate_group_id"].map(representative)

    # Only A and B are true redundancy.  C/D/E/F keep every record as its own
    # observation because collapsing them would destroy real information.
    collapsible = records["duplicate_class"].isin(
        ["A_EXACT_DATABASE_DUPLICATE", "B_SAME_MEASUREMENT_DIFF_PROV"])
    records["is_canonical_row"] = (~collapsible) | (
        records["canonical_measurement_id"] == records["group_representative_id"])
    records = records.drop(columns=["_numeric_id"])

    conflicts = detect_condition_conflicts(records, level)

    records.to_parquet(INTERMEDIATE_DIR / "records_grouped.parquet", index=False)

    group_table = (records.groupby("duplicate_group_id")
                   .agg(duplicate_class=("duplicate_class", "first"),
                        reason=("duplicate_class_reason", "first"),
                        size=("duplicate_group_id", "size"),
                        representative=("group_representative_id", "first"),
                        members=("canonical_measurement_id", lambda s: sorted(s)),
                        dois=("doi_primary", _string_set),
                        sub_sources=("sub_source_file", _string_set),
                        data_locations=("data_location", _string_set),
                        log_D_values=("log_D", lambda s: [None if pd.isna(v) else float(v) for v in s]))
                   .reset_index())
    group_table.to_parquet(AUDIT_DIR / "duplicate_groups.parquet", index=False)
    if len(conflicts):
        conflicts = conflicts.sort_values(["differing_field", "conflict_group_id"]).reset_index(drop=True)
    conflicts.to_parquet(AUDIT_DIR / "condition_conflict_candidates.parquet", index=False)

    summary = {
        "records_in": int(len(records)),
        "identity_groups": int(records["duplicate_group_id"].nunique()),
        "canonical_rows": int(records["is_canonical_row"].sum()),
        "collapsed_as_redundant": int((~records["is_canonical_row"]).sum()),
        "class_counts_by_record": records["duplicate_class"].value_counts().to_dict(),
        "class_counts_by_group": group_table["duplicate_class"].value_counts().to_dict(),
        "level_ladder": ladder,
        "tolerance_sensitivity": sensitivity,
        "condition_conflict_candidate_groups": int(len(conflicts)),
        "tolerance_default": level,
        "numeric_field_policy": ID.NUMERIC_FIELD_POLICY,
    }
    (AUDIT_DIR / "stage03_dedup_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

    printable = {k: v for k, v in summary.items()
                 if k not in ("numeric_field_policy", "level_ladder", "tolerance_sensitivity")}
    print(json.dumps(printable, indent=2))
    print("\n--- level ladder ---")
    for entry in ladder:
        print(f"  L{entry['level']} {entry['name']:42s} groups={entry['groups']:6d} "
              f"redundant_rows={entry['redundant_rows']}")
    print("\n--- tolerance sensitivity ---")
    for entry in sensitivity:
        print(f"  {entry['tolerance_level']:6s} groups={entry['identity_groups']:6d} "
              f"dup_groups={entry['groups_with_duplicates']:6d} "
              f"in_dups={entry['records_in_duplicate_groups']:6d} "
              f"value_conflicts={entry['groups_with_conflicting_values']:5d}")


if __name__ == "__main__":
    main()
