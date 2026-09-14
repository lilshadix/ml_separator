"""Stage 6c -- the human-readable quality report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_paths import CLEAN_DIR, AUDIT_DIR, REPORTS_DIR, ensure_dirs  # noqa: E402

MEANINGFUL_FIELDS = [
    "metal_symbol", "metal_oxidation_state", "extractant_primary_smiles",
    "extractant_primary_concentration_M", "acid_primary", "acid_concentration_M",
    "solvent_key", "metal_concentration_M", "temperature_C", "contact_time_min",
    "shaking_time_min", "phase_ratio_org_aq", "acid_concentration_organic_M",
    "nitrate_concentration_M", "modifier_name", "modifier_concentration_M",
    "complexant_smiles_canonical", "holdback_smiles_canonical", "log_D",
]

CONDITION_RANGES = [
    ("extractant_primary_concentration_M", "M"), ("acid_concentration_M", "M"),
    ("metal_concentration_M", "M"), ("temperature_C", "C"),
    ("contact_time_min", "min"), ("shaking_time_min", "min"),
    ("phase_ratio_org_aq", "ratio"), ("nitrate_concentration_M", "M"),
    ("log_D", "log10 D"),
]


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float) and not np.isfinite(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.{digits}g}"


def has_flag(series, flag):
    return series.map(lambda f: flag in f if f is not None else False)


def main() -> None:
    ensure_dirs()
    master = pd.read_parquet(CLEAN_DIR / "master_clean.parquet")
    groups = pd.read_parquet(AUDIT_DIR / "duplicate_groups.parquet")
    conflicts = pd.read_parquet(AUDIT_DIR / "condition_conflict_candidates.parquet")
    queue = pd.read_csv(AUDIT_DIR / "manual_review_queue.csv")
    payload = json.loads((REPORTS_DIR / "_report_payload.json").read_text())
    metal_cov = pd.read_csv(REPORTS_DIR / "metal_coverage.csv")

    dedup, series, accounting = payload["dedup"], payload["series"], payload["accounting"]
    fanout, norm = payload["fanout"], payload["normalization"]

    lines = []
    add = lines.append

    add("# Quality report — multi-metal Separation Archive for Elements build")
    add("")
    add(f"`{len(master):,}` archive records distilled from `{accounting['raw_rows']:,}` raw CSV rows "
        f"across {fanout['distinct_exp_ids']:,} distinct `exp_id` values. "
        "The frozen gen7–gen10 lanthanide pipeline was not modified, retrained or re-split.")
    add("")

    # ---------------- headline ----------------
    add("## 1. Headline numbers")
    add("")
    readiness = master["model_readiness"].value_counts()
    add(table(["quantity", "value", "meaning"], [
        ["raw CSV rows", fmt(accounting["raw_rows"]), "41 per-metal files, immutable in `raw/`"],
        ["archive records", fmt(len(master)), "one row per `exp_id`"],
        ["rows folded as export fan-out", fmt(fanout["raw_rows_folded_away"]),
         "same record re-exported once per metal file"],
        ["canonical measurements", fmt(int(master["is_canonical_row"].sum())),
         "after collapsing class A/B duplicates only"],
        ["model-ready (tier A)", fmt(int(readiness.get("A_model_ready", 0))),
         "all required fields, no unresolved conflict"],
        ["usable with caveats (tier B)", fmt(int(readiness.get("B_usable_with_caveats", 0))),
         "complete but in a value conflict or carrying a suspect flag"],
        ["not modelable (tier C)", fmt(int(readiness.get("C_not_modelable", 0))),
         "a required field is missing"],
        ["redundant duplicate (tier D)", fmt(int(readiness.get("D_redundant_duplicate", 0))),
         "non-representative member of an A/B group"],
        ["unique metals", fmt(int(master["metal_symbol"].nunique())),
         "distinct elements; the archive's 41 raw tokens include `UO2+2`, resolved to U(VI)"],
        ["unique extractant systems", fmt(int(master["extractant_system_key"].nunique())),
         "order-invariant canonical-structure keys"],
        ["unique diluent systems", fmt(int(master["solvent_key"].nunique())), "after mixture parsing"],
        ["usable curves", fmt(series["curves_usable"]), f"≥{series['min_curve_points']} points on one axis"],
    ]))
    add("")

    # ---------------- the dominant duplication mechanism ----------------
    add("## 2. Why 48,471 rows are only 16,770 measurements")
    add("")
    add("The archive was exported one CSV per **queried** metal, not per **measured** metal. "
        "A record whose measured metal is Nd appears — byte-identically — in `Am.csv`, `Ba.csv`, "
        "`Ca.csv`, `In.csv`, `La.csv`, `Mo.csv`, `Nd.csv`, `Pa.csv`, `Pr.csv`, `Th.csv`, `U.csv` "
        "and `Y.csv`, because all twelve metals were present in that experimental system.")
    add("")
    add("Before collapsing this, all 37 raw columns were checked for variation inside each "
        f"`exp_id` group: **{fanout['columns_varying_within_exp_id'] or 'none'} varied**, in "
        f"{fanout['exp_id_groups_with_variation']} of {fanout['distinct_exp_ids']:,} groups. "
        "The fan-out is therefore pure export metadata and collapsing it loses no science. "
        "Every contributing raw row is retained in `raw_row_ids` and in "
        "`raw_to_canonical_mapping.parquet`.")
    add("")
    add("**The file name is not the metal.** Any pipeline that infers the metal from the source "
        "file will mislabel a large fraction of this archive.")
    add("")
    hist = fanout["fanout_size_histogram"]
    add(table(["copies of one record", "records"],
              [[k, fmt(int(v))] for k, v in sorted(hist.items(), key=lambda kv: int(kv[0]))]))
    add("")

    # ---------------- duplicate ladder ----------------
    add("## 3. Hierarchical duplicate detection")
    add("")
    add(table(["level", "rule", "groups", "redundant rows", "population"],
              [[f"L{e['level']}", e["name"], fmt(e["groups"]), fmt(e["redundant_rows"]),
                e.get("population", "")] for e in dedup["level_ladder"]]))
    add("")
    add("Levels 2–6 describe the same 21 variables and differ only in how canonical the "
        "representation is, so each level can only merge rows. The pipeline asserts this "
        "monotonicity and fails rather than reporting a ladder where a normalisation step "
        "*creates* distinctions.")
    add("")
    add("**The 21 variables include the measured value.** These levels answer *\"is this the "
        "same recorded row?\"*, so two records that agree on every condition but disagree on "
        "log D are correctly NOT merged here. The tolerance sweep in §4 asks a different "
        "question — it keys on conditions alone and reports value disagreement separately — "
        "which is why its group counts differ from this table at the same `sig6` level.")
    add("")
    gain = dedup["level_ladder"][5]["redundant_rows"] - dedup["level_ladder"][1]["redundant_rows"]
    add(f"Normalisation beyond the export fan-out is worth only {gain} additional redundant rows: "
        "this archive's text is already clean, and essentially all duplication is the per-metal "
        "fan-out.")
    add("")

    # ---------------- classes ----------------
    add("### Duplicate group classes")
    add("")
    by_group = groups["duplicate_class"].value_counts()
    by_record = master["duplicate_class"].value_counts()
    descriptions = {
        "A_EXACT_DATABASE_DUPLICATE": "same conditions, same value, same provenance — collapsed",
        "B_SAME_MEASUREMENT_DIFF_PROV": "same conditions and value under different provenance — collapsed, all citations kept",
        "C_POSSIBLE_INDEPENDENT_REPLICATE": "value agrees only to low precision under different provenance — **kept separate**",
        "E_VALUE_CONFLICT": "identical conditions, different log D — **all values kept, never averaged**",
        "F_INSUFFICIENT_INFORMATION": "a field needed for the decision is missing — **kept, flagged**",
        "UNIQUE": "singleton scientific identity",
    }
    sizes = master.groupby("duplicate_group_id").size()
    multi = set(sizes[sizes > 1].index)
    multi_groups = (master[master["duplicate_group_id"].isin(multi)]
                    .drop_duplicates("duplicate_group_id")["duplicate_class"].value_counts())
    add(table(["class", "identity groups", "of those, groups with >1 member", "records", "treatment"],
              [[k, fmt(int(by_group.get(k, 0))), fmt(int(multi_groups.get(k, 0))),
                fmt(int(by_record.get(k, 0))), descriptions.get(k, "")] for k in
               ["UNIQUE", "A_EXACT_DATABASE_DUPLICATE", "B_SAME_MEASUREMENT_DIFF_PROV",
                "C_POSSIBLE_INDEPENDENT_REPLICATE", "E_VALUE_CONFLICT",
                "F_INSUFFICIENT_INFORMATION"]]))
    add("")
    add("The middle column matters: an *identity group* is one scientific identity, which may "
        "have a single member. Most `F_INSUFFICIENT_INFORMATION` groups are singletons — records "
        "whose missing fields prevent any duplicate judgement, not records duplicated by the "
        "archive. Only the A/B rows represent redundancy that was actually collapsed.")
    add("")
    add(f"**D_CONDITION_CONFLICT** is reported separately in `condition_conflict_candidates.parquet`: "
        f"{len(conflicts):,} groups report an *identical* log D while disagreeing about exactly one "
        "condition. A differing condition with a differing value is an ordinary titration point, "
        "not a conflict, so only the same-value case is flagged.")
    add("")

    # ---------------- tolerance ----------------
    add("## 4. Numeric tolerance sensitivity")
    add("")
    add(table(["level", "sig. digits", "identity groups", "duplicate groups",
               "records in duplicates", "value conflicts"],
              [[e["tolerance_level"], e["significant_digits"] or "exact",
                fmt(e["identity_groups"]), fmt(e["groups_with_duplicates"]),
                fmt(e["records_in_duplicate_groups"]), fmt(e["groups_with_conflicting_values"])]
               for e in dedup["tolerance_sensitivity"]]))
    add("")
    add(f"The default is `{dedup['tolerance_default']}` (6 significant figures, 1 ppm relative). "
        "Exact, 12-, 9- and 6-figure comparison give **identical** groupings, so the choice is not "
        "load-bearing: the archive's condition values carry no float-serialisation noise. Only at "
        "3 significant figures does the grouping move materially, which is coarse enough to merge "
        "genuinely different titration points and is therefore rejected.")
    add("")
    add("Per-field units, parsers and conversions are documented in `field_mapping.csv` and in "
        "`stage03_dedup_report.json` → `numeric_field_policy`.")
    add("")

    # ---------------- composition ----------------
    add("## 5. Chemical systems and multi-component handling")
    add("")
    classes = master["system_component_class"].value_counts()
    add(table(["system class", "records"], [[k, fmt(int(v))] for k, v in classes.items()]))
    add("")
    add("The gen10 audit found a measurable error penalty for representing a multi-component "
        "system by one component's structure. The frozen table does exactly that: its 18 "
        "`TODGA,DHOA` rows carry `canonical_smiles = TODGA` and DHOA is gone. Here every "
        "chemically active component is kept in the `components` list with its own role, "
        "structure and concentration, and `extractant_system_key` is the order-invariant join "
        "of the sorted canonical SMILES.")
    add("")
    add("Component order in the archive is **not** stable — `2-bromodecanoic acid, N-DPP` appears "
        "with its two SMILES in both orders — which is why the identity key sorts components "
        "before hashing.")
    add("")

    # ---------------- structural integrity ----------------
    add("### Structure/name integrity findings")
    add("")
    add("- **TODGA's SMILES is attached to 22 different extractant names.** One sub-source "
        "(`./ST*.json`, 1,104 records) records the *water-soluble masking agent's* name in "
        "`Extractant_Name` while `Extractant_SMILES` holds the organic extractant, and puts the "
        "agent's real structure in `comments_description → Holdback_Agent_SMILES`. Identity here "
        "is keyed on structure, never on name. *(The frozen pipeline hits the same bug and "
        "hard-codes a quarantine for it at `pairs.py:264`.)*")
    add("- **TBP and DHOA never carry a SMILES** in any of their 972 / 505 single-component "
        "records. Both were recovered *from the archive itself* by elimination: in two-component "
        "records such as `TODGA, TBP` every other name already has a consensus structure, so the "
        "remaining SMILES must be theirs. Recovered: "
        + ", ".join(f"`{k}`" for k in norm["structure_deduction_by_elimination"]["names_deduced"])
        + ". Their components carry `structure_source = \"deduced_by_elimination\"`, a tag "
        "used by nothing else, so they can be excluded on their own. (`name_consensus` is "
        "the tag for ordinary archive-confirmed lookups and covers most of the table — "
        "filtering on it would remove almost everything.)")
    add(f"- {int(has_flag(master['flags'], 'extractant_name_structure_conflict').sum()):,} "
        "records have a name that disagrees with the structure the archive gives that name "
        "elsewhere; all are queued for review, none were auto-resolved.")
    add("- `Br-Cosan` resolves to `Cc1cc(Br)ccc1NC(=O)CCl`, a bromo-methyl chloroacetanilide. "
        "Br-Cosan is a cobalt bis(dicarbollide); the archive's structure is chemically unrelated. "
        "It is **preserved as recorded** and flagged, not silently corrected.")
    add("")

    # ---------------- metals ----------------
    add("## 6. Metal coverage")
    add("")
    non_ln = int(((master["is_lanthanide"] == False) & master["metal_symbol"].notna()).sum())  # noqa: E712
    add(f"{int(master['is_lanthanide'].sum()):,} lanthanide records, {non_ln:,} non-lanthanide, "
        f"{int(master['metal_symbol'].isna().sum()):,} with no resolvable metal.")
    add("")
    top = metal_cov.dropna(subset=["metal_symbol"]).head(20)
    add(table(["metal", "category", "records", "model-ready", "extractant systems",
               "log D min", "log D max"],
              [[r["metal_symbol"], r["category"], fmt(int(r["records"])), fmt(int(r["model_ready"])),
                fmt(int(r["extractant_systems"])), fmt(r["log_D_min"]), fmt(r["log_D_max"])]
               for _, r in top.iterrows()]))
    add("")
    add("Full per-metal statistics are in `metal_coverage.csv`; the extractant × metal coverage "
        "matrix is in `extractant_metal_matrix.csv`.")
    add("")
    known = metal_cov.dropna(subset=["metal_symbol"])
    none_at_all = known[known["ionic_radius_coverage"] == "none"]
    partial = known[known["ionic_radius_coverage"] == "partial"]
    rows_with = int(master["ionic_radius_cn8_A"].notna().sum())
    rows_metal = int(master["metal_symbol"].notna().sum())
    add("### Ionic-radius availability")
    add("")
    add("Shannon CN=8 values are supplied only for `(element, oxidation state)` pairs that are "
        "tabulated and confirmed. Everything else is left null with `ionic_radius_status` set, "
        "rather than filled with a plausible-looking guess. Two different gaps are involved and "
        "the coverage table separates them:")
    add("")
    # A metal can lack a radius for two quite different reasons, and lumping
    # them together misattributed the cause for 5 of the 9: Sc(III), for
    # instance, IS tabulated -- its rows simply never record an oxidation state.
    def split_cause(block):
        no_ox, not_tabulated, both = [], [], []
        for symbol in block["metal_symbol"].astype(str):
            rows = master[master["metal_symbol"] == symbol]
            causes = set(rows["ionic_radius_status"].dropna())
            if causes == {"unknown_oxidation_state"}:
                no_ox.append(symbol)
            elif causes == {"requires_curation"}:
                not_tabulated.append(symbol)
            else:
                both.append(symbol)
        return sorted(no_ox), sorted(not_tabulated), sorted(both)

    no_ox, not_tab, mixed = split_cause(none_at_all)
    add(table(["situation", "metals", "which"], [
        ["**no radius on any row** — the ion is not tabulated at CN=8",
         fmt(len(not_tab)), ", ".join(not_tab)],
        ["**no radius on any row** — the archive never recorded an oxidation state, "
         "so the lookup has no key (the value may well be tabulated)",
         fmt(len(no_ox)), ", ".join(no_ox)],
        ["**no radius on any row** — both causes present",
         fmt(len(mixed)), ", ".join(mixed)],
        ["radius on some rows only — oxidation state missing on the rest",
         fmt(len(partial)), ", ".join(sorted(partial["metal_symbol"].astype(str)))],
        ["radius on every row", fmt(int((known["ionic_radius_coverage"] == "all").sum())), ""],
    ]))
    add("")
    add(f"At row level, {rows_with:,} of the {rows_metal:,} records that have a resolved metal "
        f"carry a radius ({100 * rows_with / rows_metal:.0f}%). A radius-dependent model would "
        f"silently drop the other {rows_metal - rows_with:,}.")
    add("")

    # ---------------- missingness ----------------
    add("## 7. Missingness by scientifically meaningful field")
    add("")
    rows = []
    for field in MEANINGFUL_FIELDS:
        if field not in master.columns:
            continue
        missing = int(master[field].isna().sum())
        rows.append([f"`{field}`", fmt(len(master) - missing), fmt(missing),
                     f"{100 * missing / len(master):.1f}%"])
    add(table(["field", "present", "missing", "missing %"], rows))
    add("")
    add("`contact_time_min` and `metal_concentration_M` are the two fields that most often block "
        "a row from modelling — the same two that silently remove 2,976 of 5,992 rows from the "
        "frozen gen10 cohort through its `cond__` completeness gate. Any multi-metal experiment "
        "should treat them as missing-indicator features rather than as hard filters.")
    add("")

    # ---------------- condition ranges ----------------
    add("## 8. Condition ranges")
    add("")
    rows = []
    for field, unit in CONDITION_RANGES:
        values = pd.to_numeric(master[field], errors="coerce").dropna()
        if not len(values):
            continue
        rows.append([f"`{field}`", unit, fmt(len(values)), fmt(values.min()),
                     fmt(values.quantile(0.5)), fmt(values.max())])
    add(table(["field", "unit", "n", "min", "median", "max"], rows))
    add("")

    # ---------------- curves ----------------
    add("## 9. Experimental series and curves")
    add("")
    add(f"{series['series']:,} series and {series['curves_usable']:,} usable curves "
        f"(≥{series['min_curve_points']} points), covering "
        f"{series['records_on_at_least_one_curve']:,} of {len(master):,} records. "
        f"Mean {series['points_per_curve']['mean']:.1f} points per curve, "
        f"median {series['points_per_curve']['median']:.0f}, "
        f"max {series['points_per_curve']['max']}.")
    add("")
    add(table(["curve axis", "usable curves"],
              [[k, fmt(int(v))] for k, v in sorted(series["curves_by_axis"].items(),
                                                   key=lambda kv: -kv[1])]))
    add("")
    add("Definitions mirror `gen8/series.py`: a *series* is one extractant system × one setting of "
        "every categorical condition; a *curve* is a maximal subset of a series in which exactly "
        "one axis varies. Concentration axes are read on the log10 scale because mass action is "
        "linear in log concentration.")
    add("")
    add("**`log_D` is never read to define a series or a curve.** The test suite proves this by "
        "rebuilding membership from a frame whose target has been permuted and requiring "
        "byte-identical output.")
    add("")

    # ---------------- provenance ----------------
    add("## 10. Provenance coverage")
    add("")
    no_ref = int((~master["has_source_reference"]).sum())
    no_doi = int(master["doi_primary"].isna().sum())
    other_only = master[master["doi_primary"].isna()]["reference_other"].map(
        lambda v: v[0] if v is not None and len(v) else None).value_counts()
    add(table(["quantity", "records"], [
        ["with a primary-literature DOI", fmt(int(master["doi_primary"].notna().sum()))],
        ["with a non-DOI reference only", fmt(no_doi)],
        ["**with no reference of any kind**", fmt(no_ref)],
        ["carrying the archive self-citation `10.1021/jacs.5c19738`",
         fmt(int(master["archive_citation_doi"].notna().sum()))],
        ["with a publication title parsed from comments",
         fmt(int(master["publication_title"].notna().sum()))],
        ["with a figure/table location", fmt(int(master["data_location"].notna().sum()))],
    ]))
    add("")
    add("The archive cites itself on every record it exports. That DOI is tracked separately in "
        "`archive_citation_doi` and never fills `doi_primary`, because it references the dataset, "
        "not the measurement.")
    add("")
    add(f"Every record carries *some* reference, but {no_doi:,} have no DOI. Their references are "
        "resolvable documents rather than journal articles, so those rows are traceable but "
        "harder to verify against a published table:")
    add("")
    add(table(["non-DOI reference", "records"],
              [[f"`{k}`", fmt(int(v))] for k, v in other_only.head(6).items()]))
    add("")

    # ---------------- review queue ----------------
    add("## 11. Manual-review queue")
    add("")
    counts = queue.groupby(["review_priority", "category"]).agg(
        cases=("review_id", "count"), records=("n_records", "sum")).reset_index()
    add(table(["priority", "category", "cases", "records"],
              [[int(r["review_priority"]), r["category"], fmt(int(r["cases"])), fmt(int(r["records"]))]
               for _, r in counts.sort_values(["review_priority", "category"]).iterrows()]))
    add("")
    add(f"`manual_review_queue.csv` holds {len(queue):,} cases. Each row names the affected "
        "`canonical_measurement_id`s, explains exactly why the program could not decide, and "
        "carries the raw evidence side by side. No case has an invented resolution.")
    add("")

    # ---------------- accounting ----------------
    add("## 12. Row accounting")
    add("")
    add("Every raw row leaves the pipeline as exactly one disposition; the ledger is asserted to "
        "balance and the stage fails rather than emitting a report that loses rows.")
    add("")
    add(table(["disposition", "records"],
              [[k, fmt(int(v))] for k, v in accounting["record_disposition"].items()]))
    add("")
    add(f"Sum = {sum(accounting['record_disposition'].values()):,} = "
        f"{accounting['archive_records']:,} archive records. "
        f"All {accounting['raw_rows']:,} raw rows map to a record, and every record is reachable "
        "from at least one raw row.")
    add("")

    # ---------------- reproducibility ----------------
    add("## 13. Reproducibility")
    add("")
    add(f"- RDKit `{norm['rdkit_version']}`; canonicalisation failures are preserved with their raw "
        "string and reported, never dropped.")
    add("- No wall-clock, RNG or hash-seed dependence. Group ids are assigned by sorted identity "
        "hash, and the canonical representative of a group is the lowest numeric archive id.")
    add("- `python scripts/run_all.py --verify-determinism` runs the pipeline twice and requires "
        "byte-identical artifacts.")
    add("- `raw/` is checksummed in `reports/raw_checksums.sha256` and set read-only; every "
        "artifact is hashed in `manifest.json`.")
    add("")

    (REPORTS_DIR / "quality_report.md").write_text("\n".join(lines) + "\n")
    print(f"quality_report.md written ({len(lines)} lines)")


if __name__ == "__main__":
    main()
