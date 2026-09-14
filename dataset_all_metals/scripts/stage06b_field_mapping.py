"""Build ``field_mapping.csv``: raw field -> normalized -> model use -> feature.

Every "model use" entry was established by reading the pipeline, not by
guessing from column names.  The two upstream builders are:

* ``lanthanide_dataset_builder/scripts/build_dataset_no3d.py`` -- turns the raw
  ``*_SAFE.csv`` exports into ``dataset.parquet`` (the frozen gen10 table);
* ``lanthanide-ml/scripts/dataset/build_dataset.py`` -- supplies the hardcoded
  ``LANTHANIDE_DESCRIPTORS`` metal block.

``ml_separator`` itself never sees a raw SAE column; it consumes the parquet.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_paths import REPORTS_DIR, ensure_dirs  # noqa: E402

# raw_field, normalized_field (this build), gen10_column, model_use,
# derived_feature, requirement, evidence
ROWS = [
    ("exp_id", "source_record_id / canonical_measurement_id", "safe_exp_id",
     "row identity only; explicitly listed in the builder's PROVENANCE_COLS and excluded from its dedup key",
     "-", "provenance-only",
     "build_dataset_no3d.py:237-243 (PROVENANCE_COLS)"),

    ("Extractant_Name", "extractant_names / extractant_primary_name", "extractant_name",
     "carried as a label; excluded from the content-dedup key and never a feature",
     "-", "provenance-only",
     "build_dataset_no3d.py:237-243; kept in META_AND_PLAN_COLS:253"),

    ("Extractant_SMILES", "extractant_smiles_canonical / extractant_system_key", "canonical_smiles",
     "THE structural input: one feature row per unique canonical SMILES",
     "ecfp_0..ecfp_2047 (ECFP4, Morgan radius=2, fpSize=2048, binary); MolWt, TPSA, NumHDonors, "
     "NumHAcceptors, NumRotatableBonds, NumAromaticRings, NumAliphaticRings, RingCount, "
     "FractionCSP3, MolLogP; extractant_group (chemotype)",
     "required",
     "build_dataset_no3d.py:230 (MORGAN_GENERATOR), :709-713, :740-762 (canonicalisation)"),

    ("Extractant_Concentration_M", "extractant_concentrations_M", "cond__extractant_concentration_M",
     "continuous condition; one of the 5 scalar cond__ columns and a curve axis",
     "log10 concentration for mass-action terms; curve axis 'extractant'", "required",
     "levels.py:61-64 (CONTINUOUS_CONDITION_COLUMNS); gen8/series.py AXIS_LABEL"),

    ("Acid_Name", "acid_names / acid_signature / acid_anion", "cond__acid__*",
     "one-hot categorical condition (9 acid columns in the frozen table)",
     "cond__acid__hno3, __hcl, __h2so4, __hclo4, __citric_acid, __lactic_acid, "
     "__malonic_acid, __tartaric_acid, __hno3_oxalic_acid", "required",
     "dataset.parquet columns; levels.py:240 (categorical conditions define a series)"),

    ("Acid_Concentration_M", "acid_concentration_M", "cond__acid_concentration_M",
     "continuous condition and the dominant curve axis", "log10 acid for mass action; curve axis 'acid'",
     "required", "levels.py:61-64; gen8/series.py"),

    ("Solvent_Name", "solvent_components / solvent_key", "cond__diluent__*",
     "one-hot categorical condition (41 diluent columns), part of the series key",
     "cond__diluent__<slug>, with rare diluents bucketed to cond__diluent__other", "required",
     "dataset.parquet columns; cohort trace CORRECTION 1"),

    ("Metal_Name", "metal_symbol / metal_raw", "metal / metal_symbol",
     "cohort selector and the metal axis; gen10 keeps only the 14 lanthanides",
     "Atomic Number_metal, lanthanide_index, Ionic Radius_metal (hardcoded lookup)", "required",
     "lanthanide-ml/scripts/dataset/build_dataset.py:127-142, :343-348, :446, :639-643; pairs.py:647"),

    ("Metal_Oxidation_state", "metal_oxidation_state", "(not used)",
     "NOT consumed: the frozen metal_ox column is the integer 3 for all 5,992 rows and comes from "
     "the geometry builder, not from this raw field",
     "-", "provenance-only",
     "cohort trace CORRECTION 2 (raw is 'III '/blank; frozen metal_ox is 3 everywhere)"),

    ("Metal_Concentration_mM", "metal_concentration_M", "cond__metal_concentration_mM",
     "continuous condition and a curve axis; canonical unit differs (this build stores M)",
     "log10 metal concentration; curve axis 'metal_concentration'", "optional",
     "levels.py:61-64; gen8/series.py"),

    ("Phase_Modifier_Name", "modifier_name", "cond__additive__*",
     "one-hot categorical condition (9 additive columns), part of the series key",
     "cond__additive__tbp, __1_octanol, __dhoa, __isodecanol, ...", "optional",
     "dataset.parquet columns"),

    ("Phase_Modifier_Concentration_M", "modifier_concentration_M", "(not used)",
     "NOT consumed by gen10: there is no cond__ scalar for modifier concentration",
     "-", "provenance-only", "dataset.parquet has no modifier-concentration scalar"),

    ("Holdback_Agent_Name", "holdback_name", "(not used)",
     "empty in every raw row of this export", "-", "provenance-only", "profiling: 0 / 48,471 filled"),

    ("Holdback_Agent_Concentration_M", "holdback_concentration_M", "(not used)",
     "empty in the column itself; the real values live in comments_description",
     "-", "optional", "profiling: 0 filled; 1,104 values inside comments"),

    ("Extractant_inchi", "(dropped)", "(not used)",
     "never a real InChI: every one of the 48,471 rows holds a dash placeholder "
     "('-' on 46,112 rows, '-,-' on 1,807, empty on 552)", "-", "provenance-only",
     "profiling: 3 distinct values, all placeholders; build_dataset_no3d.py:237-243 lists it as provenance"),

    ("ini_comp", "ini_comp_raw", "(not used)",
     "free-text composition summary; explicitly in PROVENANCE_COLS", "-", "provenance-only",
     "build_dataset_no3d.py:237-243"),

    ("Acid_Concentration_Organic_M", "acid_concentration_organic_M", "(not used)",
     "NOT consumed by gen10; retained here as a genuine condition (6.3% of raw rows)",
     "-", "optional", "no matching cond__ column in dataset.parquet"),

    ("f_Solvent_Name", "(dropped, duplicate)", "(not used)",
     "byte-identical to Solvent_Name on all 48,471 rows", "-", "provenance-only",
     "verified: 0 mismatches"),

    ("f_Metal_Concentration_mM", "(dropped, empty)", "(not used)",
     "empty in every raw row; the claimed fallback path is unreachable", "-", "provenance-only",
     "profiling: 0 filled; cohort trace CORRECTION 3"),

    ("DOI", "doi_primary / doi_source_all / archive_citation_doi", "(not used)",
     "provenance only; a LEAKAGE_KEYWORD in the builder and never a feature", "-", "provenance-only",
     "build_dataset_no3d.py:246-249 (LEAKAGE_KEYWORDS)"),

    ("entry_author", "entry_author", "(not used)", "provenance only", "-", "provenance-only",
     "build_dataset_no3d.py:237-243"),
    ("addition_date", "addition_date", "(not used)", "provenance only", "-", "provenance-only",
     "build_dataset_no3d.py:237-243"),

    ("obsDvalues", "(dropped)", "(not used)", "constant label 'Distribution Ratio'", "-",
     "provenance-only", "profiling: 1 distinct value"),

    ("obsDvaluesValue", "D_value / log_D", "D / log_D",
     "THE TARGET. log_D = log10(D) for finite positive D",
     "log_D; and pairwise log_SF_A_over_B = log_D_A - log_D_B", "required",
     "build_dataset_no3d.py (log_D); pairs.py:869 (pair target)"),

    ("obsTotalEnergiesUnit", "(dropped)", "(not used)",
     "never carries a unit: '-' placeholder on 44,721 rows and empty on 3,750", "-",
     "provenance-only", "profiling: 2 distinct values, both blank/placeholder"),

    ("obsTemp", "(dropped)", "(not used)", "constant label 'Temperature'", "-", "provenance-only",
     "profiling: 1 distinct value"),
    ("obsTempsValue", "temperature_C", "cond__temperature_C",
     "continuous condition and a curve axis", "curve axis 'temperature'", "required",
     "levels.py:61-64; gen8/series.py"),
    ("obsTempUnit", "(dropped)", "(not used)", "constant 'C' on all 45,289 filled rows", "-",
     "provenance-only", "profiling: 1 distinct value"),

    ("Contact_Time_min", "contact_time_min", "cond__contact_time_min",
     "continuous condition and a curve axis", "curve axis 'contact_time'", "optional",
     "levels.py:61-64; gen8/series.py"),
    ("Shaking_Time_min", "shaking_time_min", "geom_cond__shaking_time_bin",
     "binned only into the geometry-environment key, not a model scalar", "-", "optional",
     "dataset.parquet geom_cond__ columns"),

    ("Radiolytic_Dosage_kGy", "(dropped, empty)", "(not used)", "blank or dash-placeholder in every raw row", "-",
     "provenance-only", "profiling: 0 filled"),

    ("comments_obsType", "(dropped)", "(not used)", "constant label 'Comments'", "-",
     "provenance-only", "profiling: 1 distinct value"),
    ("comments_description", "publication_*, complexant_*, holdback_*, nitrate_concentration_M, "
     "data_location, sub_source_file", "(not used)",
     "gen10 discards it entirely; this build parses 16 structured keys out of it, several of "
     "which are genuine experimental conditions",
     "complexant / holdback structures and concentrations; nitrate concentration", "optional",
     "build_dataset_no3d.py:237-243 (provenance); this build: sae_normalize.COMMENT_KEYS"),

    ("volType", "(dropped)", "(not used)", "constant label 'Volume Ratio'", "-", "provenance-only",
     "profiling: 1 distinct value"),
    ("volValue", "phase_ratio_org_aq", "geom_cond__phase_ratio_bin",
     "binned only into the geometry-environment key, not a model scalar", "-", "optional",
     "dataset.parquet geom_cond__ columns"),

    ("thirdType", "(dropped, empty)", "(not used)", "blank or dash-placeholder in every raw row", "-", "provenance-only",
     "profiling: 0 filled"),
    ("thirdValue", "(dropped, empty)", "(not used)", "blank or dash-placeholder in every raw row", "-", "provenance-only",
     "profiling: 0 filled"),

    ("(derived) source CSV file name", "export_source_files / export_metals_queried", "(not used)",
     "export artefact ONLY. A record measuring Nd appears in 12 different metal files, so the "
     "file name is not evidence of the measured metal",
     "-", "provenance-only", "this build, stage01: verified on all 16,770 records"),
]

COLUMNS = ["raw_field", "normalized_field", "gen10_column", "model_use",
           "derived_feature", "requirement", "evidence"]


def main() -> None:
    ensure_dirs()
    frame = pd.DataFrame(ROWS, columns=COLUMNS)
    frame.to_csv(REPORTS_DIR / "field_mapping.csv", index=False)
    print(f"{len(frame)} field mappings written")
    print(frame["requirement"].value_counts().to_string())


if __name__ == "__main__":
    main()
