"""Script 2 of DESIGN.md section 11: recompute the data facts and freeze the cohorts.

Reads the database written by ``g18_build_db.py`` (``systems/corpus_records.csv``,
``series.csv``, ``duplicates.csv``, ``exclusions.csv``, ``registry.json`` and
``results/audit/ingest_audit.json``) and writes ``DATA_AUDIT.md`` with the tables of
DESIGN.md section 0.2 recomputed, the E1 and E2 cohort lists of PRE_REGISTRATION.md
section 2, the duplicate tiers and the replicate floor, plus ``results/audit/audit.json``
(machine-readable counts; ``tests/test_ingest.py`` compares them with the ingest audit),
``e1_cohort.csv``, ``e1_groups.csv``, ``e2_cohort.csv``, ``cohort_sha256.txt`` and
``manifest_audit.json``.  Every count that differs from DESIGN.md 0.2 or PRE_REGISTRATION.md
section 2 is listed in the discrepancy table; the audit is the frozen source.

Usage (from the repository root):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_audit.py [--systems-dir systems]

No wall-clock value is written.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import paths  # noqa: E402
from gen18proc.systems import DEFAULT_BAND, load_registry  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g18_build_db import write_manifest  # noqa: E402

PRIMARY_BAND = DEFAULT_BAND          # "20-30C" (PRE_REGISTRATION.md section 2)
TODGA_SYS = "sys_5cb78e5000d40860"
LOADING_ACTIVE_MIN_RANGE = 0.3
SASAKI_SERIES = "ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1"

#: Panel-verified numbers (DESIGN.md 0.2 / PRE_REGISTRATION.md section 2) the audit compares to.
EXPECTED: dict[str, tuple[object, str]] = {
    "rows_in": (5992, "DESIGN 0.2"), "todga_name_mismatch_rows": (129, "DESIGN 0.2"),
    "sentinel_rows": (3, "DESIGN 0.2"), "rows_after_quarantine": (5860, "DESIGN 0.2"),
    "n_publications": (105, "DESIGN 0.2"), "n_systems_corpus": (287, "DESIGN 0.2"),
    "n_loading_series_publication_aware": (10, "DESIGN 0.2 / PRE_REG 2"),
    "n_loading_series_publication_blind": (11, "DESIGN 0.2"),
    "n_unit_slip_rows": (7, "DESIGN 0.2 / PRE_REG 0"),
    "n_tied_d_groups": (130, "DESIGN 0.2"), "n_tied_d_rows": (324, "DESIGN 0.2"),
    "n_tied_d_publications": (24, "DESIGN 0.2"),
    "replicate_groups": (282, "DESIGN 0.2"), "replicate_sd_median": (0.225, "DESIGN 0.2"),
    "fittable_groups_primary_band": (233, "DESIGN 0.2 / PRE_REG 2"),
    "e1_groups": (59, "DESIGN 0.2 / PRE_REG 2"), "e1_systems": (14, "DESIGN 0.2 / PRE_REG 2"),
    "todga_rows": (514, "DESIGN 0.2"), "todga_publications": (28, "DESIGN 0.2"),
    "todga_metals": (14, "DESIGN 0.2"), "todga_pr_rows": (28, "DESIGN 0.2"),
    "todga_nd_rows": (66, "DESIGN 0.2"), "todga_pubs_with_pr_and_nd": (8, "DESIGN 0.2"),
    "todga_e1_groups": (42, "PRE_REG 2 (sys_5cb78e5000d40860: 42 groups over 18 publications)"),
    "todga_e1_max_publications": (18, "PRE_REG 2 (largest single group of the system)"),
    "todga_named_e1_groups_all_systems": (42, "PRE_REG 2 read per extractant name"),
    "loading_active_series": (7, "PRE_REG 2 (6 series + the Ce series, included)"),
    "n_nan_metal_rows": (1532, "DESIGN 0.2 (pre-quarantine count)"),
}
#: Differences that are a matter of reading, with the explanation printed beside them.
EXPLAINED: dict[str, str] = {
    "todga_e1_groups": "42 is the sum over the four TODGA-named systems; the system key gives "
                       "14 for sys_5cb78e5000d40860 (addenda/WB1.md A9)",
    "n_nan_metal_rows": "1532 is the pre-quarantine count; the quarantine removes 125 of them",
}


def sha(path: Path) -> str:
    return paths.sha256_of(path)


def fittable_table(rec: pd.DataFrame, *, band: str | None) -> pd.DataFrame:
    """Aggregate fit-eligible rows per replicate group; one row per (system, metal) with the
    number of aggregated points, distinct levels and publications (PRE_REGISTRATION.md 2)."""
    r = rec[rec["fit_eligible"]]
    if band is not None:
        r = r[r["temperature_band"] == band]
    agg = r.groupby("replicate_group_id").agg(
        system_id=("system_id", "first"), metal=("metal", "first"),
        publication_id=("publication_id", "first"),
        log_acid=("acid_nominal_M", lambda s: round(math.log10(float(s.iloc[0])), 9)),
        log_ligand=("ligand_M", lambda s: round(math.log10(float(s.iloc[0])), 9)),
        n_rep=("record_id", "size"))
    rows = []
    for (sid, metal), g in agg.groupby(["system_id", "metal"], sort=True):
        rows.append({"system_id": sid, "metal": metal, "n_points": int(len(g)),
                     "n_rows": int(g["n_rep"].sum()),
                     "n_acid_levels": int(g["log_acid"].nunique()),
                     "n_ligand_levels": int(g["log_ligand"].nunique()),
                     "n_publications": int(g["publication_id"].nunique()),
                     "publications": ";".join(sorted(g["publication_id"].unique()))})
    t = pd.DataFrame(rows)
    t["fittable"] = (t["n_points"] >= 6) & ((t["n_acid_levels"] >= 3) | (t["n_ligand_levels"] >= 3))
    t["e1"] = t["fittable"] & (t["n_publications"] >= 2)
    return t


def md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if math.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rng(v: list[float], fmt: str = ".4g") -> str:
    return f"{v[0]:{fmt}}-{v[1]:{fmt}}"


def compute_counts(sdir: Path, out: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Every count of the audit plus the tables written beside DATA_AUDIT.md."""
    rec = pd.read_csv(sdir / "corpus_records.csv",
                      dtype={"fit_ineligible_reason": object, "duplicate_flag": object,
                             "loading_series_id": object})
    series = pd.read_csv(sdir / "series.csv")
    dup = pd.read_csv(sdir / "duplicates.csv")
    excl = pd.read_csv(sdir / "exclusions.csv")
    reg = load_registry(sdir)
    with open(out / "ingest_audit.json", encoding="utf-8") as fh:
        ingest = json.load(fh)

    c: dict = {"bundle_sha256": ingest["bundle_sha256"]}
    # ---- section 0.2 rows ------------------------------------------------------------------
    c["rows_in"] = ingest["rows_in"]
    c["todga_name_mismatch_rows"] = int((excl["reason"] == "todga_name_mismatch").sum())
    c["sentinel_rows"] = int((excl["reason"] == "sentinel_logD_le_-6").sum())
    c["rows_after_quarantine"] = int(len(rec))
    c["n_records"] = int(len(rec))
    c["n_publications"] = int(rec["publication_id"].nunique())
    c["n_systems_corpus"] = int(rec["system_id"].nunique())
    c["n_systems_literature"] = int((reg["origin"] == "literature").sum())
    c["n_systems"] = int(len(reg))
    per_sys = rec.groupby("system_id")[["synergists", "modifiers"]].first().fillna("")
    c["n_two_ligand_systems"] = int((per_sys["synergists"] != "").sum())
    c["n_systems_with_modifiers"] = int((per_sys["modifiers"] != "").sum())
    aware = series[series["definition"] == "publication_aware"].reset_index(drop=True)
    blind = series[series["definition"] == "publication_blind"].reset_index(drop=True)
    c["n_loading_series_publication_aware"] = int(len(aware))
    c["n_loading_series_publication_blind"] = int(len(blind))
    aware["loading_active"] = aware["log_d_range"] >= LOADING_ACTIVE_MIN_RANGE
    c["loading_active_series"] = int(aware["loading_active"].sum())
    slip = rec[rec["duplicate_flag"] == "UNIT_SLIP_DUPLICATE"]
    c["n_unit_slip_rows"] = int(len(slip))
    c["n_unit_slip_rows_pub_0e7f3e0563"] = int((slip["publication_id"] == "pub_0e7f3e0563").sum())
    c["n_unit_slip_rows_3M_side"] = int((slip["acid_nominal_M"] == 3.0).sum())
    c["n_tied_d_groups"] = int(dup["group_id"].nunique())
    c["n_tied_d_rows"] = int(len(dup))
    c["n_tied_d_publications"] = int(dup["publication_id"].nunique())
    tiers = dup.groupby("group_id")["tier"].first()
    c["n_unit_slip_groups"] = int((tiers == "unit_slip").sum())
    c["n_tied_d_tier_groups"] = int((tiers == "tied_d").sum())
    c["n_tied_d_tier_rows"] = int((dup["tier"] == "tied_d").sum())
    c["n_tied_d_flag_rows"] = int((rec["duplicate_flag"] == "TIED_D").sum())
    grp = rec.groupby("replicate_group_id")["log_d"].agg(["size", "std"])
    multi = grp[grp["size"] >= 2]
    c["replicate_groups"] = int(len(multi))
    c["replicate_sd_median"] = float(multi["std"].median())
    grp_e = rec[rec["fit_eligible"]].groupby("replicate_group_id")["log_d"].agg(["size", "std"])
    c["replicate_sd_median_fit_eligible"] = float(grp_e[grp_e["size"] >= 2]["std"].median())
    c["n_nan_metal_rows"] = int(rec["metal_initial_mM"].isna().sum())
    c["n_nan_temperature_rows"] = int(rec["temperature_C"].isna().sum())
    bands = rec["temperature_band"].value_counts().sort_index()
    c["band_counts"] = {str(k): int(v) for k, v in bands.items()}
    reasons = rec["fit_ineligible_reason"].value_counts()
    c["fit_ineligible_reasons"] = {str(k): int(v) for k, v in reasons.items()}
    c["rows_fit_ineligible"] = int((~rec["fit_eligible"]).sum())
    c["rows_fit_eligible"] = int(rec["fit_eligible"].sum())

    # ---- E1 cohort -------------------------------------------------------------------------
    ft_band = fittable_table(rec, band=PRIMARY_BAND)
    ft_all = fittable_table(rec, band=None)
    c["fittable_groups_primary_band"] = int(ft_band["fittable"].sum())
    c["fittable_groups_all_bands"] = int(ft_all["fittable"].sum())
    e1 = ft_band[ft_band["e1"]].copy()
    c["e1_groups"] = int(len(e1))
    c["e1_systems"] = int(e1["system_id"].nunique())
    c["e1_groups_all_bands"] = int(ft_all["e1"].sum())
    c["e1_systems_all_bands"] = int(ft_all.loc[ft_all["e1"], "system_id"].nunique())
    names = reg.set_index("system_id")
    e1_sys = e1.groupby("system_id").agg(
        n_groups=("metal", "size"), metals=("metal", ";".join),
        max_publications=("n_publications", "max"),
        publications=("publications", lambda s: len(set(";".join(s).split(";"))))).reset_index()
    e1_sys["name"] = e1_sys["system_id"].map(names["name"])
    e1_sys["ligand"] = e1_sys["system_id"].map(names["ligands"])
    e1_sys["n_records"] = e1_sys["system_id"].map(names["n_records"])
    e1_sys = e1_sys[["system_id", "ligand", "name", "n_groups", "publications",
                     "max_publications", "n_records", "metals"]]
    e1_sys = e1_sys.sort_values("system_id").reset_index(drop=True)
    c["e1_system_ids"] = list(e1_sys["system_id"])
    c["e1_groups_per_system"] = {r.system_id: int(r.n_groups) for r in e1_sys.itertuples()}
    c["e1_publications_per_system"] = {r.system_id: int(r.publications)
                                       for r in e1_sys.itertuples()}
    c["e1_max_publications_per_system"] = {r.system_id: int(r.max_publications)
                                           for r in e1_sys.itertuples()}
    by_ligand = e1_sys.groupby("ligand").agg(
        n_systems=("system_id", "size"), n_groups=("n_groups", "sum"),
        max_publications=("max_publications", "max")).reset_index()
    todga_e1 = e1_sys[e1_sys["system_id"] == TODGA_SYS]
    c["todga_e1_groups"] = int(todga_e1["n_groups"].iloc[0]) if len(todga_e1) else 0
    c["todga_e1_publications_union"] = (int(todga_e1["publications"].iloc[0])
                                        if len(todga_e1) else 0)
    c["todga_e1_max_publications"] = (int(todga_e1["max_publications"].iloc[0])
                                      if len(todga_e1) else 0)
    c["todga_named_e1_groups_all_systems"] = int(
        by_ligand.loc[by_ligand["ligand"] == "TODGA", "n_groups"].sum())
    c["r1_wins_needed"] = int(math.ceil(0.6 * c["e1_systems"]))

    # ---- E2 cohort -------------------------------------------------------------------------
    e2 = aware[["loading_series_id", "system_id", "ligand_name", "metal", "publication_id",
                "acid_nominal_M", "ligand_M", "n_points", "mM_min", "mM_max", "log_d_tracer",
                "log_d_min", "log_d_max", "log_d_range", "loading_active", "tracer_record_id"]]
    c["e2_series_ids"] = list(aware["loading_series_id"])
    c["e2_blind_series_ids"] = list(blind["loading_series_id"])
    c["loading_active_series_ids"] = list(aware.loc[aware["loading_active"], "loading_series_id"])

    # ---- TODGA system facts ----------------------------------------------------------------
    t = rec[rec["system_id"] == TODGA_SYS]
    c["todga_rows"] = int(len(t))
    c["todga_publications"] = int(t["publication_id"].nunique())
    c["todga_metals"] = int(t["metal"].nunique())
    c["todga_pr_rows"] = int((t["metal"] == "Pr").sum())
    c["todga_nd_rows"] = int((t["metal"] == "Nd").sum())
    pubs_pr = set(t.loc[t["metal"] == "Pr", "publication_id"])
    pubs_nd = set(t.loc[t["metal"] == "Nd", "publication_id"])
    c["todga_pubs_with_pr_and_nd"] = int(len(pubs_pr & pubs_nd))
    c["todga_ranges"] = {
        "acid_M": [float(t["acid_nominal_M"].min()), float(t["acid_nominal_M"].max())],
        "ligand_M": [float(t["ligand_M"].min()), float(t["ligand_M"].max())],
        "metal_mM": [float(t["metal_initial_mM"].min()), float(t["metal_initial_mM"].max())],
        "temperature_C": [float(t["temperature_C"].min()), float(t["temperature_C"].max())]}
    s_rows = rec[rec["loading_series_id"] == SASAKI_SERIES].sort_values("metal_initial_mM")
    c["todga_nd_sasaki_series"] = {
        "n_points": int(len(s_rows)), "record_ids": list(s_rows["record_id"]),
        "mM": [float(s_rows["metal_initial_mM"].min()), float(s_rows["metal_initial_mM"].max())],
        "log_d": [float(s_rows["log_d"].iloc[0]), float(s_rows["log_d"].iloc[-1])],
        "experiment_series_ids": sorted(set(s_rows["experiment_series_id"].dropna()))}

    # ---- discrepancies ---------------------------------------------------------------------
    disc_rows = []
    for key, (exp, src) in EXPECTED.items():
        obs = c.get(key)
        if isinstance(exp, float):
            same = obs is not None and abs(float(obs) - exp) < 5e-4
        else:
            same = obs == exp
        status = "match" if same else ("DIFFERS (explained)" if key in EXPLAINED else "DIFFERS")
        disc_rows.append({"quantity": key, "expected": str(exp), "source": src,
                          "observed": str(obs), "status": status,
                          "explanation": EXPLAINED.get(key, "") if not same else ""})
    disc = pd.DataFrame(disc_rows)
    c["discrepancies"] = [
        f"{r.quantity}: expected {r.expected} ({r.source}), observed {r.observed}"
        + (f" -- {r.explanation}" if r.explanation else "")
        for r in disc.itertuples() if r.status != "match"]

    frozen = {p.name: sha(p) for p in (sdir / "corpus_records.csv", sdir / "series.csv",
                                        sdir / "duplicates.csv", sdir / "exclusions.csv")}
    c["cohort_sha256"] = frozen
    tables = {"e1_sys": e1_sys, "e1": e1, "by_ligand": by_ligand, "e2": e2, "dup": dup,
              "disc": disc}
    return c, tables


def render_markdown(c: dict, tb: dict[str, pd.DataFrame]) -> str:
    dup, e2 = tb["dup"], tb["e2"]
    sas = c["todga_nd_sasaki_series"]
    tr = c["todga_ranges"]
    slip_cols = ["group_id", "safe_exp_id", "publication_id", "metal", "D",
                 "cond__acid_concentration_M", "cond__metal_concentration_mM", "duplicate_flag",
                 "kept", "series_residual"]
    tied_by_pub = (dup[dup["tier"] == "tied_d"].groupby("publication_id").size()
                   .sort_values(ascending=False).head(10).rename("rows").reset_index())
    lines = [
        "# Gen18 data audit (written by `scripts/g18_audit.py`; frozen source of the cohort "
        "counts)",
        "",
        "Recomputed from `systems/corpus_records.csv`, `series.csv`, `duplicates.csv` and "
        "`exclusions.csv` as written by `scripts/g18_build_db.py` (bundle SHA-256 "
        f"`{c['bundle_sha256']}`). Every count below is the value the pre-registration "
        "commits to; where it differs from DESIGN.md 0.2 or PRE_REGISTRATION.md section 2 the "
        "discrepancy table (section 7) says so. No model was fitted before this file was written.",
        "",
        "Regime of every number: cohort = bundle after the gen13 quarantine and the unit-slip "
        "rule; no hold-out (descriptive); averaging unit named per table; parameter status: "
        "none (no parameter exists yet).",
        "",
        "## 1. Data facts of DESIGN.md 0.2, recomputed",
        "",
        "| fact | value |", "|---|---|",
        f"| bundle rows -> after gen13 quarantine | {c['rows_in']} -> "
        f"{c['rows_after_quarantine']} ({c['todga_name_mismatch_rows']} "
        f"TODGA-structure-under-foreign-name rows, {c['sentinel_rows']} sentinel rows at "
        "log D <= -6) |",
        f"| publications (gen6 `publication_id`, none missing) | {c['n_publications']} |",
        f"| systems under the key of section 3.1 | {c['n_systems_corpus']} corpus + "
        f"{c['n_systems_literature']} literature = {c['n_systems']} |",
        "| two-ligand (synergist) systems / systems with alcohol modifiers | "
        f"{c['n_two_ligand_systems']} / {c['n_systems_with_modifiers']} |",
        "| loading series, publication-aware (>= 3 distinct metal concentrations at fixed "
        "system, metal, publication, acid, extractant concentration; after the duplicate rule) "
        f"| **{c['n_loading_series_publication_aware']}** |",
        "| loading series, publication-blind (sensitivity only) | "
        f"{c['n_loading_series_publication_blind']} |",
        f"| loading-active series (log D range >= {LOADING_ACTIVE_MIN_RANGE}; the rising Ce "
        f"series included) | {c['loading_active_series']} |",
        "| unit-slip duplicate rows (`UNIT_SLIP_DUPLICATE`, fit-ineligible) | "
        f"**{c['n_unit_slip_rows']}** ({c['n_unit_slip_rows_pub_0e7f3e0563']} in "
        f"pub_0e7f3e0563, {c['n_unit_slip_rows_3M_side']} on the 3 M side; "
        f"{c['n_unit_slip_groups']} pairs) |",
        "| tied-D groups corpus-wide (same publication, SMILES, metal, D equal to 6 significant "
        f"digits at different conditions) | {c['n_tied_d_groups']} groups, "
        f"{c['n_tied_d_rows']} rows, {c['n_tied_d_publications']} publications; of these the "
        f"unit-slip tier is {c['n_unit_slip_groups']} groups and the `TIED_D` tier "
        f"{c['n_tied_d_tier_groups']} groups / {c['n_tied_d_tier_rows']} rows (kept, flagged) |",
        "| replicate groups (exact 64-column condition key + publication + SMILES + metal, "
        f">= 2 rows) and median within-group sd of log D | {c['replicate_groups']}, "
        f"**{c['replicate_sd_median']:.3f}** (fit-eligible rows only: "
        f"{c['replicate_sd_median_fit_eligible']:.3f}) |",
        f"| fit-ineligible rows | {c['rows_fit_ineligible']} ({c['fit_ineligible_reasons']}) |",
        "| system-metal groups fittable (>= 6 aggregated points, >= 3 distinct levels on log "
        f"acid or log extractant), band {PRIMARY_BAND} (NaN temperature assigned to it) | "
        f"**{c['fittable_groups_primary_band']}** (all bands pooled: "
        f"{c['fittable_groups_all_bands']}) |",
        f"| of these, spanning >= 2 publications (E1) | **{c['e1_groups']} groups in "
        f"{c['e1_systems']} systems** (all bands: {c['e1_groups_all_bands']} / "
        f"{c['e1_systems_all_bands']}) |",
        f"| TODGA / nitrate / aliphatic / no additive (`{TODGA_SYS}`) | {c['todga_rows']} rows, "
        f"{c['todga_publications']} publications, {c['todga_metals']} metals, acid "
        f"{_rng(tr['acid_M'])} M, extractant {_rng(tr['ligand_M'])} M, metal "
        f"{_rng(tr['metal_mM'])} mM, {_rng(tr['temperature_C'], 'g')} C; Pr "
        f"{c['todga_pr_rows']} rows, Nd {c['todga_nd_rows']} rows, "
        f"{c['todga_pubs_with_pr_and_nd']} publications with both |",
        f"| TODGA/Nd 3 M HNO3 / 0.1 M loading series pub_5a68dc5665 | {sas['n_points']} points, "
        f"{_rng(sas['mM'], 'g')} mM, log D {sas['log_d'][0]:.2f} -> {sas['log_d'][1]:.2f}; "
        f"records {', '.join(sas['record_ids'])}; series "
        f"{', '.join(sas['experiment_series_ids'])} |",
        "| rows with NaN metal concentration / NaN temperature (after quarantine) | "
        f"{c['n_nan_metal_rows']} / {c['n_nan_temperature_rows']} |",
        f"| temperature bands (NaN -> {DEFAULT_BAND}) | {c['band_counts']} |",
        "| corpus coverage of the Pr/Nd case | no PC88A, Cyanex 272 or D2EHPA rows (both case "
        "systems are literature entries with 0 records) |",
        "",
        "## 2. E1 cohort (PRE_REGISTRATION.md section 2; unit of the decision = system)",
        "",
        f"Fit-eligible records of band {PRIMARY_BAND}, aggregated per replicate group; a "
        "(system, metal) group is fittable with >= 6 aggregated points and >= 3 distinct levels "
        "on log acid or log extractant; it enters E1 when it spans >= 2 publications. "
        "`publications` is the union over the system's E1 groups; `max_publications` the "
        f"largest single group. R1 (iii) needs M1 to beat B1 in ceil(0.6 x {c['e1_systems']}) "
        f"= {c['r1_wins_needed']} systems.",
        "", md_table(tb["e1_sys"]), "",
        "Aggregated by extractant name (the panel's counts in PRE_REGISTRATION.md section 2 "
        "are per extractant name, not per system key):", "", md_table(tb["by_ligand"]), "",
        "Per-group table: `results/audit/e1_groups.csv`.", "",
        "## 3. E2 cohort (publication-aware loading series; unit = series)", "",
        md_table(e2.drop(columns=["tracer_record_id"])), "",
        f"Loading-active subset (range >= {LOADING_ACTIVE_MIN_RANGE}): "
        f"{', '.join(c['loading_active_series_ids'])}.", "",
        "Publication-blind series (sensitivity X5): "
        + ", ".join(c["e2_blind_series_ids"]) + ".", "",
        "## 4. Duplicates", "",
        "Unit-slip tier (the flagged copy is fit-ineligible; the kept copy carries no flag):",
        "", md_table(dup[dup["tier"] == "unit_slip"][slip_cols]), "",
        f"Tied-D tier: {c['n_tied_d_tier_groups']} groups / {c['n_tied_d_tier_rows']} rows kept "
        "with `TIED_D` (exploratory sensitivity X4 refits without them). Rows per publication "
        "(top 10):", "", md_table(tied_by_pub), "",
        "## 5. Replicate floor", "",
        f"{c['replicate_groups']} replicate groups with >= 2 rows; median within-group sd of "
        f"log D **{c['replicate_sd_median']:.4f}** (all rows) / "
        f"{c['replicate_sd_median_fit_eligible']:.4f} (fit-eligible rows). This is the floor "
        "below which a log D difference has no process consequence (R1 margin 0.05 is inside "
        "it).", "",
        "## 6. Cohort freeze", "",
        "SHA-256 of the frozen tables (`results/audit/cohort_sha256.txt`):", "",
        *[f"- `{v}`  {k}" for k, v in c["cohort_sha256"].items()], "",
        "## 7. Discrepancies against DESIGN.md 0.2 and PRE_REGISTRATION.md section 2", "",
        md_table(tb["disc"]), "",
    ]
    if c["discrepancies"]:
        lines += ["Recorded discrepancies (the audit is the frozen source; the pre-registration "
                  "is not edited, a dated pre-fit addendum records them):", ""]
        lines += [f"- {d}" for d in c["discrepancies"]]
    else:
        lines += ["No discrepancies."]
    lines += [
        "", "## 8. Declared assumptions carried by every derived parameter", "",
        "- `cond__metal_concentration_mM` is the initial aqueous concentration of the row's "
        "single metal (`OA_ASSUMED`; addenda/ORCHESTRATOR_prefit_20260913.md keeps it for every "
        "E2 series).",
        "- O/A = 1 for every corpus loading row.",
        "- nominal acid = equilibrium acidity (`EQUILIBRIUM_ACID_ASSUMED_NOMINAL`); aqueous "
        "nitrate = nominal HNO3.",
        f"- NaN temperature -> band {DEFAULT_BAND} ({c['n_nan_temperature_rows']} rows).",
        "- unit-slip rule applied on the metal-concentration axis only (addenda/WB1.md A3).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    args = ap.parse_args()
    sdir = Path(args.systems_dir)
    if not sdir.is_absolute():
        sdir = paths.G18_ROOT / sdir
    out = paths.RESULTS_AUDIT_DIR
    out.mkdir(parents=True, exist_ok=True)

    c, tb = compute_counts(sdir, out)
    tb["e1_sys"].to_csv(out / "e1_cohort.csv", index=False, lineterminator="\n")
    tb["e1"].sort_values(["system_id", "metal"]).to_csv(out / "e1_groups.csv", index=False,
                                                        lineterminator="\n")
    tb["e2"].to_csv(out / "e2_cohort.csv", index=False, lineterminator="\n")
    with open(out / "cohort_sha256.txt", "w", encoding="utf-8", newline="\n") as fh:
        for k, v in c["cohort_sha256"].items():
            fh.write(f"{v}  {k}\n")
    with open(paths.DATA_AUDIT_MD, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_markdown(c, tb))
    audit_json = out / "audit.json"
    with open(audit_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(c, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")
    outputs = [paths.DATA_AUDIT_MD, audit_json, out / "e1_cohort.csv", out / "e1_groups.csv",
               out / "e2_cohort.csv", out / "cohort_sha256.txt"]
    write_manifest(out, {"corpus_records": sdir / "corpus_records.csv",
                         "series": sdir / "series.csv", "duplicates": sdir / "duplicates.csv",
                         "exclusions": sdir / "exclusions.csv",
                         "ingest_audit": out / "ingest_audit.json"},
                   outputs, 18, vars(args), name="manifest_audit.json",
                   script=Path(__file__).name)
    print(f"DATA_AUDIT.md written: {c['rows_after_quarantine']} rows, {c['n_systems']} systems, "
          f"E1 {c['e1_groups']} groups / {c['e1_systems']} systems, E2 "
          f"{c['n_loading_series_publication_aware']} series, unit-slip rows "
          f"{c['n_unit_slip_rows']}, discrepancies {len(c['discrepancies'])}")
    for d in c["discrepancies"]:
        print("  DIFFERS:", d)
    return 0


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    raise SystemExit(main())
