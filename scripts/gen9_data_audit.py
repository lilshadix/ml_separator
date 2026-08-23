#!/usr/bin/env python
"""Data audit — the anomalies, and what they cost.

gen8 found one confirmed corruption (DMDPhPDA: one paper entered twice, one copy
mantissa-only, differences exactly 0–4 decades) and one strong suspicion (TWE-24:
the worst ligand in the cohort, sitting ~4 log units above five siblings measured on
the identical grid in the same campaign).  The brief asks gen9 to revisit both, to
sweep for the same failure modes elsewhere, and — importantly — **not to delete
anything**.

Deleting a suspect ligand would silently change the cohort every generation since
gen6 has been scored on, and would make gen9's numbers incomparable with gen7's for
a reason that has nothing to do with gen9.  So this script *flags*, quantifies the
cost, and writes the three sensitivity cohorts as row-id manifests that a downstream
run can apply.

Scans performed:

* **exact-decade offsets** between duplicate condition cells that share a canonical
  SMILES — the DMDPhPDA signature;
* **family-local level outliers**: ligands whose condition-adjusted level sits more
  than three log units from their own chemotype's distribution — the TWE-24
  signature;
* **duplicate canonical SMILES with inconsistent names**, which is how DMDPhPDA
  hid from earlier searches;
* **implausible condition values** (non-positive concentrations, temperatures
  outside 0–200 °C, contact times ≤ 0);
* **contagion**: what each flagged ligand costs its nearest neighbour under a 1-NN
  level lookup, which is how gen8 showed one suspect ligand cost TWE-29 3.9 log units.

``TWE24_STATUS`` is emitted as ``VERIFIED`` / ``SUSPECT`` / ``UNRESOLVED`` with its
evidence.  Retrieving the primary document is outside what this script can do, so the
honest outcome here is ``UNRESOLVED`` with the internal evidence quantified — and it
is reported that way rather than upgraded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import load_cohort  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen9_shape" / "data_audit"
OOF_PATH = REPO_ROOT / "runs" / "gen7_architecture" / "finalists" / "oof_predictions.parquet"
DMDPHPDA_COHORTS = (REPO_ROOT / "runs" / "gen8_architecture" / "case_studies"
                    / "dmdphpda_cohorts.json")
TWE24_SMILES = "CCCCOP(=S)(CP(=S)(OCCCC)OCCCC)OCCCC"
MODEL = "REC_ecfp_plus_recovered"

#: A level this far from its chemotype's own distribution is worth a human look.
LEVEL_OUTLIER_DECADES = 3.0
#: How close a ratio must be to an integer power of ten to count as a decade shift.
DECADE_TOLERANCE = 1e-6


#: A coarser condition bucket than ``condition_id``.  ``condition_id`` encodes the
#: full setting including extractant concentration and diluent, so two ligands
#: measured on "the same acid grid" usually share *no* ``condition_id`` at all —
#: which silently made the leave-one-out adjustment fall back to each ligand's own
#: value and reported every level as exactly zero.  The bucket keeps what a level
#: comparison actually needs held fixed: which metal, and how acidic.
BUCKET_COLUMNS = ("metal_symbol", "cond__acid_concentration_M")


def condition_adjusted_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Each ligand's level, raw and adjusted for the conditions it was measured at.

    Two columns rather than one, because neither alone is honest.

    ``level_raw`` is the median ``log_D``.  It is what gen8's case study compared,
    and it is directly interpretable — but a ligand measured only at 5 M acid looks
    anomalous simply for having been measured there.

    ``level_adjusted`` removes the *leave-one-out* mean of each (metal, acidity)
    bucket, so a ligand cannot define its own correction.  It is ``NaN`` — never
    silently zero — where the bucket contains no other ligand, and
    ``adjusted_coverage`` reports what fraction of a ligand's rows were adjustable.
    A comparison drawn on a ligand with low coverage is a comparison across grids
    and is labelled as one.
    """
    columns = [c for c in BUCKET_COLUMNS if c in frame.columns]
    bucket = pd.Series(
        ["|".join(row) for row in zip(*[
            np.asarray([repr(v) for v in pd.to_numeric(frame[c], errors="coerce")
                       .round(6).to_numpy(dtype=float)], dtype=object)
            if c != "metal_symbol"
            else np.asarray([str(v) for v in frame[c].to_numpy()], dtype=object)
            for c in columns])], index=frame.index)
    work = frame.assign(_bucket=bucket)
    total = work.groupby("_bucket")["log_D"].transform("sum")
    count = work.groupby("_bucket")["log_D"].transform("size")
    ligands_here = work.groupby("_bucket")["extractant"].transform("nunique")
    loo = (total - work["log_D"]) / (count - 1).replace(0, np.nan)
    loo = loo.where(ligands_here > 1)
    work["level_residual"] = work["log_D"] - loo

    out = work.groupby(["extractant", "tanimoto_cluster"]).agg(
        level_raw=("log_D", "median"),
        level_adjusted=("level_residual", "median"),
        adjusted_coverage=("level_residual", lambda s: float(s.notna().mean())),
        n_rows=("log_D", "size"), log_d_max=("log_D", "max"),
        log_d_min=("log_D", "min")).reset_index()
    # The scan runs on the raw level, which is defined for every ligand; the adjusted
    # level travels alongside so a flag can be checked against it.
    out["level"] = out["level_raw"]
    return out


OUTLIER_COLUMNS = ["extractant", "grouping", "group", "level", "family_median",
                   "family_sd", "deviation_decades", "n_family", "n_rows", "reason"]


def family_outliers(levels: pd.DataFrame, frame: pd.DataFrame,
                    *, cut: float = LEVEL_OUTLIER_DECADES) -> pd.DataFrame:
    """Ligands far from the level distribution of the peers they should resemble.

    Three peer definitions, because one is not enough and gen8's own case shows why.
    TWE-24 does **not** surface against its Tanimoto-0.7 chemotype — its chemotype is
    small — but it is glaring against the five thiophosphoryl compounds measured on
    the identical grid in the same campaign.  So the scan runs over the chemotype,
    over a donor-motif family, and over the publication a ligand's rows came from,
    and reports whichever grouping flags it.
    """
    motif = frame.drop_duplicates("extractant").set_index("extractant")
    smiles = motif.index.to_series()
    families = {
        "thiophosphoryl": smiles.str.contains(r"P\(=S\)", regex=True, na=False),
        "phosphoryl": smiles.str.contains(r"P\(=O\)", regex=True, na=False),
        "diglycolamide": smiles.str.contains("C(=O)COCC(=O)", regex=False, na=False),
    }
    work = levels.copy()
    work["motif"] = "other"
    for name, mask in families.items():
        work.loc[work["extractant"].isin(set(smiles[mask])), "motif"] = name

    doi_column = "nuisance__doi" if "nuisance__doi" in frame.columns else None
    if doi_column is not None:
        primary_doi = (frame.groupby("extractant")[doi_column]
                       .agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ""))
        work["doi"] = work["extractant"].map(primary_doi).fillna("")
    else:
        work["doi"] = ""

    rows = []
    for grouping in ("tanimoto_cluster", "motif", "doi"):
        for group, block in work.groupby(grouping):
            if not str(group) or len(block) < 3:
                continue
            for _, row in block.iterrows():
                others = block[block["extractant"] != row["extractant"]]["level"]
                if len(others) < 2:
                    continue
                centre = float(others.median())
                deviation = float(row["level"]) - centre
                if abs(deviation) >= cut:
                    rows.append({
                        "extractant": row["extractant"], "grouping": grouping,
                        "group": str(group), "level": float(row["level"]),
                        "family_median": centre, "family_sd": float(others.std()),
                        "deviation_decades": deviation, "n_family": int(len(block)),
                        "n_rows": int(row["n_rows"]), "reason": f"level_outlier_vs_{grouping}"})
    if not rows:
        return pd.DataFrame(columns=OUTLIER_COLUMNS)
    return pd.DataFrame.from_records(rows)[OUTLIER_COLUMNS]


#: The condition columns that define a *cell* for the duplicate scan.  Deliberately
#: **not** ``condition_id``: gen8 established that DMDPhPDA's two copies sit in
#: different ``condition_id`` values, so keying on it is exactly how the corruption
#: stayed invisible.  Keying on the measured conditions themselves finds it.
CELL_COLUMNS = ("metal_symbol", "cond__acid_concentration_M",
                "cond__extractant_concentration_M", "cond__temperature_C",
                "cond__metal_concentration_mM")

#: The *core* cell: what a duplicated publication cannot help but share.  gen8's
#: DMDPhPDA copies differ only in ``cond__temperature_C`` — one copy records 25 °C,
#: the other records nothing — so a scan keyed on the full condition set puts them in
#: different cells and finds nothing.  An unreported field is exactly how a duplicate
#: hides, so the scan runs twice: once on the full cell, once on the core.
CORE_CELL_COLUMNS = ("metal_symbol", "cond__acid_concentration_M",
                     "cond__extractant_concentration_M")

DUPLICATE_COLUMNS = ["extractant", "scope", "cell", "metal_symbol", "n_values", "gap",
                     "nearest_integer", "distance_from_integer", "is_exact_decade"]


def _cell_key(frame: pd.DataFrame, columns, extras) -> pd.Series:
    """A stable string key over condition columns.

    ``.astype(str)`` on a pandas-3 frame can leave real floats in an object array —
    the same trap ``gen8.series._group_key`` documents — so every part goes through
    an explicit ``repr`` over a float array.
    """
    parts: list[np.ndarray] = []
    for column in columns:
        if column == "metal_symbol":
            parts.append(np.asarray([str(v) for v in frame[column].to_numpy()], dtype=object))
            continue
        values = pd.to_numeric(frame[column], errors="coerce").round(6).to_numpy(dtype=float)
        parts.append(np.asarray([repr(v) for v in values], dtype=object))
    for group in extras:
        if not group:
            continue
        block = frame[group].to_numpy(dtype=float).round(3)
        parts.append(np.asarray(["".join(repr(v) for v in row) for row in block], dtype=object))
    return pd.Series(["|".join(row) for row in zip(*parts)], index=frame.index)


def duplicate_cell_scan(frame: pd.DataFrame) -> pd.DataFrame:
    """Cells measured twice for the same ligand — and whether the gap is a decade.

    Replicated measurements scatter continuously.  A publication entered twice with
    one copy transcribed mantissa-only lands on integer decades to ten significant
    figures, which is what makes the DMDPhPDA signature findable at all.
    """
    acid = [c for c in frame.columns if c.startswith("cond__acid__")]
    diluent = [c for c in frame.columns if c.startswith("cond__diluent__")]
    scopes = {
        "full_cell": ([c for c in CELL_COLUMNS if c in frame.columns], (acid, diluent)),
        "core_cell": ([c for c in CORE_CELL_COLUMNS if c in frame.columns], (acid,)),
    }
    rows = []
    for scope, (columns, extras) in scopes.items():
        work = frame.assign(_cell=_cell_key(frame, columns, extras))
        for (extractant, cell), block in work.groupby(["extractant", "_cell"], sort=True):
            if len(block) < 2:
                continue
            array = np.sort(block["log_D"].to_numpy(dtype=float))
            for gap in np.diff(array):
                nearest = float(np.round(gap))
                rows.append({
                    "extractant": extractant, "scope": scope, "cell": cell,
                    "metal_symbol": str(block["metal_symbol"].iloc[0]),
                    "n_values": int(len(array)), "gap": float(gap), "nearest_integer": nearest,
                    "distance_from_integer": float(abs(gap - nearest)),
                    "is_exact_decade": bool(abs(gap - nearest) < DECADE_TOLERANCE
                                            and nearest != 0)})
    if not rows:
        return pd.DataFrame(columns=DUPLICATE_COLUMNS)
    return pd.DataFrame.from_records(rows)[DUPLICATE_COLUMNS]


def name_mismatch_scan(frame: pd.DataFrame) -> dict:
    """Rows whose recorded extractant name does not match the structure modelled.

    gen7 recovered this column and found it marks *unmodelled second species*: a
    synergistic system in which two extractants were present and only one structure
    was captured.  The model is then asked to predict a mixture from one component's
    fingerprint, which is not a modelling failure but a labelling one — and it is
    worth a sensitivity cohort rather than a deletion.
    """
    if "rec__name_mismatch" not in frame.columns:
        return {"available": False}
    flagged = frame["rec__name_mismatch"].fillna(0) > 0
    per_ligand = frame.loc[flagged].groupby("extractant").size().sort_values(ascending=False)
    return {
        "available": True,
        "n_rows": int(flagged.sum()),
        "share_of_cohort": float(flagged.mean()),
        "n_extractants_affected": int(per_ligand.size),
        "worst": {str(k): int(v) for k, v in per_ligand.head(8).items()},
    }


def smiles_name_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
    """One canonical SMILES under several names — how DMDPhPDA hid."""
    if "nuisance__extractant_name" not in frame.columns:
        return pd.DataFrame(columns=["extractant", "names", "n_names", "n_rows"])
    rows = []
    for smiles, block in frame.groupby("extractant"):
        names = sorted({str(n) for n in block["nuisance__extractant_name"].dropna()})
        if len(names) > 1:
            rows.append({"extractant": smiles, "names": " | ".join(names),
                         "n_names": len(names), "n_rows": int(len(block))})
    return pd.DataFrame.from_records(rows)


def implausible_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "acid_concentration_non_positive": (
            "cond__acid_concentration_M", lambda s: s.notna() & (s <= 0)),
        "extractant_concentration_non_positive": (
            "cond__extractant_concentration_M", lambda s: s.notna() & (s <= 0)),
        "temperature_out_of_range": (
            "cond__temperature_C", lambda s: s.notna() & ((s < 0) | (s > 200))),
        "contact_time_non_positive": (
            "cond__contact_time_min", lambda s: s.notna() & (s <= 0)),
        "acid_concentration_implausibly_high": (
            "cond__acid_concentration_M", lambda s: s.notna() & (s > 20)),
    }
    rows = []
    for name, (column, test) in checks.items():
        if column not in frame.columns:
            continue
        mask = test(pd.to_numeric(frame[column], errors="coerce"))
        rows.append({"check": name, "column": column, "n_rows": int(mask.sum()),
                     "extractants": int(frame.loc[mask, "extractant"].nunique())})
    return pd.DataFrame.from_records(rows)


def contagion_cost(frame: pd.DataFrame, levels: pd.DataFrame, suspects) -> pd.DataFrame:
    """What each suspect costs its nearest neighbour under a 1-NN level lookup.

    The same benchmark gen8 used to show TWE-24 costs TWE-29 3.9 log units: predict
    each ligand's level as its nearest ECFP neighbour's, with and without the suspect
    in the reference set.
    """
    from lanthanide_separation.gen6.chemistry import ChemistryMap

    cache = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "chemistry_map.parquet"
    if not cache.exists():
        return pd.DataFrame()
    chemistry = ChemistryMap.from_parquet(cache)
    level_of = levels.set_index("extractant")["level"].to_dict()
    everyone = [x for x in level_of if x in set(chemistry.extractants)]

    def lookup_error(reference: list[str]) -> pd.Series:
        table = chemistry.nearest_neighbour(everyone, reference, exclude_self=True)
        table = table.set_index("extractant")
        errors = {}
        for ligand in everyone:
            if ligand not in table.index:
                continue
            neighbour = table.loc[ligand]
            name = neighbour.get("nn_partner")
            if name is None or name not in level_of:
                continue
            errors[ligand] = abs(level_of[ligand] - level_of[name])
        return pd.Series(errors)

    baseline = lookup_error(everyone)
    rows = []
    for suspect in suspects:
        if suspect not in everyone:
            continue
        without = lookup_error([x for x in everyone if x != suspect])
        shared = baseline.index.intersection(without.index).difference([suspect])
        delta = (baseline[shared] - without[shared])
        worst = delta.abs().idxmax() if len(delta) else None
        rows.append({
            "suspect": suspect,
            "n_neighbours_changed": int((delta.abs() > 1e-9).sum()),
            "cohort_mean_level_error_with": float(baseline[shared].mean()),
            "cohort_mean_level_error_without": float(without[shared].mean()),
            "cohort_delta": float(delta.mean()),
            "worst_affected": worst,
            "worst_affected_delta": float(delta[worst]) if worst is not None else np.nan})
    return pd.DataFrame.from_records(rows)


def twe24_evidence(frame: pd.DataFrame, levels: pd.DataFrame) -> dict:
    """The internal case for and against TWE-24, quantified.

    The primary document is not in this repository and cannot be retrieved from here,
    so the verdict this script can honestly reach is ``UNRESOLVED`` — with every
    internal number a human would need in order to settle it.
    """
    present = TWE24_SMILES in set(frame["extractant"])
    if not present:
        return {"status": "ABSENT", "note": "TWE-24 is not in this cohort"}
    block = frame[frame["extractant"] == TWE24_SMILES]
    row = levels[levels["extractant"] == TWE24_SMILES]
    chemotype = str(row["tanimoto_cluster"].iloc[0]) if len(row) else ""
    siblings = levels[(levels["tanimoto_cluster"] == chemotype)
                      & (levels["extractant"] != TWE24_SMILES)]
    # the P=S campaign: every ligand carrying a thiophosphoryl motif
    thio = frame[frame["extractant"].str.contains(r"P\(=S\)", regex=True, na=False)]
    thio_levels = levels[levels["extractant"].isin(set(thio["extractant"]))
                         & (levels["extractant"] != TWE24_SMILES)]
    raw_d = np.power(10.0, block["log_D"].to_numpy(dtype=float))
    evidence = {
        "status": "UNRESOLVED",
        "n_rows": int(len(block)),
        "level_raw": float(row["level_raw"].iloc[0]) if len(row) else float("nan"),
        "level_adjusted": float(row["level_adjusted"].iloc[0]) if len(row) else float("nan"),
        "adjusted_coverage": float(row["adjusted_coverage"].iloc[0]) if len(row) else float("nan"),
        "log_d_values": [float(v) for v in np.sort(block["log_D"].to_numpy(dtype=float))],
        "raw_D_values": [float(v) for v in np.sort(raw_d)],
        "chemotype": chemotype,
        "n_chemotype_siblings": int(len(siblings)),
        "sibling_level_median": float(siblings["level_raw"].median()) if len(siblings) else float("nan"),
        "n_thiophosphoryl_siblings": int(len(thio_levels)),
        "thio_sibling_level_median": float(thio_levels["level_raw"].median()) if len(thio_levels) else float("nan"),
        "thio_sibling_level_max": float(thio_levels["level_raw"].max()) if len(thio_levels) else float("nan"),
        "thio_sibling_levels": sorted(float(v) for v in thio_levels["level_raw"]),
        "decades_above_thio_siblings": (
            float(row["level_raw"].iloc[0] - thio_levels["level_raw"].median())
            if len(row) and len(thio_levels) else float("nan")),
        "shares_grid_with_siblings": bool(
            set(block["condition_id"]) & set(thio[thio["extractant"] != TWE24_SMILES]["condition_id"])),
        "primary_document_retrieved": False,
        "verdict_reason": (
            "The internal evidence is consistent with a ~3-decade transcription error "
            "and equally consistent with real chemistry no descriptor here can see. "
            "Settling it requires the primary document, which is not in this repository. "
            "Reported UNRESOLVED rather than upgraded to SUSPECT-with-a-number, and the "
            "row is NOT removed: deleting it would move the cohort every generation "
            "since gen6 has been scored on."),
    }
    if "nuisance__extractant_name" in block.columns:
        evidence["names"] = sorted({str(n) for n in block["nuisance__extractant_name"].dropna()})
    if "rec__doi" in block.columns:
        evidence["dois"] = sorted({str(d) for d in block["rec__doi"].dropna()})
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--level-cut", type=float, default=LEVEL_OUTLIER_DECADES)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    frame = cohort.frame
    print(f"cohort {frame.shape}  fingerprint {cohort.fingerprint}")

    levels = condition_adjusted_levels(frame)
    levels.to_csv(args.out / "condition_adjusted_levels.csv", index=False)

    outliers = family_outliers(levels, frame, cut=args.level_cut)
    outliers.to_csv(args.out / "level_outliers.csv", index=False)
    print(f"\n--- family-local level outliers (>= {args.level_cut} decades) ---")
    if len(outliers):
        print(outliers.sort_values("deviation_decades", key=abs, ascending=False)
              [["extractant", "grouping", "group", "level", "family_median",
                "deviation_decades", "n_family", "n_rows"]].head(20).to_string(index=False))
    else:
        print("  none")

    duplicates = duplicate_cell_scan(frame)
    duplicates.to_csv(args.out / "duplicate_cells.csv", index=False)
    decades = duplicates[duplicates["is_exact_decade"]] if len(duplicates) else duplicates
    print(f"\n--- duplicate condition cells: {len(duplicates)} gaps, "
          f"{len(decades)} exactly an integer number of decades ---")
    if len(decades):
        print(decades.groupby(["scope", "extractant"]).agg(
            n_gaps=("gap", "size"), decades=("nearest_integer", lambda s: sorted(set(s)))
        ).to_string())

    mismatch = name_mismatch_scan(frame)
    (args.out / "name_mismatch.json").write_text(json.dumps(mismatch, indent=2))
    print("\n--- recorded name does not match the modelled structure ---")
    print(json.dumps(mismatch, indent=2))

    conflicts = smiles_name_conflicts(frame)
    conflicts.to_csv(args.out / "smiles_name_conflicts.csv", index=False)
    print(f"\n--- canonical SMILES carrying more than one name: {len(conflicts)} ---")
    if len(conflicts):
        print(conflicts.to_string(index=False))

    implausible = implausible_conditions(frame)
    implausible.to_csv(args.out / "implausible_conditions.csv", index=False)
    print("\n--- condition plausibility ---")
    print(implausible.to_string(index=False))

    suspects = sorted(set(outliers["extractant"]) | {TWE24_SMILES})
    contagion = contagion_cost(frame, levels, suspects)
    if len(contagion):
        contagion.to_csv(args.out / "contagion.csv", index=False)
        print("\n--- what each suspect costs a 1-NN level lookup ---")
        print(contagion[["suspect", "n_neighbours_changed", "cohort_delta",
                         "worst_affected_delta"]].to_string(index=False))

    twe24 = twe24_evidence(frame, levels)
    (args.out / "twe24.json").write_text(json.dumps(twe24, indent=2, default=str))
    print(f"\n--- TWE24_STATUS = {twe24['status']} ---")
    for key in ("level_raw", "thio_sibling_level_median", "decades_above_thio_siblings",
                "n_thiophosphoryl_siblings", "shares_grid_with_siblings"):
        if key in twe24:
            print(f"  {key}: {twe24[key]}")

    # --- the flag table the brief asks for -------------------------------
    flags = []
    for _, row in outliers.iterrows():
        flags.append({"entity": row["extractant"],
                      "reason": f"level outlier vs {row['grouping']} ({row['group']})",
                      "magnitude": f"{row['deviation_decades']:+.2f} decades vs peer median",
                      "source": "condition-adjusted level, this script",
                      "status": "FLAGGED, retained"})
    if len(decades):
        for (scope, extractant), block in decades.groupby(["scope", "extractant"]):
            flags.append({
                "entity": extractant,
                "reason": f"duplicate {scope} values differing by exact decades",
                "magnitude": f"{len(block)} gaps at {sorted(set(block['nearest_integer']))} decades",
                "source": "duplicate cell scan, this script",
                "status": "FLAGGED, retained (DMDPhPDA cohorts already defined)"})
    if twe24.get("status") != "ABSENT":
        flags.append({
            "entity": "TWE-24", "reason": "level far above same-campaign siblings",
            "magnitude": f"{twe24.get('decades_above_thio_siblings', float('nan')):+.2f} decades",
            "source": "gen8 case study, re-derived here",
            "status": f"TWE24_STATUS={twe24['status']}; primary document not retrievable here"})
    flag_table = pd.DataFrame(flags)
    flag_table.to_csv(args.out / "anomaly_flags.csv", index=False)

    summary = {
        "cohort_fingerprint": cohort.fingerprint, "n_rows": int(len(frame)),
        "n_level_outliers": int(len(outliers)),
        "n_duplicate_gaps": int(len(duplicates)),
        "n_exact_decade_gaps": int(len(decades)),
        "n_smiles_name_conflicts": int(len(conflicts)),
        "name_mismatch": mismatch,
        "twe24_status": twe24["status"],
        "dmdphpda_cohorts_available": DMDPHPDA_COHORTS.exists(),
        "nothing_deleted": True,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
