"""g19_audit_corpus.py -- the corpus audit and coverage tables.

Brief section 28 Phase A items 2-10 and the counts behind item 14, the condition representation of
section 4.3, and figures 1, 2, 4, 5, 6 of section 25.  Nothing here fits or trains a model.

Run from the repository root::

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_audit_corpus.py

Outputs (all under ``generations/gen19_chem_transfer``):

* ``data_audit/dataset_hashes.csv`` -- every archive file, binary + LF digests, match to the archive's
  own manifest / raw checksum list; the frozen bundle against its pinned prefix.
* ``data_audit/columns.csv`` -- every archive column + the ``g19_`` columns: dtype, fill, uniques, role.
* ``data_audit/counts.json`` -- tier/category/class row counts, unique counts (all rows and MODEL
  rows), section 1.4 metadata availability, unit sanity counts, diluent-family rules.
* ``data_audit/matrix_rows_system_x_metal.csv``, ``matrix_pubs_system_x_metal.csv``,
  ``matrix_rows_primary_extractant_x_metal.csv`` and ``data_audit/sparsity.json``.
* ``data_audit/metal_coverage.csv``, ``prnd_coverage.csv``, ``lanthanide_coverage.csv``,
  ``actinide_coverage.csv``, ``condition_coverage.csv``.
* ``figures/F01_observation_matrix.png``, ``F02_density_by_metal.png``, ``F04_lanthanide_coverage.png``,
  ``F05_actinide_coverage.png``, ``F06_condition_coverage.png``.

Conventions.  Matrices and coverage tables use MODEL-tier rows (``g19_tier == "MODEL"``).  A metal whose
oxidation state the archive never recorded is labelled ``"Nd(?)"`` -- it is *not* imputed to +3 -- so
every MODEL row stays in the matrix; ``sparsity.json`` reports the known-state matrix, the matrix with
the ``(?)`` columns, and the symbol-level matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.data import load as g19load  # noqa: E402
from gen19ct.data import normalize as N  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json  # noqa: E402

NAME = "g19_audit_corpus"
LANTHANIDES = ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")
CATEGORY_ORDER = ("lanthanide", "rare_earth_non_lanthanide", "actinide", "transition_metal",
                  "post_transition_metal", "alkaline_earth")
SCHEMA_SECTION_ROLE = {
    "Provenance": "provenance", "Primary experimental information": "primary_experimental",
    "Target": "target", "Derived reference values": "derived_reference",
    "Series / curve structure": "series", "Quality tiers": "quality",
}
G19_PROVENANCE = {"g19_publication_id", "g19_publication_status", "g19_publication_refs", "g19_study_id",
                  "g19_bundle_exp_id"}

# figure styling (dataviz reference palette; categorical slots in fixed order)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
CAT_COLORS = {"lanthanide": "#2a78d6", "actinide": "#eb6834", "alkaline_earth": "#1baf7a",
              "transition_metal": "#eda100", "rare_earth_non_lanthanide": "#e87ba4",
              "post_transition_metal": "#008300"}
SCATTER_COLORS = {"lanthanide": "#2a78d6", "actinide": "#eb6834", "other metal": "#1baf7a"}
CAT_SHORT = {"lanthanide": "lanthanide", "rare_earth_non_lanthanide": "Sc/Y", "actinide": "actinide",
             "transition_metal": "transition", "post_transition_metal": "post-trans.", "alkaline_earth": "alk. earth"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "g19_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DPI = 130


# =========================================================================== helpers
def roman(n: int) -> str:
    return {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}.get(int(n), str(int(n)))


def state_label(metal, ox) -> str | None:
    if N.is_missing(metal):
        return None
    return f"{metal}(?)" if N.is_missing(ox) else f"{metal}({roman(ox)})"


def gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    n = x.size
    return float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1.0) / n)


def mode_text(values) -> tuple[str, int]:
    """Most frequent string (ties alphabetical) and the number of distinct strings."""
    c = Counter(v for v in values if not N.is_missing(v))
    if not c:
        return "NA", 0
    best = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best, len(c)


def stringify(v) -> str:
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, sort_keys=True, default=str)
    if N.is_missing(v):
        return "<NA>"
    return str(v)


def system_id(key: str) -> str:
    return "sys_" + hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:10]


def fill_fraction(series: pd.Series) -> float:
    return float(series.notna().mean()) if len(series) else float("nan")


def tuple_filled(t) -> bool:
    return isinstance(t, tuple) and len(t) > 0 and not all(np.isnan(x) for x in t)


def truncate(s: str, n: int = 34) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "~"


# =========================================================================== 1. hashes
def dataset_hashes() -> pd.DataFrame:
    manifest = json.loads(paths.ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
    recorded: dict[str, tuple[str, int, str]] = {}
    for name, a in manifest.get("artifacts", {}).items():
        recorded[a["workspace_path"]] = (a["sha256"], a.get("bytes"), "manifest.json:artifacts")
    for name, a in manifest.get("raw_inputs", {}).items():
        recorded[f"dataset_all_metals/raw/{name}"] = (a["sha256"], a.get("bytes"), "manifest.json:raw_inputs")
    for name, a in manifest.get("external_inputs", {}).items():
        recorded[f"dataset_all_metals/{name}"] = (a["sha256"], a.get("bytes"), "manifest.json:external_inputs")
    raw_sums: dict[str, str] = {}
    for line in (paths.ARCHIVE_DIR / "reports" / "raw_checksums.sha256").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            raw_sums[f"dataset_all_metals/raw/{parts[-1].lstrip('*')}"] = parts[0]

    rows = []
    files = sorted((p for p in paths.ARCHIVE_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts),
                   key=lambda p: paths.rel(p))
    for p in files:
        rp = paths.rel(p)
        d = paths.digests(p)
        bytes_lf = None
        if d["sha256_lf"] is not None:
            bytes_lf = len(p.read_bytes().replace(b"\r\n", b"\n"))
        rec_sha, rec_bytes, rec_src = recorded.get(rp, (None, None, None))
        m_match = "no_recorded_digest"
        if rec_sha is not None:
            m_match = "binary" if rec_sha == d["sha256"] else ("lf_normalised" if rec_sha == d["sha256_lf"] else "MISMATCH")
        raw_sha = raw_sums.get(rp)
        r_match = "no_recorded_digest"
        if raw_sha is not None:
            r_match = "binary" if raw_sha == d["sha256"] else ("lf_normalised" if raw_sha == d["sha256_lf"] else "MISMATCH")
        recorded_any = rec_sha is not None or raw_sha is not None
        verified = (m_match in ("binary", "lf_normalised") or rec_sha is None) and \
                   (r_match in ("binary", "lf_normalised") or raw_sha is None)
        rows.append({
            "path": rp, "group": p.relative_to(paths.ARCHIVE_DIR).parts[0] if len(p.relative_to(paths.ARCHIVE_DIR).parts) > 1 else "(root)",
            "bytes": d["bytes"], "bytes_lf": bytes_lf, "sha256": d["sha256"], "sha256_lf": d["sha256_lf"],
            "recorded_sha256": rec_sha, "recorded_bytes": rec_bytes, "recorded_source": rec_src,
            "recorded_bytes_match": (pd.NA if rec_bytes is None else
                                     ("binary" if rec_bytes == d["bytes"] else
                                      ("lf_normalised" if rec_bytes == bytes_lf else "MISMATCH"))),
            "manifest_match": m_match, "raw_checksum_sha256": raw_sha, "raw_checksum_match": r_match,
            "has_recorded_digest": recorded_any, "verified": verified if recorded_any else pd.NA,
            "is_headline_dataset_hash": rp == paths.rel(paths.ARCHIVE_MASTER),
        })
    b = paths.digests(paths.BUNDLE_DATASET)
    ok = str(b["sha256"]).startswith(paths.BUNDLE_DATASET_SHA256_PREFIX)
    rows.append({
        "path": paths.rel(paths.BUNDLE_DATASET), "group": "bundle", "bytes": b["bytes"], "bytes_lf": None,
        "sha256": b["sha256"], "sha256_lf": b["sha256_lf"],
        "recorded_sha256": paths.BUNDLE_DATASET_SHA256_PREFIX + " (prefix)", "recorded_bytes": None,
        "recorded_source": "gen3_protocol.json / gen13sep.paths (pinned prefix)", "recorded_bytes_match": pd.NA,
        "manifest_match": "binary_prefix" if ok else "MISMATCH", "raw_checksum_sha256": None,
        "raw_checksum_match": "no_recorded_digest", "has_recorded_digest": True, "verified": ok,
        "is_headline_dataset_hash": False,
    })
    out = pd.DataFrame(rows)
    for c in ("bytes_lf", "recorded_bytes"):
        out[c] = pd.array(out[c], dtype="Int64")
    return out


# =========================================================================== 2. columns
def parse_schema_md() -> dict[str, dict]:
    section, out = None, {}
    for line in (paths.ARCHIVE_DIR / "reports" / "schema.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = re.match(r"^\|\s*`([^`]+)`\s*\|\s*([^|]*)\|\s*([^|]*)\|\s*(.*)\|\s*$", line)
        if m and section:
            role = next((r for k, r in SCHEMA_SECTION_ROLE.items() if section.startswith(k)), "unmapped")
            nn = m.group(3).strip().replace(",", "")
            out[m.group(1)] = {"schema_section": section, "schema_role": role, "schema_dtype": m.group(2).strip(),
                               "schema_nonnull": int(nn) if nn.isdigit() else None}
    return out


#: Roles for archive columns that schema.md does not list (checked against the 134-column parquet).
#: Every entry is INFERRED from the column name and content, not from archive documentation.
UNLISTED_COLUMN_ROLE = {
    "n_complexants": ("primary_experimental", "INFERRED: count of aqueous_complexant components"),
    "complexant_names": ("primary_experimental", "INFERRED: list form of complexant_name"),
    "complexant_smiles_all": ("primary_experimental", "INFERRED: list form of complexant_smiles_canonical"),
    "complexant_concentrations_M": ("primary_experimental", "INFERRED: list form of complexant_concentration_M"),
    "complexant_signature": ("primary_experimental", "INFERRED: name@concentration join of the complexants"),
    "conflicting_component_names": ("quality", "INFERRED: names that disagree for one structure/component"),
    "conflicting_component_detail": ("quality", "INFERRED: detail of conflicting_component_names"),
    "has_suspect_flag": ("quality", "INFERRED: boolean summary of the flags column"),
}


def _is_list(v) -> bool:
    return isinstance(v, (np.ndarray, list, tuple))


def columns_table(df: pd.DataFrame, is_model: np.ndarray, archive_columns: list[str]) -> pd.DataFrame:
    """One row per column.  ``non_null`` is pandas ``notna`` (comparable with schema.md); ``filled`` is
    stricter: not NA, not an empty/whitespace string, not an empty list.  ``n_unique`` counts distinct
    *filled* values (lists stringified), so NA and ``[]`` are never counted as a value."""
    schema = parse_schema_md()
    prov = set(g19load.PROVENANCE_COLUMNS)
    rows = []
    for c in df.columns:
        s = df[c]
        list_valued = bool(s.map(_is_list).any())
        nonnull = int(s.map(lambda v: True if _is_list(v) else not (v is None or v is pd.NA or
                                                                    (isinstance(v, float) and np.isnan(v)))).sum())
        nonempty = int(s.map(lambda v: _is_list(v) and len(v) > 0).sum()) if list_valued else None
        filled_mask = s.map(lambda v: (len(v) > 0) if _is_list(v) else not N.is_missing(v)).astype(bool)
        n_empty_string = int(s.map(lambda v: isinstance(v, str) and v.strip() == "").sum())
        n_unique = int(s[filled_mask].map(stringify).nunique())
        sch = schema.get(c, {})
        in_archive = c in archive_columns
        if sch:
            role, basis = sch["schema_role"], f"schema.md section: {sch['schema_section']}"
        elif in_archive:
            role, basis = UNLISTED_COLUMN_ROLE.get(c, ("unmapped", "not in schema.md; no inference made"))
        else:
            role = "provenance" if c in G19_PROVENANCE else "g19_derived"
            basis = "gen19ct.data.load derived column"
        rows.append({
            "column": c, "source": "archive" if in_archive else "g19_derived (gen19ct.data.load)",
            "schema_section": sch.get("schema_section", "g19_derived" if not in_archive else "NOT_IN_SCHEMA_MD"),
            "schema_role": role, "schema_role_basis": basis,
            "in_PROVENANCE_COLUMNS": c in prov, "dtype": str(s.dtype), "list_valued": list_valued,
            "non_null": nonnull, "non_empty_lists": nonempty, "n_empty_string": n_empty_string,
            "n_filled_all": int(filled_mask.sum()), "n_filled_model": int(filled_mask[is_model].sum()),
            "fill_fraction_all": float(filled_mask.mean()),
            "fill_fraction_model": float(filled_mask[is_model].mean()),
            "n_unique": n_unique,
            "schema_md_dtype": sch.get("schema_dtype"), "schema_md_nonnull": sch.get("schema_nonnull"),
            "nonnull_matches_schema_md": (sch.get("schema_nonnull") == nonnull) if sch else pd.NA,
        })
    out = pd.DataFrame(rows)
    out["non_empty_lists"] = pd.array(out["non_empty_lists"], dtype="Int64")
    out["schema_md_nonnull"] = pd.array(out["schema_md_nonnull"], dtype="Int64")
    return out


# =========================================================================== 3. counts
METADATA_14 = [  # brief section 1.4 -> archive columns (ABSENT when none)
    ("publication/source ID", ["g19_publication_id"], "derived by gen19ct.data.load from doi_all/reference_other"),
    ("DOI", ["doi_primary"], "non-DOI rows carry a CORDIS/INIS/thesis reference instead"),
    ("table/figure/page", ["data_location"], "parsed from comments; absent on most rows"),
    ("metal", ["metal_symbol"], ""),
    ("oxidation state", ["metal_oxidation_state"], "never imputed"),
    ("extractant", ["extractant_primary_name"], "names are untrustworthy (TODGA key carries 22 names)"),
    ("extractant family", [], "no family column in the archive; must be assigned from structure"),
    ("extractant structure identifier", ["extractant_system_key"], "canonical SMILES join"),
    ("diluent", ["solvent_key"], ""),
    ("acid", ["acid_primary"], ""),
    ("aqueous composition", ["aqueous_phase_metals_declared", "complexant_name", "holdback_smiles_canonical",
                             "nitrate_concentration_M"], "any of these; 'none present' cannot be told from 'not recorded'"),
    ("extractant concentration", ["extractant_primary_concentration_M"], ""),
    ("initial metal concentration", ["metal_concentration_M"], "initial vs equilibrium not stated"),
    ("equilibrium/final pH", [], "no pH column; acid molarity is not pH"),
    ("acidity", ["acid_concentration_M"], "initial vs equilibrium acidity not stated"),
    ("O/A", ["phase_ratio_org_aq"], ""),
    ("temperature", ["temperature_C"], ""),
    ("contact time", ["contact_time_min", "shaking_time_min"], "either column; never both on one row"),
    ("saponification", [], "no saponification field"),
    ("modifiers", ["modifier_name"], "'no modifier' cannot be told from 'not recorded'"),
    ("complexants", ["complexant_name"], "'no complexant' cannot be told from 'not recorded'"),
    ("loading", [], "no organic-loading field"),
    ("measured D", ["log_D"], "log10 of D_value; null when D missing or non-positive"),
    ("D directly reported vs reconstructed", [], "no such flag"),
    ("uncertainty if reported", [], "no uncertainty column"),
    ("data-status / quality flag", ["model_readiness"], "also flags, duplicate_class, in_value_conflict"),
]


def uniques(sub: pd.DataFrame, meta: pd.DataFrame, cv: pd.DataFrame) -> dict:
    names = set()
    for arr in sub["extractant_names"]:
        names.update(str(x) for x in arr if not N.is_missing(x))
    comps = set()
    for arr in sub["solvent_components"]:
        comps.update(str(x) for x in arr)
    return {
        "metals_symbol": int(sub["metal_symbol"].nunique()),
        "metal_states_known_ox": int(sub["g19_metal_state"].nunique()),
        "metal_states_incl_unknown_ox": int(meta.loc[sub.index, "state_label"].nunique()),
        "metals_with_unknown_ox_rows": int(sub.loc[sub["metal_symbol"].notna() & sub["metal_oxidation_state"].isna(),
                                                   "metal_symbol"].nunique()),
        "oxidation_states": int(sub["metal_oxidation_state"].nunique()),
        "metal_categories": int(sub["metal_category"].nunique()),
        "extractant_system_keys": int(sub["extractant_system_key"].nunique()),
        "extractant_primary_smiles": int(sub["extractant_primary_smiles"].nunique()),
        "extractant_primary_names": int(sub["extractant_primary_name"].nunique()),
        "extractant_names_all": len(names),
        "extractant_families": None,
        "publications": int(sub["g19_publication_id"].nunique()),
        "studies": int(sub["g19_study_id"].nunique()),
        "acid_primary": int(sub["acid_primary"].nunique()),
        "acid_signature": int(cv.loc[sub.index, "acid_signature"].nunique()),
        "diluent_solvent_key": int(sub["solvent_key"].nunique()),
        "diluent_solvent_primary": int(sub["solvent_primary"].nunique()),
        "diluent_solvent_components": len(comps),
        "diluent_family": int(cv.loc[sub.index, "diluent_family"].nunique()),
        "temperatures_C": int(sub["temperature_C"].nunique()),
    }


def acid_semantics(df: pd.DataFrame, cv: pd.DataFrame, is_model: np.ndarray) -> dict:
    """Counts behind the acid-molarity semantics flags of ``normalize.condition_vector`` (INFERRED)."""
    grid, low = cv["acid_M_log10_grid"].astype(bool), cv["acid_M_below_1e-3"].astype(bool)
    has_acid = cv["acid_concentration_M"].notna()
    model = pd.Series(is_model, index=df.index)
    acid = cv["acid_concentration_M"]

    def by(col, mask):
        g = pd.DataFrame({"k": df[col].map(stringify), "flag": mask, "has": has_acid})[model]
        t = g.groupby("k").agg(model_rows_with_acid=("has", "sum"), flagged=("flag", "sum"))
        t = t[t["flagged"] > 0]
        t["share_flagged"] = t["flagged"] / t["model_rows_with_acid"]
        t = t.loc[sorted(t.index, key=lambda k: (-int(t.at[k, "flagged"]), k))]
        return [{"key": k, "flagged": int(r.flagged), "model_rows_with_acid": int(r.model_rows_with_acid),
                 "share_flagged": float(r.share_flagged)} for k, r in t.iterrows()]

    def rng(mask):
        v = acid[mask]
        return {"min": float(v.min()), "median": float(v.median()), "max": float(v.max())} if len(v) else None

    sf = acid.map(N.significant_figures)
    eligible = sf.map(lambda v: v is not None and not N.is_missing(v) and v > N.ROUND_SIG_MAX).astype(bool)
    chance = 2.0 * N.LOG10_GRID_TOL / N.LOG10_GRID_STEP
    return {
        "rules": N.ACID_SEMANTICS_FLAGS,
        "status": "INFERRED heuristic; no value is altered, no row is dropped",
        "chance_false_positive_model": {
            "rows_with_more_than_round_sig_figs_model": int((eligible & model).sum()),
            "chance_probability_per_row": chance,
            "expected_chance_hits_model": float((eligible & model).sum() * chance),
            "note": "a value with arbitrary mantissa lands on the 0.01 log10 grid (within tolerance) with probability "
                    "2*tol/step; publications with only a few flagged rows are compatible with chance",
        },
        "rows_with_acid_M_all": int(has_acid.sum()), "rows_with_acid_M_model": int((has_acid & model).sum()),
        "acid_M_log10_grid_all": int(grid.sum()), "acid_M_log10_grid_model": int((grid & model).sum()),
        "acid_M_below_1e-3_all": int(low.sum()), "acid_M_below_1e-3_model": int((low & model).sum()),
        "both_flags_model": int((grid & low & model).sum()),
        "acid_M_range_log10_grid_model": rng(grid & model),
        "acid_M_range_below_1e-3_model": rng(low & model),
        "log10_grid_model_by_publication": by("g19_publication_id", grid),
        "log10_grid_model_by_metal": by("metal_symbol", grid),
        "below_1e-3_model_by_publication": by("g19_publication_id", low),
        "below_1e-3_model_by_acid_primary": by("acid_primary", low),
    }


def counts_json(df, meta, cv, is_model, ck_all) -> dict:
    model = df[is_model]
    by = lambda s: {str(k): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}  # noqa: E731
    avail = []
    for item, cols, note in METADATA_14:
        if not cols:
            avail.append({"item": item, "columns": [], "status": "ABSENT", "rows_filled_all": 0,
                          "rows_filled_model": 0, "fill_fraction_all": 0.0, "fill_fraction_model": 0.0, "note": note})
            continue
        mask = pd.Series(False, index=df.index)
        for c in cols:
            mask |= df[c].map(lambda v: not N.is_missing(v)).astype(bool)
        fa, fm = float(mask.mean()), float(mask[is_model].mean())
        avail.append({"item": item, "columns": cols, "status": "AVAILABLE" if fa >= 0.95 else "PARTIAL",
                      "rows_filled_all": int(mask.sum()), "rows_filled_model": int(mask[is_model].sum()),
                      "fill_fraction_all": fa, "fill_fraction_model": fm, "note": note})
    scan_rx = re.compile(r"(^|_)(ph|sapon\w*|loading|uncert\w*|error|stdev|std|sigma|famil\w*|reconstruct\w*|reported)(_|$)")
    scan = sorted(c for c in df.columns if scan_rx.search(c.lower()))
    comp_counter = Counter()
    for arr in df["solvent_components"]:
        comp_counter.update(str(x) for x in arr)
    dil_rules = {}
    for name in sorted(comp_counter):
        cls, rule = N.classify_solvent_component(name)
        dil_rules[name] = {"component_class": cls, "rule": rule, "rows": int(comp_counter[name])}
    ph_text = int(df["comments_raw"].fillna("").str.contains(r"\bpH\b", regex=True).sum())
    ek_all = N.experiment_key(df, condition_keys=ck_all)
    flags = N.unit_sanity_flags(cv)
    flagged_rows = {k: [{"canonical_measurement_id": str(df.at[i, "canonical_measurement_id"]),
                         "g19_tier": str(df.at[i, "g19_tier"]), "metal_symbol": stringify(df.at[i, "metal_symbol"]),
                         "value_acid_M": N.format_sig(cv.at[i, "acid_concentration_M"]),
                         "value_extractant_primary_M": N.format_sig(cv.at[i, "extractant_primary_concentration_M"])}
                        for i in flags.index[flags[k]]][:25] for k in flags.columns}
    acid = acid_semantics(df, cv, is_model)
    plaus = df["metal_oxidation_state_plausible"]
    implausible = df[plaus.map(lambda v: isinstance(v, (bool, np.bool_)) and not bool(v))]
    ox_plaus = {
        "column": "metal_oxidation_state_plausible (archive)",
        "rows_by_value_all": {stringify(k): int(v) for k, v in plaus.map(stringify).value_counts().sort_index().items()},
        "implausible_rows_all": int(len(implausible)),
        "implausible_rows_model": int((implausible["g19_tier"] == "MODEL").sum()),
        "implausible_metal_states": {f"{m}({roman(o)})": int(n) for (m, o), n in
                                     implausible.groupby(["metal_symbol", "metal_oxidation_state"]).size().items()},
        "implausible_publications": sorted(implausible["g19_publication_id"].unique().tolist()),
        "note": "kept in every table under the recorded state and marked in metal_coverage.csv; never corrected",
    }
    return {
        "dataset": {"archive_master": paths.rel(paths.ARCHIVE_MASTER),
                    "archive_master_sha256": paths.digests(paths.ARCHIVE_MASTER)["sha256"],
                    "archive_master_sha256_pinned": paths.ARCHIVE_MASTER_SHA256,
                    "n_archive_columns": int(len([c for c in df.columns if not c.startswith("g19_")])),
                    "n_g19_columns": int(len([c for c in df.columns if c.startswith("g19_")]))},
        "rows": {
            "total": int(len(df)), "by_tier": by(df["g19_tier"]),
            "by_metal_category_all": by(df["metal_category"].fillna("<no metal>")),
            "by_metal_category_model": by(model["metal_category"].fillna("<no metal>")),
            "by_system_component_class_all": by(df["system_component_class"]),
            "by_system_component_class_model": by(model["system_component_class"]),
            "by_publication_status_all": by(df["g19_publication_status"]),
            "by_publication_status_model": by(model["g19_publication_status"]),
            "model_rows_unknown_oxidation_state": int(model["metal_oxidation_state"].isna().sum()),
            "by_diluent_family_all": by(cv["diluent_family"]),
            "by_diluent_family_model": by(cv.loc[is_model, "diluent_family"]),
        },
        "unique": {"all_rows": uniques(df, meta, cv), "model_rows": uniques(model, meta, cv)},
        "extractant_families": {"value": None, "status": "NOT_IN_ARCHIVE",
                                "reason": "the archive has no extractant-family column; a structure-based family "
                                          "assignment is outside this audit (never inferred from names)"},
        "metadata_availability_section_1_4": avail,
        "absent_field_column_scan": {"regex": scan_rx.pattern, "matching_columns": scan,
                                     "ph_word_in_comments_raw_rows": ph_text,
                                     "note": "matches are listed for inspection; none is a pH, saponification, "
                                             "loading, uncertainty, family or reported-vs-reconstructed field "
                                             "unless listed here"},
        "unit_sanity": {"rules": N.UNIT_SANITY_RULES, "all_rows": N.unit_sanity_counts(cv),
                        "model_rows": N.unit_sanity_counts(cv[is_model]),
                        "flagged_rows_first_25": flagged_rows,
                        "note": "counts only; flagged values are kept unaltered"},
        "acid_molarity_semantics": acid,
        "oxidation_state_plausibility": ox_plaus,
        "condition_keys": {"sig": N.DEFAULT_SIG, "fields": list(N.CONDITION_KEY_FIELDS),
                           "n_condition_keys_all": int(ck_all.nunique()),
                           "n_condition_keys_model": int(ck_all[is_model].nunique()),
                           "n_experiment_keys_all": int(ek_all.nunique()),
                           "n_experiment_keys_model": int(ek_all[is_model].nunique())},
        "diluent_family_rules": {"families": list(N.DILUENT_FAMILIES),
                                 "component_classification": dil_rules},
    }


# =========================================================================== 4-5. matrices
def ordered_states(labels, meta) -> list[str]:
    info = meta.dropna(subset=["state_label"]).groupby("state_label").agg(
        cat=("metal_category", "first"), z=("atomic_number", "first"), ox=("metal_oxidation_state", "first"))

    def key(lbl):
        r = info.loc[lbl]
        ci = CATEGORY_ORDER.index(r["cat"]) if r["cat"] in CATEGORY_ORDER else len(CATEGORY_ORDER)
        return (ci, float(r["z"]), 99.0 if pd.isna(r["ox"]) else float(r["ox"]), lbl)
    return sorted(set(labels), key=key)


def system_labels(model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in model.groupby("extractant_system_key", sort=True):
        joined = [" + ".join(sorted(str(x) for x in arr)) for arr in g["extractant_names"]]
        label, n_names = mode_text(joined)
        rows.append({"extractant_system_key": key, "system_id": system_id(key), "system_label": label,
                     "primary_name_modal": mode_text(g["extractant_primary_name"])[0],
                     "n_distinct_name_strings": n_names,
                     "n_components": int(g["n_organic_extractants"].max()),
                     "system_component_classes": "|".join(sorted(g["system_component_class"].unique()))})
    return pd.DataFrame(rows).set_index("extractant_system_key")


def build_matrices(model: pd.DataFrame, states: list[str], labels: pd.DataFrame):
    rows_m = pd.crosstab(model["extractant_system_key"], model["state_label"]).reindex(columns=states, fill_value=0)
    pubs_m = model.groupby(["extractant_system_key", "state_label"])["g19_publication_id"].nunique().unstack(
        fill_value=0).reindex(index=rows_m.index, columns=states, fill_value=0)
    total = rows_m.sum(axis=1)
    order = sorted(rows_m.index, key=lambda k: (-int(total[k]), k))
    rows_m, pubs_m = rows_m.loc[order], pubs_m.loc[order]

    def frame(mat, kind):
        head = labels.loc[mat.index, ["system_id", "system_label", "primary_name_modal", "n_distinct_name_strings",
                                      "n_components", "system_component_classes"]].copy()
        head.insert(0, "extractant_system_key", mat.index)
        head["total_rows"] = total.loc[mat.index].astype(int).values
        if kind == "pubs":
            head["total_publications"] = model.groupby("extractant_system_key")["g19_publication_id"].nunique() \
                .loc[mat.index].astype(int).values
        present = rows_m.loc[mat.index] > 0
        known = [c for c in present.columns if not c.endswith("(?)")]
        head["n_metal_state_columns"] = present.sum(axis=1).astype(int).values
        head["n_metal_states_known_ox"] = present[known].sum(axis=1).astype(int).values
        head["n_elements"] = model.groupby("extractant_system_key")["metal_symbol"].nunique() \
            .loc[mat.index].astype(int).values
        out = pd.concat([head.reset_index(drop=True), mat.reset_index(drop=True).astype(int)], axis=1)
        return out

    prim = pd.crosstab(model["extractant_primary_smiles"], model["state_label"]).reindex(columns=states, fill_value=0)
    ptot = prim.sum(axis=1)
    porder = sorted(prim.index, key=lambda k: (-int(ptot[k]), k))
    prim = prim.loc[porder]
    plabel = model.groupby("extractant_primary_smiles")["extractant_primary_name"].agg(lambda v: mode_text(v)[0])
    pn = model.groupby("extractant_primary_smiles")["extractant_primary_name"].nunique()
    phead = pd.DataFrame({"extractant_primary_smiles": prim.index, "primary_label": plabel.loc[prim.index].values,
                          "n_distinct_primary_names": pn.loc[prim.index].astype(int).values,
                          "n_system_keys": model.groupby("extractant_primary_smiles")["extractant_system_key"]
                          .nunique().loc[prim.index].astype(int).values,
                          "total_rows": ptot.loc[prim.index].astype(int).values,
                          "n_metal_state_columns": (prim > 0).sum(axis=1).astype(int).values,
                          "n_metal_states_known_ox": (prim[[c for c in prim.columns if not c.endswith("(?)")]] > 0)
                          .sum(axis=1).astype(int).values,
                          "n_elements": model.groupby("extractant_primary_smiles")["metal_symbol"].nunique()
                          .loc[prim.index].astype(int).values})
    prim_out = pd.concat([phead, prim.reset_index(drop=True).astype(int)], axis=1)
    return rows_m, pubs_m, frame(rows_m, "rows"), frame(pubs_m, "pubs"), prim_out


def sparsity_block(model: pd.DataFrame, col: str, labels: pd.DataFrame, note: str) -> dict:
    sub = model[model[col].notna()]
    rows_m = pd.crosstab(sub["extractant_system_key"], sub[col])
    pubs_m = sub.groupby(["extractant_system_key", col])["g19_publication_id"].nunique().unstack(fill_value=0) \
        .reindex(index=rows_m.index, columns=rows_m.columns, fill_value=0)
    v = rows_m.to_numpy()
    nz = v > 0
    n_rows = int(v.sum())
    sys_tot = rows_m.sum(axis=1).sort_values(ascending=False, kind="mergesort")
    sys_tot = sys_tot.loc[sorted(sys_tot.index, key=lambda k: (-int(sys_tot[k]), k))]
    pub_tot = sub["g19_publication_id"].value_counts()
    pub_tot = pub_tot.loc[sorted(pub_tot.index, key=lambda k: (-int(pub_tot[k]), k))]
    span_sys = nz.sum(axis=1)
    span_state = nz.sum(axis=0)
    return {
        "note": note, "axis_columns": col, "n_rows": n_rows,
        "shape": [int(v.shape[0]), int(v.shape[1])], "n_cells": int(v.size), "nonzero_cells": int(nz.sum()),
        "density": float(nz.sum() / v.size),
        "cells_ge_1_rows": int((v >= 1).sum()), "cells_ge_5_rows": int((v >= 5).sum()),
        "cells_ge_10_rows": int((v >= 10).sum()), "cells_ge_20_rows": int((v >= 20).sum()),
        "cells_ge_2_publications": int((pubs_m.to_numpy() >= 2).sum()),
        "cells_ge_3_publications": int((pubs_m.to_numpy() >= 3).sum()),
        "systems_spanning_ge_3": int((span_sys >= 3).sum()), "systems_spanning_ge_5": int((span_sys >= 5).sum()),
        "systems_spanning_ge_10": int((span_sys >= 10).sum()),
        "metal_axis_spanning_ge_3_systems": int((span_state >= 3).sum()),
        "metal_axis_spanning_ge_5_systems": int((span_state >= 5).sum()),
        "metal_axis_spanning_ge_10_systems": int((span_state >= 10).sum()),
        "gini_rows_over_all_cells": gini(v.ravel()), "gini_rows_over_nonzero_cells": gini(v[nz]),
        "share_rows_top1_system": float(sys_tot.iloc[:1].sum() / n_rows),
        "share_rows_top5_systems": float(sys_tot.iloc[:5].sum() / n_rows),
        "share_rows_top1_publication": float(pub_tot.iloc[:1].sum() / n_rows),
        "share_rows_top5_publications": float(pub_tot.iloc[:5].sum() / n_rows),
        "top5_systems": [{"system_id": labels.loc[k, "system_id"], "label": labels.loc[k, "system_label"],
                          "rows": int(sys_tot[k])} for k in sys_tot.index[:5]],
        "top5_publications": [{"publication_id": k, "rows": int(pub_tot[k])} for k in pub_tot.index[:5]],
    }


# =========================================================================== 6. metal coverage
def metal_coverage(df: pd.DataFrame, meta: pd.DataFrame, cv: pd.DataFrame, states_all: list[str]) -> pd.DataFrame:
    rows = []
    has_metal = meta["state_label"].notna()
    for lbl in states_all:
        m = has_metal & (meta["state_label"] == lbl)
        g = df[m]
        gm = g[g["g19_tier"] == "MODEL"]
        ld = gm["log_D"].astype(float)
        plaus = {stringify(v) for v in g["metal_oxidation_state_plausible"]}
        rows.append({
            "metal_state": lbl, "metal_symbol": g["metal_symbol"].iloc[0],
            "oxidation_state": g["metal_oxidation_state"].iloc[0], "oxidation_state_known": not lbl.endswith("(?)"),
            "oxidation_state_plausible_archive": "|".join(sorted(plaus)),
            "model_rows_acid_M_log10_grid": int(cv.loc[gm.index, "acid_M_log10_grid"].sum()),
            "model_rows_acid_M_below_1e-3": int(cv.loc[gm.index, "acid_M_below_1e-3"].sum()),
            "metal_category": g["metal_category"].iloc[0], "atomic_number": g["atomic_number"].iloc[0],
            "model_rows": int(len(gm)), "target_only_rows": int((g["g19_tier"] == "TARGET_ONLY").sum()),
            "no_target_rows": int((g["g19_tier"] == "NO_TARGET").sum()),
            "model_systems": int(gm["extractant_system_key"].nunique()),
            "model_primary_extractants": int(gm["extractant_primary_smiles"].nunique()),
            "model_publications": int(gm["g19_publication_id"].nunique()),
            "model_studies": int(gm["g19_study_id"].nunique()),
            "model_acids": int(gm["acid_signature"].replace("", pd.NA).nunique()),
            "model_diluents": int(gm["solvent_key"].nunique()),
            "logD_min": ld.min() if len(ld) else np.nan, "logD_median": ld.median() if len(ld) else np.nan,
            "logD_max": ld.max() if len(ld) else np.nan, "logD_sd": ld.std(ddof=1) if len(ld) > 1 else np.nan,
            "frac_model_with_metal_concentration": fill_fraction(gm["metal_concentration_M"]),
            "frac_model_with_phase_ratio": fill_fraction(gm["phase_ratio_org_aq"]),
            "frac_model_with_contact_time": fill_fraction(gm["contact_time_min"]),
            "frac_model_with_contact_or_shaking_time": float(
                (gm["contact_time_min"].notna() | gm["shaking_time_min"].notna()).mean()) if len(gm) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["oxidation_state"] = pd.array(out["oxidation_state"].round().astype("Int64"), dtype="Int64")
    out["atomic_number"] = pd.array(out["atomic_number"].round().astype("Int64"), dtype="Int64")
    return out


# =========================================================================== 7. Pr/Nd
def prnd_coverage(model: pd.DataFrame, labels: pd.DataFrame, ck: pd.Series, ck_nometal: pd.Series) -> pd.DataFrame:
    sub = model[model["metal_symbol"].isin(["Pr", "Nd"])]
    rows = []
    ln_model = model[model["metal_symbol"].isin(LANTHANIDES)]
    for key in sorted(sub["extractant_system_key"].unique()):
        g = sub[sub["extractant_system_key"] == key]
        pr, nd = g[g["metal_symbol"] == "Pr"], g[g["metal_symbol"] == "Nd"]
        pubs_pr, pubs_nd = set(pr["g19_publication_id"]), set(nd["g19_publication_id"])

        def shared(keys: pd.Series):
            kp = set(zip(pr["g19_publication_id"], keys.loc[pr.index]))
            kn = set(zip(nd["g19_publication_id"], keys.loc[nd.index]))
            both = kp & kn
            n_pr = int(sum((p, k) in both for p, k in zip(pr["g19_publication_id"], keys.loc[pr.index])))
            n_nd = int(sum((p, k) in both for p, k in zip(nd["g19_publication_id"], keys.loc[nd.index])))
            return len(both), n_pr, n_nd
        s_full, s_pr, s_nd = shared(ck)
        s_nm, _, _ = shared(ck_nometal)
        lns = ln_model[ln_model["extractant_system_key"] == key]["metal_symbol"].unique()
        others = [x for x in LANTHANIDES if x in set(lns) and x not in ("Pr", "Nd")]
        rows.append({
            "extractant_system_key": key, "system_id": labels.loc[key, "system_id"],
            "system_label": labels.loc[key, "system_label"],
            "pr_rows": int(len(pr)), "pr_iii_rows": int((pr["metal_oxidation_state"] == 3).sum()),
            "pr_unknown_ox_rows": int(pr["metal_oxidation_state"].isna().sum()),
            "nd_rows": int(len(nd)), "nd_iii_rows": int((nd["metal_oxidation_state"] == 3).sum()),
            "nd_unknown_ox_rows": int(nd["metal_oxidation_state"].isna().sum()),
            "publications_with_pr": len(pubs_pr), "publications_with_nd": len(pubs_nd),
            "publications_with_both": len(pubs_pr & pubs_nd),
            "shared_condition_keys_same_publication": s_full,
            "pr_rows_in_shared_keys": s_pr, "nd_rows_in_shared_keys": s_nd,
            "shared_condition_keys_same_publication_excl_metal_conc": s_nm,
            "n_other_ln": len(others), "other_ln_present": "|".join(others) if others else "",
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["shared_condition_keys_same_publication", "publications_with_both", "pr_rows", "nd_rows",
                            "extractant_system_key"], ascending=[False, False, False, False, True]).reset_index(drop=True)


# =========================================================================== 8-9. Ln / An coverage
def longest_run(present: list[bool]) -> int:
    best = cur = 0
    for p in present:
        cur = cur + 1 if p else 0
        best = max(best, cur)
    return best


def lanthanide_coverage(model: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    ln = model[model["metal_symbol"].isin(LANTHANIDES)]
    cols = [f"{x}(III)" for x in LANTHANIDES]
    rows = []
    for key, g in ln.groupby("extractant_system_key", sort=True):
        iii = g[g["metal_oxidation_state"] == 3]
        cnt = iii["metal_symbol"].value_counts()
        rec = {"extractant_system_key": key, "system_id": labels.loc[key, "system_id"],
               "system_label": labels.loc[key, "system_label"]}
        for x, c in zip(LANTHANIDES, cols):
            rec[c] = int(cnt.get(x, 0))
        pres_iii = [rec[c] > 0 for c in cols]
        any_state = set(g["metal_symbol"])
        pres_any = [x in any_state for x in LANTHANIDES]
        rec.update({
            "ln_unknown_ox_rows": int(g["metal_oxidation_state"].isna().sum()),
            "ln_other_ox_rows": int((g["metal_oxidation_state"].notna() & (g["metal_oxidation_state"] != 3)).sum()),
            "total_ln_rows": int(len(g)), "n_publications_ln": int(g["g19_publication_id"].nunique()),
            "n_ln_iii": int(sum(pres_iii)), "longest_run_iii": longest_run(pres_iii),
            "longest_run_iii_pm_skipped": longest_run([p for x, p in zip(LANTHANIDES, pres_iii) if x != "Pm"]),
            "n_ln_any_state": int(sum(pres_any)), "longest_run_any_state": longest_run(pres_any),
            "longest_run_any_state_pm_skipped": longest_run([p for x, p in zip(LANTHANIDES, pres_any) if x != "Pm"]),
            "ln_iii_present": "|".join(x for x, p in zip(LANTHANIDES, pres_iii) if p),
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(["n_ln_iii", "total_ln_rows", "extractant_system_key"],
                           ascending=[False, False, True]).reset_index(drop=True)


def actinide_coverage(model: pd.DataFrame, meta: pd.DataFrame, labels: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    an = model[model["metal_category"] == "actinide"]
    an_states = [s for s in states if s in set(meta.loc[an.index, "state_label"])]
    ln = model[model["metal_symbol"].isin(LANTHANIDES)]
    ln_by_sys = ln.groupby("extractant_system_key")["metal_symbol"].agg(lambda v: set(v))
    ln_rows = ln["extractant_system_key"].value_counts()
    rows = []
    for key, g in an.groupby("extractant_system_key", sort=True):
        cnt = meta.loc[g.index, "state_label"].value_counts()
        rec = {"extractant_system_key": key, "system_id": labels.loc[key, "system_id"],
               "system_label": labels.loc[key, "system_label"]}
        for s in an_states:
            rec[s] = int(cnt.get(s, 0))
        lnset = ln_by_sys.get(key, set())
        rec.update({
            "total_actinide_rows": int(len(g)),
            "n_actinide_states_known_ox": int(sum(1 for s in an_states if rec[s] > 0 and not s.endswith("(?)"))),
            "n_actinide_elements": int(g["metal_symbol"].nunique()),
            "actinide_elements": "|".join(sorted(g["metal_symbol"].unique())),
            "n_publications_actinide": int(g["g19_publication_id"].nunique()),
            "also_has_ln_rows": bool(len(lnset) > 0), "ln_rows": int(ln_rows.get(key, 0)),
            "n_ln_elements": len(lnset), "ln_elements": "|".join(x for x in LANTHANIDES if x in lnset),
            "has_am_and_eu": bool("Am" in set(g["metal_symbol"]) and "Eu" in lnset),
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(["total_actinide_rows", "extractant_system_key"], ascending=[False, True]).reset_index(drop=True)


# =========================================================================== 10. conditions
CONDITION_VARS = [  # (variable, kind, unit)
    ("acid_signature", "categorical", ""), ("acid_primary", "categorical", ""), ("acid_anion", "categorical", ""),
    ("acid_concentration_M", "numeric", "M"), ("acid_concentration_organic_M", "numeric", "M"),
    ("nitrate_concentration_M", "numeric", "M"),
    ("extractant_primary_concentration_M", "numeric", "M"), ("extractant_total_concentration_M", "numeric", "M"),
    ("extractant_concentrations_sorted_M", "tuple", "M"),
    ("metal_concentration_M", "numeric", "M"), ("phase_ratio_org_aq", "numeric", "O/A"),
    ("solvent_key", "categorical", ""), ("diluent_family", "categorical", ""),
    ("modifier_name", "categorical", ""), ("modifier_concentration_M", "numeric", "M"),
    ("complexant_signature", "categorical", ""),
    ("complexant_structure_key", "categorical", ""), ("complexant_concentrations_sorted_M", "tuple", "M"),
    ("holdback_structure_key", "categorical", ""), ("holdback_concentrations_sorted_M", "tuple", "M"),
    ("temperature_C", "numeric", "C"), ("contact_time_min", "numeric", "min"),
    ("shaking_time_min", "numeric", "min"),
    ("pH", "not_available", ""), ("saponification_degree", "not_available", ""),
    ("organic_loading", "not_available", ""), ("ionic_strength_M", "not_available", "M"),
]


def condition_coverage(df, meta, cv, is_model) -> pd.DataFrame:
    cvm = cv[is_model]
    mm = df[is_model]
    cats = [c for c in CATEGORY_ORDER if c in set(mm["metal_category"])]
    groups = pd.Series(list(zip(mm["extractant_system_key"], meta.loc[mm.index, "state_label"],
                                mm["g19_publication_id"])), index=mm.index)
    n_groups_multi = int((groups.value_counts() >= 2).sum())
    rows = []
    for var, kind, unit in CONDITION_VARS:
        s_all, s = cv[var], cvm[var]
        if kind == "tuple":
            filled_all, filled = s_all.map(tuple_filled), s.map(tuple_filled)
        elif kind == "not_available":
            filled_all, filled = pd.Series(False, index=s_all.index), pd.Series(False, index=s.index)
        else:
            filled_all, filled = s_all.map(lambda v: not N.is_missing(v)), s.map(lambda v: not N.is_missing(v))
        rec = {"variable": var, "kind": kind, "unit": unit,
               "fill_fraction_all": float(filled_all.mean()), "fill_fraction_model": float(filled.mean()),
               "n_model_filled": int(filled.sum())}
        vals = pd.to_numeric(s[filled], errors="coerce").astype(float) if kind == "numeric" else None
        for q, name in ((None, "min"), (0.05, "p05"), (0.5, "median"), (0.95, "p95"), (None, "max")):
            if vals is None or vals.empty:
                rec[name] = np.nan
            elif name == "min":
                rec[name] = float(vals.min())
            elif name == "max":
                rec[name] = float(vals.max())
            else:
                rec[name] = float(vals.quantile(q))
        text = s[filled].map(lambda v: N.format_sig(v, N.DEFAULT_SIG))
        rec["n_distinct_model"] = int(text.nunique())
        for c in cats:
            idx = mm.index[mm["metal_category"] == c]
            rec[f"fill_model_{c}"] = float(filled.loc[idx].mean())
        for c in cats:
            idx = mm.index[(mm["metal_category"] == c).to_numpy() & filled.to_numpy()]
            rec[f"median_model_{c}"] = (float(pd.to_numeric(s.loc[idx], errors="coerce").median())
                                        if kind == "numeric" and len(idx) else np.nan)
            rec[f"n_distinct_model_{c}"] = int(s.loc[idx].map(lambda v: N.format_sig(v, N.DEFAULT_SIG)).nunique())
        if kind == "not_available":
            rec.update({"n_groups_varying": 0, "n_systems_varying": 0,
                        "note": N.NOT_AVAILABLE.get(var, "")})
        else:
            t = pd.DataFrame({"g": groups[filled], "v": text})
            nun = t.groupby("g")["v"].nunique()
            varying = nun[nun >= 2].index
            rec["n_groups_varying"] = int(len(varying))
            rec["n_systems_varying"] = int(len({g[0] for g in varying}))
            rec["note"] = ""
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["n_groups_with_ge2_rows"] = n_groups_multi
    out["n_systems_model"] = int(mm["extractant_system_key"].nunique())
    return out


# =========================================================================== figures
def _heat(ax, mat: np.ndarray, vmax: float):
    data = np.where(mat > 0, np.log10(np.maximum(mat, 1)), np.nan)
    cmap = SEQ_BLUE.copy()
    cmap.set_bad("#ffffff")
    return ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                     interpolation="nearest")


def _style(ax):
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7)


def fig_observation_matrix(rows_m: pd.DataFrame, labels: pd.DataFrame, meta: pd.DataFrame, out: Path, top: int = 80):
    total = rows_m.sum(axis=1)
    keep = rows_m.index[:top]
    sub = rows_m.loc[keep]
    sub = sub.loc[:, sub.sum(axis=0) > 0]
    cats = meta.dropna(subset=["state_label"]).groupby("state_label")["metal_category"].first()
    share = total.loc[keep].sum() / total.sum()
    fig_w = 4.2 + 0.19 * sub.shape[1]
    fig_h = 1.8 + 0.155 * sub.shape[0]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.log10(max(1, sub.to_numpy().max())))
    im = _heat(ax, sub.to_numpy(), vmax)
    ax.set_yticks(range(sub.shape[0]))
    ax.set_yticklabels([f"{truncate(labels.loc[k, 'system_label'], 30)}  ({int(total[k])})" for k in sub.index],
                       fontsize=6.2, color=INK)
    ax.set_xticks(range(sub.shape[1]))
    ax.set_xticklabels(sub.columns, rotation=90, fontsize=6.5, color=INK)
    prev = None
    for j, c in enumerate(sub.columns):
        cat = cats.get(c)
        if prev is not None and cat != prev:
            ax.axvline(j - 0.5, color=INK2, lw=0.8)
        if cat != prev:
            ax.text(j - 0.4, -1.0, CAT_SHORT.get(cat, str(cat)), fontsize=7, color=INK2, ha="left", va="bottom")
        prev = cat
    ax.set_xticks(np.arange(-0.5, sub.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, sub.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#f0efec", lw=0.3)
    ax.tick_params(which="minor", length=0)
    _style(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("log10(MODEL rows in cell); white = no row", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6, colors=INK2)
    ax.set_title(f"Which metal states has each extractant system been measured with?\n"
                 f"MODEL rows; TRUNCATED to the top {len(keep)} of {len(rows_m)} systems by row count "
                 f"({share:.0%} of rows); columns grouped by metal category; '(?)' = oxidation state not recorded",
                 fontsize=8.5, color=INK, loc="left", pad=46)
    ax.set_ylabel("extractant system (modal name; total rows)", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_density_by_metal(mc: pd.DataFrame, out: Path):
    d = mc[mc["model_rows"] > 0].copy()
    d = d.iloc[::-1].reset_index(drop=True)
    colors = [CAT_COLORS.get(c, "#9a9994") for c in d["metal_category"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.17 * len(d) + 1.6), sharey=True)
    y = np.arange(len(d))
    for ax, col, xl in ((axes[0], "model_rows", "MODEL rows (log scale)"),
                        (axes[1], "model_systems", "distinct extractant systems (log scale)")):
        ax.barh(y, d[col], color=colors, height=0.72, edgecolor="white", linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(xl, fontsize=7.5, color=INK2)
        for yi, v in zip(y, d[col]):
            ax.text(v * 1.08, yi, f"{int(v)}", va="center", fontsize=5.8, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        _style(ax)
        ax.set_xlim(0.8, d[col].max() * 3)
        ax.set_ylim(-0.7, len(d) - 0.3)
    implaus = d["oxidation_state_plausible_archive"].astype(str).eq("False")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{s} *" if bad else s for s, bad in zip(d["metal_state"], implaus)],
                            fontsize=6.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[c]) for c in CATEGORY_ORDER if c in set(d["metal_category"])]
    axes[1].legend(handles, [c.replace("_", " ") for c in CATEGORY_ORDER if c in set(d["metal_category"])],
                   fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("How much MODEL data, and how many distinct extractant systems, does each metal state have?\n"
                 "'(?)' = oxidation state not recorded by the archive (not imputed); "
                 "'*' = state the archive itself marks implausible (kept as recorded)",
                 fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_lanthanide(lc: pd.DataFrame, out: Path):
    d = lc[lc["n_ln_iii"] >= 3].reset_index(drop=True)
    cols = [f"{x}(III)" for x in LANTHANIDES]
    mat = d[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 1.6 + 0.15 * len(d)))
    im = _heat(ax, mat, float(np.log10(max(1, mat.max()))))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(LANTHANIDES, fontsize=7, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{truncate(l, 30)}" for l in d["system_label"]], fontsize=6, color=INK)
    for i, (n, r, rs) in enumerate(zip(d["n_ln_iii"], d["longest_run_iii"], d["longest_run_iii_pm_skipped"])):
        ax.text(len(cols) - 0.3, i, f" n={n}  run={r}/{rs}", va="center", fontsize=5.8, color=INK2)
    ax.set_xlim(-0.5, len(cols) + 3.2)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(d), 1), minor=True)
    ax.grid(which="minor", color="#f0efec", lw=0.3)
    ax.tick_params(which="minor", length=0)
    _style(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("log10(MODEL rows); white = none", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6, colors=INK2)
    ax.set_title(f"Which trivalent lanthanides are measured under the same extractant system?\n"
                 f"{len(d)} of {len(lc)} Ln-bearing systems (MODEL rows) have >=3 Ln(III); n = Ln(III) count;\n"
                 f"run = longest contiguous La..Lu run, Pm counted as a gap / Pm skipped; "
                 f"unknown-oxidation-state rows excluded",
                 fontsize=8, color=INK, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_actinide(ac: pd.DataFrame, out: Path):
    meta_cols = {"extractant_system_key", "system_id", "system_label", "total_actinide_rows",
                 "n_actinide_states_known_ox", "n_actinide_elements", "actinide_elements", "n_publications_actinide",
                 "also_has_ln_rows", "ln_rows", "n_ln_elements", "ln_elements", "has_am_and_eu"}
    scols = [c for c in ac.columns if c not in meta_cols]
    d = ac.reset_index(drop=True)
    mat = d[scols].to_numpy()
    fig, ax = plt.subplots(figsize=(1.9 + 0.3 * len(scols) + 2.6, 1.7 + 0.135 * len(d)))
    im = _heat(ax, mat, float(np.log10(max(1, mat.max()))))
    ax.set_xticks(range(len(scols)))
    ax.set_xticklabels(scols, rotation=90, fontsize=6.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([truncate(l, 30) for l in d["system_label"]], fontsize=5.6, color=INK)
    xm = len(scols) + 0.2
    for i, (has, nl) in enumerate(zip(d["also_has_ln_rows"], d["n_ln_elements"])):
        if has:
            ax.plot(xm, i, marker="o", ms=3.4, color=CAT_COLORS["lanthanide"], mec="white", mew=0.5)
            ax.text(xm + 0.45, i, f"{nl} Ln", va="center", fontsize=5.4, color=INK2)
    ax.text(xm, -1.0, "also has\nLn rows", fontsize=6, color=INK2, ha="center", va="bottom")
    ax.set_xlim(-0.5, len(scols) + 1.8)
    ax.set_xticks(np.arange(-0.5, len(scols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(d), 1), minor=True)
    ax.grid(which="minor", color="#f0efec", lw=0.3)
    ax.tick_params(which="minor", length=0)
    _style(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label("log10(MODEL rows); white = none", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6, colors=INK2)
    n_ln = int(d["also_has_ln_rows"].sum())
    ax.set_title(f"Which actinide states are measured under each extractant system, and does that system also have "
                 f"lanthanide rows?\nall {len(d)} actinide-bearing systems (MODEL rows); {n_ln} also have Ln rows "
                 f"(blue dot, Ln element count); '(?)' = oxidation state not recorded",
                 fontsize=7.8, color=INK, loc="left", pad=40)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def _marker_area(n):
    return 4.0 + 7.0 * np.sqrt(np.asarray(n, dtype=float))


def fig_conditions(df, cv, is_model, cc: pd.DataFrame, out: Path):
    """Small multiples (one panel per metal group, shared log axes): marker area ~ number of MODEL rows
    at each exact (acid M, primary extractant M) point, so dense groups cannot hide sparse ones; then
    the marginal fill fraction of every condition variable per group."""
    mm = df[is_model]
    cvm = cv[is_model]
    grp = mm["metal_category"].map(lambda c: c if c in ("lanthanide", "actinide") else "other metal")
    groups = ("lanthanide", "actinide", "other metal")
    fig = plt.figure(figsize=(13.5, 11.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.32, wspace=0.08)
    ok = (cvm["acid_concentration_M"] > 0) & (cvm["extractant_primary_concentration_M"] > 0)
    x_all, y_all = cvm.loc[ok, "acid_concentration_M"], cvm.loc[ok, "extractant_primary_concentration_M"]
    xlim = (10 ** np.floor(np.log10(x_all.min())), 10 ** np.ceil(np.log10(x_all.max())))
    ylim = (10 ** np.floor(np.log10(y_all.min())), 10 ** np.ceil(np.log10(y_all.max())))
    first = None
    for k, g in enumerate(groups):
        ax = fig.add_subplot(gs[0, k], sharex=first, sharey=first)
        first = first or ax
        m = ok & (grp == g)
        pts = pd.DataFrame({"x": cvm.loc[m, "acid_concentration_M"].map(N.round_sig),
                            "y": cvm.loc[m, "extractant_primary_concentration_M"].map(N.round_sig)})
        cnt = pts.groupby(["x", "y"]).size().reset_index(name="n")
        ax.scatter(cnt["x"], cnt["y"], s=_marker_area(cnt["n"]), color=SCATTER_COLORS[g],
                   alpha=0.45, edgecolors="white", linewidths=0.3)
        ax.axvline(N.LOW_ACID_M, color=INK2, lw=0.7, ls="--")
        n_low = int((cvm.loc[m, "acid_M_below_1e-3"]).sum())
        n_grid = int((cvm.loc[m, "acid_M_log10_grid"]).sum())
        ax.text(0.02, 0.97, f"{g}: {int(m.sum())} rows at {len(cnt)} distinct points\n"
                            f"{n_low} rows < 1e-3 M acid (left of dashed line)\n"
                            f"{n_grid} rows with acid M on a 0.01 log10 grid (INFERRED pH/log-axis origin)",
                transform=ax.transAxes, fontsize=6.8, color=INK, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        _style(ax)
        ax.set_xlabel("aqueous acid (M, as recorded; not pH)", fontsize=7.5, color=INK2)
        if k == 0:
            ax.set_ylabel("primary extractant (M)", fontsize=7.5, color=INK2)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    for n_ref in (1, 10, 100):
        first.scatter([], [], s=_marker_area(n_ref), color="#9a9994", alpha=0.6, label=f"{n_ref} rows")
    first.legend(fontsize=6.5, frameon=False, loc="lower left", title="marker area ~ sqrt(rows)", title_fontsize=6.5)
    n_missing = int((~ok).sum())
    first.set_title(f"acid x extractant concentration by metal group (MODEL rows; {n_missing} rows lack a positive "
                    f"value for one of the two)", fontsize=8.2, color=INK, loc="left")

    ax2 = fig.add_subplot(gs[1, :])
    show = cc[cc["kind"] != "not_available"].reset_index(drop=True)
    na = cc[cc["kind"] == "not_available"]["variable"].tolist()
    other_idx = mm.index[~mm["metal_category"].isin(["lanthanide", "actinide"])]
    fills = {"lanthanide": show["fill_model_lanthanide"].to_numpy(), "actinide": show["fill_model_actinide"].to_numpy()}
    other_fill = []
    for var, kind in zip(show["variable"], show["kind"]):
        s = cv.loc[other_idx, var]
        f = s.map(tuple_filled) if kind == "tuple" else s.map(lambda v: not N.is_missing(v))
        other_fill.append(float(f.mean()) if len(f) else np.nan)
    fills["other metal"] = np.array(other_fill)
    y = np.arange(len(show))[::-1]
    h = 0.27
    for k, g in enumerate(groups):
        ax2.barh(y + (1 - k) * h, fills[g], height=h, color=SCATTER_COLORS[g], edgecolor="white", lw=0.4,
                 label=f"{g} (n={int((grp == g).sum())})")
    ax2.set_yticks(y)
    ax2.set_yticklabels(show["variable"], fontsize=7, color=INK)
    ax2.set_ylim(-0.7, len(show) - 0.3)
    ax2.set_xlim(0, 1.32)
    ax2.set_xticks(np.linspace(0, 1, 6))
    ax2.set_xlabel("fraction of MODEL rows with the variable recorded", fontsize=8, color=INK2)
    ax2.grid(axis="x", color=GRID, lw=0.5)
    ax2.set_axisbelow(True)
    _style(ax2)
    ax2.legend(fontsize=7, frameon=False, loc="center right")
    ax2.set_title("marginal fill fractions by metal group; not in the archive at all (always NA): " + ", ".join(na),
                  fontsize=8.2, color=INK, loc="left")
    fig.suptitle("Is condition space covered well enough to separate metal effects from condition effects?",
                 fontsize=10, color=INK, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.05)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# =========================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-systems", type=int, default=80, help="systems shown in F01")
    ap.add_argument("--sig", type=int, default=N.DEFAULT_SIG, help="significant figures in condition keys")
    ns = ap.parse_args(argv)

    A, F = paths.ensure_dir(paths.DATA_AUDIT_DIR), paths.ensure_dir(paths.FIGURES_DIR)
    outs = {k: A / f"{k}.csv" for k in ("dataset_hashes", "columns", "metal_coverage", "matrix_rows_system_x_metal",
                                        "matrix_pubs_system_x_metal", "matrix_rows_primary_extractant_x_metal",
                                        "prnd_coverage", "lanthanide_coverage", "actinide_coverage",
                                        "condition_coverage")}
    outs["counts"], outs["sparsity"] = A / "counts.json", A / "sparsity.json"
    figs = {"F01": F / "F01_observation_matrix.png", "F02": F / "F02_density_by_metal.png",
            "F04": F / "F04_lanthanide_coverage.png", "F05": F / "F05_actinide_coverage.png",
            "F06": F / "F06_condition_coverage.png"}

    with Run(NAME, args=vars(ns), seed=None) as run:
        run.inputs(paths.ARCHIVE_MASTER, paths.ARCHIVE_MANIFEST, paths.ARCHIVE_DIR / "reports" / "schema.md",
                   paths.ARCHIVE_DIR / "reports" / "raw_checksums.sha256", paths.BUNDLE_DATASET,
                   paths.G19_ROOT / "gen19ct" / "data" / "load.py", paths.G19_ROOT / "gen19ct" / "data" / "normalize.py")

        write_csv(dataset_hashes(), outs["dataset_hashes"])

        df = g19load.load_archive(copy=False)  # read-only use; never mutated
        archive_columns = [c for c in df.columns if not c.startswith("g19_")]
        is_model = (df["g19_tier"] == "MODEL").to_numpy()
        meta = pd.DataFrame({
            "state_label": [state_label(m, o) for m, o in zip(df["metal_symbol"], df["metal_oxidation_state"])],
            "metal_category": df["metal_category"], "atomic_number": df["atomic_number"],
            "metal_oxidation_state": df["metal_oxidation_state"]}, index=df.index)
        cv = N.condition_vector(df)
        ck_all = N.condition_key(cv, ns.sig)
        ck_nometal = N.condition_key(cv, ns.sig, exclude=["metal_concentration_M"])

        write_csv(columns_table(df, is_model, archive_columns), outs["columns"])
        write_json(outs["counts"], counts_json(df, meta, cv, is_model, ck_all))

        model = df[is_model].copy()
        model["state_label"] = meta.loc[model.index, "state_label"]
        labels = system_labels(model)
        states = ordered_states(model["state_label"], meta)
        rows_m, pubs_m, rows_csv, pubs_csv, prim_csv = build_matrices(model, states, labels)
        write_csv(rows_csv, outs["matrix_rows_system_x_metal"])
        write_csv(pubs_csv, outs["matrix_pubs_system_x_metal"])
        write_csv(prim_csv, outs["matrix_rows_primary_extractant_x_metal"])

        model["state_known"] = model["g19_metal_state"]
        sparsity = {
            "system_x_metal_state": sparsity_block(model, "state_known", labels,
                                                   "MODEL rows with a recorded oxidation state; columns = g19_metal_state"),
            "system_x_metal_state_incl_unknown_ox": sparsity_block(
                model, "state_label", labels, "all MODEL rows; unknown oxidation state kept as its own 'X(?)' column "
                                              "(the layout of matrix_rows_system_x_metal.csv)"),
            "system_x_metal_symbol": sparsity_block(model, "metal_symbol", labels,
                                                    "all MODEL rows; columns = element symbol, states pooled"),
        }
        write_json(outs["sparsity"], sparsity)

        states_all = ordered_states(meta["state_label"].dropna(), meta)
        mc = metal_coverage(df, meta, cv, states_all)
        write_csv(mc, outs["metal_coverage"])
        write_csv(prnd_coverage(model, labels, ck_all, ck_nometal), outs["prnd_coverage"])
        lc = lanthanide_coverage(model, labels)
        write_csv(lc, outs["lanthanide_coverage"])
        ac = actinide_coverage(model, meta, labels, states)
        write_csv(ac, outs["actinide_coverage"])
        cc = condition_coverage(df, meta, cv, is_model)
        write_csv(cc, outs["condition_coverage"])

        fig_observation_matrix(rows_m, labels, meta, figs["F01"], top=ns.top_systems)
        fig_density_by_metal(mc, figs["F02"])
        fig_lanthanide(lc, figs["F04"])
        fig_actinide(ac, figs["F05"])
        fig_conditions(df, cv, is_model, cc, figs["F06"])

        run.outputs(list(outs.values()))
        run.outputs(list(figs.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
