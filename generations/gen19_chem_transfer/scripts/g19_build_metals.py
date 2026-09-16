"""g19_build_metals.py -- the gen19 metal descriptor table, its alias audit and its cross-checks.

Task B of Phase A (brief sections 4.1, 15, 27).  No model is fitted and ``log_D`` is read only to
count rows that carry a target.

Writes
    descriptors/metals.csv                       one row per (symbol, oxidation state) + Ln(III)/An(III)
    descriptors/metals_sources.md                provenance and verification status of every column
    data_audit/metal_alias_audit.csv             archive metal label combinations, unresolved rows,
                                                 mixed-state symbols
    data_audit/metal_descriptor_coverage.csv     MODEL-row coverage per column and every cross-check
    manifests/g19_build_metals.json (+ run_info)

Run from the repository root:
    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_metals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import importlib.util  # noqa: E402
from collections import Counter  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import metals as M  # noqa: E402
from gen19ct.data.load import load_archive  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_text  # noqa: E402

OUT_TABLE = paths.DESCRIPTORS_DIR / "metals.csv"
OUT_SOURCES = paths.DESCRIPTORS_DIR / "metals_sources.md"
OUT_ALIAS = paths.DATA_AUDIT_DIR / "metal_alias_audit.csv"
OUT_COVERAGE = paths.DATA_AUDIT_DIR / "metal_descriptor_coverage.csv"

METALS_PY = paths.G19_ROOT / "gen19ct" / "chemistry" / "metals.py"
GEN13_METALS = paths.GEN13_ROOT / "gen13sep" / "metals.py"
GEN11_METALREP = paths.REPO_ROOT / "src" / "lanthanide_separation" / "gen11" / "metalrep.py"
ARCHIVE_CHEM = paths.ARCHIVE_DIR / "scripts" / "sae_chem.py"
ARCHIVE_RAW_ROWS = paths.ARCHIVE_DIR / "intermediate" / "raw_rows.parquet"

RADIUS_TOL_A = 0.005

COVERAGE_COLUMNS = ["section", "check", "metal_category", "symbol", "oxidation_state", "column",
                    "g19_value", "other_value", "other_source", "other_n_distinct", "n_rows", "abs_diff",
                    "status", "note", "n_model_rows", "n_nonmissing", "fraction_nonmissing"]


# ------------------------------------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------------------------------------ #

def _load_module(name: str, path: Path):
    """Import a single file without executing its package ``__init__`` (read-only use)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _missing(v: Any) -> bool:
    return M._is_missing(v)


def _fmt(v: Any) -> str:
    if _missing(v):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def _ox_int(v: Any) -> int | None:
    return None if _missing(v) else int(v)


def _stringify_list(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return "|".join(sorted(str(x) for x in v))


def _state_counts(frame: pd.DataFrame) -> str:
    c = Counter("NA" if _missing(o) else M.ROMAN[int(o)] for o in frame["metal_oxidation_state"])
    order = sorted(c, key=lambda k: (99 if k == "NA" else M.parse_oxidation_state(k)))
    return ";".join(f"{k}:{c[k]}" for k in order)


def _compare(section: str, check: str, symbol: str | None, ox: int | None, column: str, g19: Any,
             other: Any, other_source: str, *, numeric: bool = False, note: str = "",
             n_rows: int | None = None, other_n_distinct: int | None = None,
             category: str | None = None) -> dict[str, Any]:
    abs_diff = None
    if _missing(g19) and _missing(other):
        status = "both_missing"
    elif _missing(g19):
        status = "other_only"
    elif _missing(other):
        status = "g19_only"
    elif numeric:
        abs_diff = abs(float(g19) - float(other))
        status = "agree" if abs_diff <= RADIUS_TOL_A + 1e-12 else "DISAGREE"
    else:
        status = "agree" if str(g19) == str(other) else "DISAGREE"
    return {"section": section, "check": check, "metal_category": category, "symbol": symbol,
            "oxidation_state": ox, "column": column, "g19_value": _fmt(g19), "other_value": _fmt(other),
            "other_source": other_source, "other_n_distinct": other_n_distinct, "n_rows": n_rows,
            "abs_diff": abs_diff, "status": status, "note": note}


# ------------------------------------------------------------------------------------------------ #
# 1. descriptor table
# ------------------------------------------------------------------------------------------------ #

def archive_keys(df: pd.DataFrame) -> list[tuple[str, int | None]]:
    sub = df[df["metal_symbol"].notna()]
    keys = {(s, _ox_int(o)) for s, o in zip(sub["metal_symbol"], sub["metal_oxidation_state"])}
    return sorted(keys, key=lambda k: (M.ATOMIC_NUMBER[k[0]], -1 if k[1] is None else k[1]))


# ------------------------------------------------------------------------------------------------ #
# 2. alias audit
# ------------------------------------------------------------------------------------------------ #

def alias_audit(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    forms_by_key = {(s, _ox_int(o)): (None if _missing(f) else f)
                    for s, o, f in zip(table["symbol"], table["oxidation_state"], table["species_form"])}
    is_model = df["g19_tier"] == "MODEL"
    has_target = df["log_D"].notna() & np.isfinite(df["log_D"].astype(float))
    keys = ["metal_raw", "metal_oxidation_state_raw", "metal_symbol", "metal_oxidation_state",
            "metal_species_form"]
    work = df[keys].copy()
    work["_model"] = is_model.to_numpy()
    work["_target"] = has_target.to_numpy()
    work["_src"] = df["metal_oxidation_state_source"]
    work["_plaus"] = df["metal_oxidation_state_plausible"].map(lambda v: "" if v is None or _missing(v) else str(bool(v)))
    work["_state"] = df["g19_metal_state"]
    for combo, grp in work.groupby(keys, dropna=False, sort=True):
        raw, ox_raw, sym, ox, form = (None if _missing(v) else v for v in combo)
        ox = _ox_int(ox)
        issues = []
        norm_sym = norm_ox = norm_form = None
        agrees = None
        detail = ""
        if sym is None:
            issues.append("metal_unresolved")
        else:
            if ox is None:
                issues.append("state_missing")
            if form is not None:
                issues.append("species_token_expanded")
            plaus = set(grp["_plaus"])
            if "False" in plaus:
                issues.append("implausible_state")
            if raw is not None:
                try:
                    norm_sym, norm_ox, norm_form = M.normalize_metal(raw, ox_raw)
                    agrees = (norm_sym == sym) and (norm_ox == ox)
                except ValueError as exc:
                    agrees = False
                    detail = f"normalize_metal raised: {exc}"
                if agrees is False:
                    issues.append("normalize_mismatch")
            labels = {s for s in grp["_state"] if not _missing(s)}
            for lab in labels:
                if M.normalize_metal(lab)[:2] != (sym, ox):
                    issues.append("g19_metal_state_roundtrip_mismatch")
            if form is not None:
                detail = (f"archive species token {raw!r} expanded to {sym}({M.ROMAN[ox]}); "
                          f"gen19 species_form {norm_form}")
            if "implausible_state" in issues:
                sel = df.index[(df["metal_symbol"] == sym) & (df["metal_oxidation_state"] == ox)]
                ready = df.loc[sel, "model_readiness"].value_counts().to_dict()
                dois = df.loc[sel, "doi_primary"].fillna("").value_counts().to_dict()
                detail = (f"{sym}({M.ROMAN[ox]}) is outside the accessible aqueous states "
                          f"{sorted(M.ACCESSIBLE_STATES.get(sym, []))}; gen19 descriptor row keeps "
                          f"element-level columns only; model_readiness={ready}; doi_primary={dois}; "
                          "these rows are g19_tier MODEL")
        g19_form = forms_by_key.get((sym, ox)) if sym is not None else None
        rows.append({
            "audit_section": "combination", "metal_raw": raw, "metal_oxidation_state_raw": ox_raw,
            "metal_symbol": sym, "metal_oxidation_state": ox, "metal_species_form": form,
            "metal_oxidation_state_source": "|".join(sorted({s for s in grp["_src"] if not _missing(s)})),
            "archive_state_plausible": "|".join(sorted({p for p in grp["_plaus"] if p})),
            "g19_metal_state": "|".join(sorted({s for s in grp["_state"] if not _missing(s)})),
            "normalize_symbol": norm_sym, "normalize_oxidation_state": norm_ox,
            "normalize_species_form": norm_form, "normalize_agrees": agrees,
            "g19_species_form": g19_form,
            "n_rows": len(grp), "n_target_rows": int(grp["_target"].sum()),
            "n_model_rows": int(grp["_model"].sum()),
            "issues": ";".join(issues), "detail": detail,
        })

    # unresolved metal rows: the archive label is empty; say what is left
    unresolved = df[df["metal_symbol"].isna()]
    raw_note = ""
    if ARCHIVE_RAW_ROWS.exists() and len(unresolved):
        rr = pd.read_parquet(ARCHIVE_RAW_ROWS, columns=["Metal_Name", "Metal_Oxidation_state",
                                                        "source_record_id"])
        rr = rr[rr["source_record_id"].isin(set(unresolved["source_record_id"].astype(str)))]
        names = {"" if _missing(v) else str(v).strip() for v in rr["Metal_Name"]}
        states = {"" if _missing(v) else str(v).strip() for v in rr["Metal_Oxidation_state"]}
        raw_note = (f"intermediate/raw_rows.parquet: {len(rr)} raw export rows behind all "
                    f"{len(unresolved)} unresolved records; "
                    f"distinct Metal_Name={sorted(names)}; distinct Metal_Oxidation_state={sorted(states)}")
    u = pd.DataFrame({
        "flags": unresolved["flags"].map(_stringify_list),
        "export_metals_queried": unresolved["export_metals_queried"].map(_stringify_list),
        "has_target": (unresolved["log_D"].notna()).to_numpy(),
        "model_readiness": unresolved["model_readiness"],
        "ini_comp_raw": unresolved["ini_comp_raw"],
    })
    for (flags, queried, tgt, ready), grp in u.groupby(["flags", "export_metals_queried", "has_target",
                                                       "model_readiness"], dropna=False, sort=True):
        top = grp["ini_comp_raw"].fillna("").value_counts().head(3)
        rows.append({
            "audit_section": "unresolved_metal", "metal_raw": None, "metal_oxidation_state_raw": None,
            "n_rows": len(grp), "n_target_rows": int(bool(tgt)) * len(grp), "n_model_rows": 0,
            "issues": f"flags={flags}", "detail": (
                f"export_metals_queried={queried} (export fan-out artefact, NOT the measured metal); "
                f"model_readiness={ready}; top ini_comp_raw: "
                + "; ".join(f"{k!r} x{v}" for k, v in top.items()) + (f"; {raw_note}" if raw_note else "")),
        })

    # symbols whose rows carry more than one state (element + state is the metal)
    for sym, grp in df[df["metal_symbol"].notna()].groupby("metal_symbol", sort=False):
        states = {_ox_int(o) for o in grp["metal_oxidation_state"]} - {None}
        model = grp[grp["g19_tier"] == "MODEL"]
        n_missing = int(grp["metal_oxidation_state"].isna().sum())
        issues = []
        if len(states) >= 2:
            issues.append("mixed_states")
        if n_missing:
            issues.append("state_missing_rows")
        rows.append({
            "audit_section": "state_mix_by_symbol", "metal_symbol": sym,
            "metal_oxidation_state_raw": None, "n_rows": len(grp),
            "n_target_rows": int((grp["log_D"].notna()).sum()), "n_model_rows": len(model),
            "issues": ";".join(issues),
            "detail": (f"n_known_states={len(states)}; all_rows={_state_counts(grp)}; "
                       f"model_rows={_state_counts(model) if len(model) else ''}; "
                       f"missing_state_rows={n_missing}; missing_state_model_rows="
                       f"{int(model['metal_oxidation_state'].isna().sum())}"),
        })
    out = pd.DataFrame(rows)
    order = ["audit_section", "metal_raw", "metal_oxidation_state_raw", "metal_symbol",
             "metal_oxidation_state", "metal_species_form", "metal_oxidation_state_source",
             "archive_state_plausible", "g19_metal_state", "normalize_symbol", "normalize_oxidation_state",
             "normalize_species_form", "normalize_agrees", "g19_species_form", "n_rows", "n_target_rows",
             "n_model_rows", "issues", "detail"]
    out = out.reindex(columns=order)
    for c in ("metal_oxidation_state", "normalize_oxidation_state"):
        out[c] = pd.array([None if _missing(v) else int(v) for v in out[c]], dtype="Int64")
    out["_z"] = out["metal_symbol"].map(lambda s: M.ATOMIC_NUMBER.get(s, 999) if isinstance(s, str) else 999)
    sec = {"combination": 0, "unresolved_metal": 1, "state_mix_by_symbol": 2}
    out["_s"] = out["audit_section"].map(sec)
    out = out.sort_values(["_s", "_z", "metal_oxidation_state", "metal_raw", "n_rows", "detail"],
                          na_position="first", kind="mergesort")
    return out.drop(columns=["_z", "_s"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ #
# 3. coverage and cross-checks
# ------------------------------------------------------------------------------------------------ #

def coverage(df: pd.DataFrame, table: pd.DataFrame) -> list[dict[str, Any]]:
    model = df[df["g19_tier"] == "MODEL"]
    keys = [M.descriptor_key(s, o) for s, o in zip(model["g19_metal"], model["g19_ox"])]
    tab = table.copy()
    tab["_key"] = [M.descriptor_key(s, o) for s, o in zip(tab["symbol"], tab["oxidation_state"])]
    missing = sorted(set(keys) - set(tab["_key"]))
    if missing:
        raise RuntimeError(f"MODEL rows without a descriptor row: {missing}")
    joined = pd.DataFrame({"_key": keys, "metal_category": model["metal_category"].to_numpy()}).merge(
        tab, on="_key", how="left", validate="many_to_one")
    rows = []
    cats = sorted(joined["metal_category"].unique()) + ["ALL"]
    for cat in cats:
        sub = joined if cat == "ALL" else joined[joined["metal_category"] == cat]
        for col in M.VALUE_COLUMNS:
            n_ok = int(sub[col].notna().sum())
            rows.append({"section": "coverage", "check": "model_row_nonmissing", "metal_category": cat,
                         "column": col, "n_model_rows": len(sub), "n_nonmissing": n_ok,
                         "fraction_nonmissing": n_ok / len(sub) if len(sub) else None,
                         "note": "MODEL rows (g19_tier) joined on (g19_metal, g19_ox); NA-state rows get "
                                 "the element-level row"})
    return rows


def crosscheck_archive(df: pd.DataFrame, table: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    tab = {(s, _ox_int(o)): r for s, o, r in zip(table["symbol"], table["oxidation_state"],
                                                 table.to_dict("records"))}
    sub = df[df["metal_symbol"].notna()].copy()
    sub["_ox"] = [_ox_int(o) for o in sub["metal_oxidation_state"]]
    src = "dataset_all_metals master_clean"
    for (sym, ox), grp in sub.groupby(["metal_symbol", "_ox"], dropna=False, sort=False):
        ox = _ox_int(ox)
        r = tab[(sym, ox)]
        cat = r["category"]
        vals = grp["ionic_radius_cn8_A"].dropna().round(6)
        modal = vals.mode().iloc[0] if len(vals) else None
        rows.append(_compare("archive_crosscheck", "ionic_radius_cn8_A_modal", sym, ox, "radius_cn8_A",
                             r["radius_cn8_A"], modal, src + " ionic_radius_cn8_A (modal)", numeric=True,
                             n_rows=len(grp), other_n_distinct=int(vals.nunique()), category=cat,
                             note=f"rows with archive radius: {len(vals)}; archive ionic_radius_status="
                                  + "|".join(sorted(set(grp['ionic_radius_status'].dropna())))))
        z = grp["atomic_number"].dropna()
        rows.append(_compare("archive_crosscheck", "atomic_number", sym, ox, "Z", r["Z"],
                             int(z.mode().iloc[0]) if len(z) else None, src + " atomic_number",
                             n_rows=len(grp), other_n_distinct=int(z.nunique()), category=cat))
        mc = grp["metal_category"].dropna()
        rows.append(_compare("archive_crosscheck", "metal_category", sym, ox, "category", r["category"],
                             mc.mode().iloc[0] if len(mc) else None, src + " metal_category",
                             n_rows=len(grp), other_n_distinct=int(mc.nunique()), category=cat))
        li = grp["lanthanide_index"].dropna()
        g19_li = None if _missing(r["series_index"]) or r["series"] != "Ln" else int(r["series_index"]) + 1
        rows.append(_compare("archive_crosscheck", "lanthanide_index_minus_1", sym, ox, "series_index",
                             g19_li, int(li.mode().iloc[0]) if len(li) else None,
                             src + " lanthanide_index (Z-56)", n_rows=len(grp),
                             other_n_distinct=int(li.nunique()), category=cat,
                             note="g19_value shown as series_index+1 (archive convention La=1)"))
        pl = {bool(v) for v in grp["metal_oxidation_state_plausible"] if v is not None and not _missing(v)}
        rows.append(_compare("archive_crosscheck", "oxidation_state_plausible", sym, ox, "state_plausible",
                             r["state_plausible"], (next(iter(pl)) if len(pl) == 1 else ("|".join(map(str, sorted(pl))) or None)),
                             src + " metal_oxidation_state_plausible", n_rows=len(grp),
                             other_n_distinct=len(pl), category=cat))
        forms = {v for v in grp["metal_species_form"] if not _missing(v)}
        if forms:
            rows.append(_compare("archive_crosscheck", "metal_species_form", sym, ox, "species_form",
                                 r["species_form"], "|".join(sorted(forms)), src + " metal_species_form",
                                 n_rows=int(grp["metal_species_form"].notna().sum()), category=cat,
                                 note="archive labels only the UO2+2 token; gen19 labels every U(VI) "
                                      "uranyl (vocabulary difference, same species)"))
    return rows


def crosscheck_archive_module(table: pd.DataFrame) -> list[dict[str, Any]]:
    chem = _load_module("_g19_sae_chem", ARCHIVE_CHEM)
    tab = {(s, _ox_int(o)): r for s, o, r in zip(table["symbol"], table["oxidation_state"],
                                                 table.to_dict("records"))}
    src = "dataset_all_metals/scripts/sae_chem.py"
    rows = []
    for (sym, ox), val in sorted(chem.SHANNON_CN8.items(), key=lambda kv: (M.ATOMIC_NUMBER[kv[0][0]], kv[0][1])):
        g = tab.get((sym, ox), {}).get("radius_cn8_A") if (sym, ox) in tab else M.SHANNON_RADII.get((sym, ox), {}).get(8)
        rows.append(_compare("archive_module_crosscheck", "SHANNON_CN8", sym, ox, "radius_cn8_A", g, val,
                             src + " SHANNON_CN8", numeric=True, category=M._ELEMENT_ROWS[sym][4]))
    for sym in sorted(chem.ATOMIC_NUMBER, key=M.ATOMIC_NUMBER.get):
        rows.append(_compare("archive_module_crosscheck", "ATOMIC_NUMBER", sym, None, "Z", M.ATOMIC_NUMBER[sym],
                             chem.ATOMIC_NUMBER[sym], src + " ATOMIC_NUMBER", category=M._ELEMENT_ROWS[sym][4]))
        rows.append(_compare("archive_module_crosscheck", "metal_category", sym, None, "category",
                             M._ELEMENT_ROWS[sym][4], chem.metal_category(sym), src + " metal_category()",
                             category=M._ELEMENT_ROWS[sym][4]))
    for sym in sorted(set(chem.PLAUSIBLE_OX) | set(M.ACCESSIBLE_STATES), key=M.ATOMIC_NUMBER.get):
        g = ",".join(map(str, sorted(M.ACCESSIBLE_STATES.get(sym, [])))) or None
        o = ",".join(map(str, sorted(chem.PLAUSIBLE_OX.get(sym, [])))) or None
        rows.append(_compare("archive_module_crosscheck", "PLAUSIBLE_OX", sym, None, "state_plausible", g, o,
                             src + " PLAUSIBLE_OX", category=M._ELEMENT_ROWS[sym][4],
                             note="gen19 adds Ac and Bk for the An(III) series rows" if sym in ("Ac", "Bk") else ""))
    return rows


def crosscheck_gen13(table: pd.DataFrame) -> list[dict[str, Any]]:
    g13 = _load_module("_g19_gen13_metals", GEN13_METALS)
    tab = {(s, _ox_int(o)): r for s, o, r in zip(table["symbol"], table["oxidation_state"],
                                                 table.to_dict("records"))}
    src = "generations/gen13_separation/gen13sep/metals.py"
    note = ("gen13 docstring calls these 'Shannon crystal radii' (Jordan 2023 SI) but the numbers are "
            "Shannon EFFECTIVE ionic radii (crystal radii are 0.14 A larger); label issue, not a value issue")
    rows = []
    for sym in g13.LANTHANIDES:
        r = tab[(sym, 3)]
        rows.append(_compare("gen13_crosscheck", "SHANNON_RADIUS_CN8", sym, 3, "radius_cn8_A", r["radius_cn8_A"],
                             g13.SHANNON_RADIUS_CN8[sym], src, numeric=True, category="lanthanide", note=note))
        rows.append(_compare("gen13_crosscheck", "SHANNON_RADIUS_CN9", sym, 3, "radius_cn9_A", r["radius_cn9_A"],
                             g13.SHANNON_RADIUS_CN9[sym], src, numeric=True, category="lanthanide", note=note))
        rows.append(_compare("gen13_crosscheck", "ATOMIC_NUMBER", sym, 3, "Z", r["Z"], g13.ATOMIC_NUMBER[sym],
                             src, category="lanthanide"))
        rows.append(_compare("gen13_crosscheck", "F_COUNT", sym, 3, "f_electron_count", r["f_electron_count"],
                             g13.F_COUNT[sym], src, category="lanthanide"))
    return rows


def crosscheck_gen11(table: pd.DataFrame) -> list[dict[str, Any]]:
    g11 = _load_module("_g19_gen11_metalrep", GEN11_METALREP)
    src = "src/lanthanide_separation/gen11/metalrep.py"
    rows = []
    for sym, val in g11.LN3_RADIUS_CN8.items():
        rows.append(_compare("gen11_crosscheck", "LN3_RADIUS_CN8", sym, 3, "radius_cn8_A",
                             M.SHANNON_RADII[(sym, 3)][8], val, src, numeric=True, category="lanthanide"))
    for sym, d in g11.LANTHANIDE_DESCRIPTORS.items():
        rows.append(_compare("gen11_crosscheck", "LANTHANIDE_DESCRIPTORS.Ionic Radius_metal", sym, 3,
                             "radius_cn8_A", M.SHANNON_RADII[(sym, 3)][8], d["Ionic Radius_metal"], src,
                             numeric=True, category="lanthanide"))
        rows.append(_compare("gen11_crosscheck", "LANTHANIDE_DESCRIPTORS.lanthanide_index", sym, 3,
                             "series_index", M.ATOMIC_NUMBER[sym] - 57 + 1, d["lanthanide_index"], src,
                             category="lanthanide", note="g19_value shown as series_index+1 (frozen convention La=1)"))
    for sym, (period, group, block) in sorted(g11.PERIODIC_TABLE.items(), key=lambda kv: M.ATOMIC_NUMBER[kv[0]]):
        name, p, g, b, cat = M._ELEMENT_ROWS[sym]
        rows.append(_compare("gen11_crosscheck", "PERIODIC_TABLE.period", sym, None, "period", p, period, src, category=cat))
        rows.append(_compare("gen11_crosscheck", "PERIODIC_TABLE.group", sym, None, "group", g, group, src, category=cat,
                             note=("convention: gen19 leaves the f-block group NA (task specification); gen11 "
                                   "uses the group-3 slot") if b == "f" else ""))
        rows.append(_compare("gen11_crosscheck", "PERIODIC_TABLE.block", sym, None, "block", b, block, src, category=cat))
        rows.append(_compare("gen11_crosscheck", "ATOMIC_NUMBER", sym, None, "Z", M.ATOMIC_NUMBER[sym],
                             g11.ATOMIC_NUMBER[sym], src, category=cat))
    for (sym, oxf_key), val in sorted(g11.OXO_SPECIES_CHARGE.items(),
                                      key=lambda kv: (M.ATOMIC_NUMBER[kv[0][0]], kv[0][1])):
        ox = int(oxf_key)
        mine = M._ion_values(sym, ox)
        rows.append(_compare("gen11_crosscheck", "OXO_SPECIES_CHARGE", sym, ox, "species_charge",
                             mine["species_charge"][0], int(val), src, category=M._ELEMENT_ROWS[sym][4],
                             note=f"gen19 species_form {mine['species_form'][0]}"))
    for r in table.to_dict("records"):
        sym, ox = r["symbol"], _ox_int(r["oxidation_state"])
        if sym not in g11.PERIODIC_TABLE:
            continue
        oxf = float("nan") if ox is None else float(ox)
        fc = g11.f_electron_count(sym, oxf)
        note = ""
        if r["species_form"] == "implausible_state":
            note = "gen19 withholds ion-level values for an implausible state; gen11 passes the state through"
        elif ox is None and not np.isnan(fc):
            note = ("convention: gen11 returns the state-independent f count for non-f-block elements; "
                    "gen19 keeps ion-level columns NA on NA-state rows")
        rows.append(_compare("gen11_crosscheck", "f_electron_count()", sym, ox, "f_electron_count",
                             r["f_electron_count"], None if np.isnan(fc) else int(fc), src,
                             category=r["category"], note=note))
        ec = g11.effective_charge(sym, oxf)
        note = ""
        if (sym, ox) == ("Pa", 5):
            note = "gen11 keeps Pa(V) at the oxidation state (documented limitation); gen19 leaves the charge NA"
        elif r["species_form"] == "implausible_state":
            note = "gen19 withholds ion-level values for an implausible state; gen11 passes the state through"
        elif ox is None:
            note = "both NA: no state recorded"
        rows.append(_compare("gen11_crosscheck", "effective_charge() (net species charge)", sym, ox,
                             "species_charge", r["species_charge"], None if np.isnan(ec) else int(ec), src,
                             category=r["category"], note=note))
    return rows


# ------------------------------------------------------------------------------------------------ #
# 4. sources document
# ------------------------------------------------------------------------------------------------ #

COLUMN_DOC: list[tuple[str, str, str]] = [
    ("symbol / oxidation_state", "key", "archive `metal_symbol` / `metal_oxidation_state`; series rows by definition. "
     "A metal is element + state; `oxidation_state` NA marks rows whose state the archive never recorded."),
    ("metal_state_label", "key", "`Nd(III)` form, identical to `gen19ct.data.load` `g19_metal_state`."),
    ("category_state_key", "key", "`lanthanide(III)`, `actinide(VI)`, ... -- the series x oxidation level of the "
     "brief section 15 decomposition e_shared + e_series + e_oxidation + e_element."),
    ("row_origin / in_archive", "meta", "`archive`, `ln3_series`, `an3_series` (joined with +)."),
    ("state_plausible", "meta", "state within `ACCESSIBLE_STATES` (= archive `PLAUSIBLE_OX`, plus Ac, Bk)."),
    ("Z, name, period", "element", "definition (IUPAC)."),
    ("group", "element", "IUPAC 1-18; NA for every f-block element (La-Lu, Ac-Cf) by the gen19 task convention."),
    ("block", "element", "Aufbau block; La, Lu and Ac labelled `f` as series members (gen11 convention)."),
    ("category", "element", "identical to the archive `metal_category` rule (`sae_chem.metal_category`)."),
    ("series, series_index", "element", "Ln: Z-57 (La=0..Lu=14); An: Z-89 (Ac=0..Cf=9). The archive "
     "`lanthanide_index` and the frozen bundle use Z-56 (La=1): offset by one, checked."),
    ("formal_charge", "ion", "definition: = oxidation state."),
    ("species_form", "ion", "`free_ion`; `uranyl`/`neptunyl`/`plutonyl`/`americyl` for An(V)/An(VI) of U, Np, Pu, "
     "Am; `pertechnetate` for Tc(VII); `protactinium_v_oxo` for Pa(V); `non_aqueous_state` for a Shannon-"
     "tabulated ion outside the aqueous states (Pa(III) series row); `implausible_state` (archive Sr(III))."),
    ("species_charge", "ion", "net charge of the species: bare cation = z; AnO2(+) = +1; AnO2(2+) = +2; "
     "TcO4- = -1; Pa(V) NA."),
    ("effective_charge", "ion", "Choppin electrostatic scale: bare cation Z_eff = z (definition); UO2 2+ 3.2 and "
     "NpO2+ 2.2 from Choppin & Rao, Radiochim. Acta 37 (1984) 143 (doi:10.1524/ract.1984.37.3.143); the other "
     "actinyls NA."),
    ("f_electron_count, d_electron_count, electron_configuration", "ion", "ionic-model arithmetic: Ln(z+) "
     "[Xe]4f^(Z-54-z); An(z+) [Rn]5f^(Z-86-z); d-block (n-1)d^(group-z) with the period-6 core including 4f14; "
     "s-block noble-gas core; p-block ions (In3+, Pb2+, Bi3+) from an explicit table. Formal counts, not "
     "spectroscopic ground terms."),
    ("radius_cn6_A, radius_cn8_A, radius_cn9_A", "ion", "Shannon, Acta Cryst. A32 (1976) 751, Table 1, "
     "effective ionic radii (IR, O2- = 1.40 A scale), angstrom; NA where Shannon gives no entry for that "
     "ion and CN."),
    ("electronegativity_pauling", "element (Pb(II) state-specific)", "Pauling scale; Allred (1961) / Huheey (1993) "
     "as quoted by WebElements, transcribed from Wikipedia 'Electronegativities of the elements (data page)', which "
     "gives one value per element without its oxidation state. Exception: the Pb(II) row uses Allred's +2-state "
     "value 1.87 (Wikipedia 'Electronegativity'); the tabulated 2.33 is Pb(IV). The Fe 1.83, Cr 1.66 and Mo 2.16 "
     "element values carry an unverified oxidation state (INFERRED: Allred's M(II) values) while the extraction "
     "chemistry is Fe(III), Cr(III), Mo(VI)."),
    ("hsab_class (+ hsab_class_basis)", "ion", "Pearson, JACS 85 (1963) 3533. Basis `explicit` = in the "
     "hard/borderline/soft table as reproduced by Wikipedia 'HSAB theory'; `recalled` = standard Pearson "
     "list entry not re-checked against the printed table; `analogy` = INFERRED within a Pearson class."),
    ("polarizability_atom_au", "element", "ELEMENT property of the neutral atom (Schwerdtfeger & Nagle 2018 table "
     "requested). All NA in this build -- see below."),
]

UNSURE: list[str] = [
    "**Polarizability (all rows NA).** The Schwerdtfeger & Nagle table (Mol. Phys. 117 (2019) 1200, "
    "doi:10.1080/00268976.2018.1535143) could not be opened from the build host (tandfonline and "
    "ingentaconnect HTTP 403; ctcp.massey.ac.nz, researchgate.net and github.com unresolvable). No value "
    "was written from memory. To fill: transcribe Table 1 recommended alpha_D (a.u.) with its uncertainty "
    "into `metals.py` and add a `_source` citing the table row.",
    "**Pauling electronegativity NA for Pm, Eu, Yb** (the source gives only a 1.1-1.2 range) **and Tb** "
    "(not in Allred 1961; one-significant-figure estimates of 1.1 and 1.2 circulate; 1.1 would break the "
    "Gd 1.20 -> Dy 1.22 trend). Low-precision estimates kept but not trustworthy for within-series "
    "features: Ac 1.1, Th 1.3, Pa 1.5, Hf 1.3, Am/Cm/Bk/Cf 1.3. Tc 1.9 (WebElements) may differ from "
    "CRC/Lange (2.10 was reported in one read of the data page, not reproduced in a second read).",
    "**Shannon radii verification.** Ln(III) CN6/CN8/CN9 agree with gen13 (CN8, CN9), gen11 (CN8) and the "
    "archive (CN8). Non-Ln CN8 values agree with the archive's own `SHANNON_CN8`. CN6 values of the actinide, "
    "Pd, Tc, Bi, In, Cd, Pb, Hf, Zr, Y, Sc, Ca, Sr, Ba ions were compared with the Wikipedia effective-radii "
    "table (read through a summarising fetch whose column alignment was off by one charge for actinides; "
    "after re-alignment every value matched). CN9 values outside the lanthanides (Y 1.075, Ca 1.18, Sr 1.31, "
    "Ba 1.47, Pb 1.35, Zr 0.89, Th 1.09, U(IV) 1.05, Pa(V) 0.95) rest on a single transcription and were not "
    "independently re-checked in this build.",
    "**Gd(III) CN6 = 0.938 A.** One summarising read of the web effective-radii table returned 93.5 pm, "
    "but the same page's crystal radius (107.8 pm = IR + 14 pm) implies 0.938 A, which matches the recalled "
    "Shannon value; kept 0.938, flagged (the CN6 strict La->Lu decrease holds either way).",
    "**Ac(III) CN6 = 1.12 A** (Shannon 1976; consistent with the Wikipedia crystal radius 1.26 A = IR + 0.14). "
    "The same web table's effective-radius row returned 1.065 A (CN6) / 1.22 A (CN9) for Ac(III), apparently "
    "from a later source; kept at the Shannon value, flagged.",
    "**Am(VI)** has no Shannon entry; all Am(VI) radii NA. **Np(V), Np(VI), Pu(III), Pu(VI), Cm(III), Cf(III), "
    "Pd(II), Pd(IV), Tc(VII)** have CN6 only.",
    "**HSAB classes by analogy (INFERRED):** Ba(II), Y(III), Hf(IV), Np(IV), all An(III) rows, Np(VI), Pu(VI), "
    "Am(VI), and Np(V) (weakest: NpO2+ effective charge 2.2). **Recalled, not re-checked:** Ca(II), Sr(II), "
    "In(III), Zr(IV), Pu(IV), U(VI) (UO2 2+), Cd(II) soft, Bi(III) borderline. NA: Tc(VII) (anion), Pa(V), "
    "Pd(IV), and every NA-state row.",
    "**Effective charges.** Choppin & Rao (1984) values quoted via OSTI 1366418 (NpO2+ 2.2, UO2 2+ 3.2); the "
    "class-wide values for NpO2 2+, PuO2 2+, AmO2 2+ are NA (variants 3.2/3.3 circulate). Free-ion "
    "effective charge = z is the scale's anchor, not a measurement.",
    "**Species forms are formal.** Hydrolysis (Zr, Hf, Pa, Pu(IV)) and chloro/nitrato complexation (Pd, "
    "U(VI)) are condition properties and are not encoded here. Redox potentials (brief section 4.1) are not "
    "built in this task.",
]


def sources_md(table: pd.DataFrame, alias: pd.DataFrame, cov: pd.DataFrame) -> str:
    L: list[str] = []
    L.append("# gen19 metal descriptors -- sources and verification")
    L.append("")
    L.append("Generated by `generations/gen19_chem_transfer/scripts/g19_build_metals.py` from "
             "`gen19ct/chemistry/metals.py`; every number below is computed from the files this script writes "
             "(`descriptors/metals.csv`, `data_audit/metal_alias_audit.csv`, `data_audit/metal_descriptor_coverage.csv`).")
    L.append("")
    L.append("Rebuild: `PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer "
             ".venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_metals.py`")
    L.append("")
    n_arch = int(table["in_archive"].sum())
    L.append(f"Table: {len(table)} rows x {table.shape[1]} columns; {n_arch} rows are archive (symbol, state) "
             f"combinations ({int((table['in_archive'] & table['oxidation_state'].isna()).sum())} with the state "
             f"missing), {int(table['row_origin'].str.contains('ln3').sum())} Ln(III) and "
             f"{int(table['row_origin'].str.contains('an3').sum())} An(III) series rows (overlaps counted in both).")
    L.append("")
    L.append("Rule: every value column `c` has `c_source`; a missing value carries `NA_REASON: ...` there. "
             "Ion-level columns are NA on rows with no recorded oxidation state.")
    L.append("")
    L.append("## Columns")
    L.append("")
    L.append("| column | level | source / definition |")
    L.append("|---|---|---|")
    for c, lvl, s in COLUMN_DOC:
        L.append(f"| `{c}` | {lvl} | {s} |")
    L.append("")
    L.append("## Missing values per column (descriptor table)")
    L.append("")
    L.append("| column | non-missing rows | missing rows |")
    L.append("|---|---|---|")
    for c in M.VALUE_COLUMNS:
        n = int(table[c].notna().sum())
        L.append(f"| `{c}` | {n} | {len(table) - n} |")
    L.append("")
    L.append("## Values I was unsure of, and what was verified")
    L.append("")
    for u in UNSURE:
        L.append(f"- {u}")
    L.append("")
    L.append("## Cross-check summary (data_audit/metal_descriptor_coverage.csv)")
    L.append("")
    cc = cov[cov["section"] != "coverage"]
    summ = cc.groupby(["section", "status"]).size().unstack(fill_value=0)
    cols = list(summ.columns)
    L.append("| section | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for sec, r in summ.iterrows():
        L.append(f"| {sec} | " + " | ".join(str(int(r[c])) for c in cols) + " |")
    L.append("")
    dis = cc[cc["status"] == "DISAGREE"]
    L.append(f"Every `DISAGREE` row ({len(dis)}):")
    L.append("")
    if len(dis):
        L.append("| section | check | symbol | state | g19 | other | note |")
        L.append("|---|---|---|---|---|---|---|")
        for r in dis.to_dict("records"):
            L.append(f"| {r['section']} | {r['check']} | {r['symbol']} | {_fmt(r['oxidation_state'])} | "
                     f"{r['g19_value']} | {r['other_value']} | {r['note'] or ''} |")
    else:
        L.append("(none)")
    L.append("")
    one = cc[cc["status"].isin(["g19_only", "other_only"])]
    L.append(f"Every one-side-missing row ({len(one)}), grouped:")
    L.append("")
    if len(one):
        L.append("| section | check | status | n | symbol(state) | note |")
        L.append("|---|---|---|---|---|---|")
        for (sec, chk, st, note), g in one.fillna({"note": ""}).groupby(["section", "check", "status", "note"], sort=True):
            who = ", ".join(f"{s}({_fmt(o) or 'NA'})" for s, o in zip(g["symbol"], g["oxidation_state"]))
            L.append(f"| {sec} | {chk} | {st} | {len(g)} | {who} | {note} |")
    L.append("")
    rad = cc[cc["column"].astype(str).str.startswith("radius") & cc["status"].isin(["DISAGREE"])]
    L.append(f"Radius disagreements > {RADIUS_TOL_A} A: {len(rad)}.")
    oo = cc[cc["status"].isin(["g19_only", "other_only"]) & cc["column"].astype(str).str.startswith("radius")]
    L.append(f"Radius presence mismatches (one side missing): {len(oo)} -- listed in the CSV "
             "(`status` g19_only / other_only).")
    L.append("")
    L.append("## Coverage of MODEL rows (fraction non-missing, by archive metal_category)")
    L.append("")
    cv = cov[cov["section"] == "coverage"].pivot(index="column", columns="metal_category",
                                                 values="fraction_nonmissing")
    cv = cv.reindex(list(M.VALUE_COLUMNS))
    cats = list(cv.columns)
    nmodel = cov[(cov["section"] == "coverage") & (cov["column"] == "Z")].set_index("metal_category")["n_model_rows"]
    L.append("| column | " + " | ".join(f"{c} (n={int(nmodel[c])})" for c in cats) + " |")
    L.append("|---|" + "---|" * len(cats))
    for col, r in cv.iterrows():
        L.append(f"| `{col}` | " + " | ".join(f"{r[c]:.3f}" for c in cats) + " |")
    L.append("")
    L.append("## Alias audit summary (data_audit/metal_alias_audit.csv)")
    L.append("")
    comb = alias[alias["audit_section"] == "combination"]
    L.append(f"- label combinations: {len(comb)}; rows {int(comb['n_rows'].sum())}")
    for tag in ("metal_unresolved", "state_missing", "species_token_expanded", "implausible_state",
                "normalize_mismatch", "g19_metal_state_roundtrip_mismatch"):
        m = comb[comb["issues"].fillna("").str.contains(tag)]
        L.append(f"- `{tag}`: {len(m)} combinations, {int(m['n_rows'].sum())} rows, "
                 f"{int(m['n_model_rows'].sum())} MODEL rows")
    mix = alias[(alias["audit_section"] == "state_mix_by_symbol") & alias["issues"].fillna("").str.contains("mixed_states")]
    L.append(f"- symbols with >= 2 recorded states: {len(mix)} -- " + "; ".join(
        f"{r['metal_symbol']} ({r['detail'].split('; ')[1]})" for r in mix.to_dict("records")))
    L.append("")
    L.append("Leave-metal-out folds must key on `metal_state_label` (element + state), and decide explicitly "
             "what to do with the NA-state rows of each symbol (they cannot be assigned a state here).")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    with Run("g19_build_metals", args={}, seed=None) as run:
        run.inputs(paths.ARCHIVE_MASTER, ARCHIVE_RAW_ROWS, METALS_PY, GEN13_METALS, GEN11_METALREP, ARCHIVE_CHEM)
        df = load_archive(copy=False)
        keys = archive_keys(df)
        table = M.build_descriptor_table(keys, archive_keys=keys)
        dup = table.duplicated(["symbol", "oxidation_state"]).sum()
        if dup:
            raise RuntimeError(f"{dup} duplicate (symbol, oxidation_state) keys")
        write_csv(table, OUT_TABLE)

        alias = alias_audit(df, table)
        write_csv(alias, OUT_ALIAS)

        rows = coverage(df, table)
        rows += crosscheck_archive(df, table)
        rows += crosscheck_archive_module(table)
        rows += crosscheck_gen13(table)
        rows += crosscheck_gen11(table)
        cov = pd.DataFrame(rows).reindex(columns=COVERAGE_COLUMNS)
        for c in ("oxidation_state", "other_n_distinct", "n_rows", "n_model_rows", "n_nonmissing"):
            cov[c] = pd.array([None if _missing(v) else int(v) for v in cov[c]], dtype="Int64")
        write_csv(cov, OUT_COVERAGE)

        write_text(OUT_SOURCES, sources_md(table, alias, cov))
        run.outputs(OUT_TABLE, OUT_ALIAS, OUT_COVERAGE, OUT_SOURCES)

        cc = cov[cov["section"] != "coverage"]
        print(f"metals.csv rows={len(table)} cols={table.shape[1]}")
        print(cc.groupby(["section", "status"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
