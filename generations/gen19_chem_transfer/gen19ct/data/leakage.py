"""``data/leakage.py`` -- reusable leakage detectors and fold guards (brief sections 12 and 27).

Every function takes an archive-shaped frame (the output of ``gen19ct.data.load.load_archive`` or a
subset of it; synthetic frames in the tests carry only the columns a function needs) and returns a
plain DataFrame / dict.  Nothing here fits a predictive model, and nothing here reads ``log_D`` to
*define* a key: the value is used only to measure how close two already-matched records are, and in
:func:`provenance_group_predictability` as a descriptive leave-one-out group-mean statistic.

Detectors
---------
:func:`exact_duplicate_groups`
    the archive's own duplicate groups (class A/B collapsed, C/E/F kept) and checks that
    ``is_canonical_row`` means what the schema says (false only for non-representative A/B members).
:func:`near_duplicate_key` / :func:`near_duplicate_pairs` / :func:`near_duplicate_pair_table`
    the gen11 ``cell_key`` (``src/lanthanide_separation/gen11/leakage_audit.py:177``) ported to the
    multi-metal archive: structure *system* key, metal *state* (element + oxidation state), primary acid,
    solvent key, acid M, extractant M, temperature at 6 or 3 significant figures; a missing value is the
    literal token ``NA`` and matches another ``NA`` (the inclusive reading -- a sweep that over-reports
    and is then filtered).  ``strict=True`` adds metal concentration, O/A, modifier (name, M), complexant
    signature (name@M), nitrate M and holdback agent (structure, M).
:func:`cross_publication_copies`
    same key, ``|delta log D| <= 0.005``, different ``g19_publication_id``.
:func:`double_digitisation`
    same publication, same 3 s.f. key, ``|delta log D| <= 0.02``, not one A/B group.
:func:`doi_source_multiplicity` / :func:`corrected_publication_id` / :func:`publication_link_components`
    one DOI under several fold groups / sub-sources / authors; one title under several (raw) DOIs; and the
    union-find publication groups that would remove each kind of split.
:func:`metal_alias_split_risk`
    element vs element+state labels (and export file vs measured metal) that could put one experiment on
    both sides of a leave-metal-out fold.
:func:`metadata_target_leak`
    D written inside free text; provenance groupings that "predict" log D; serial correlation of log D
    along the archive id.
:func:`cross_boundary_burden`
    how many rows / publications must share a fold group so no near-duplicate crosses a V1 boundary.
:func:`bundle_overlap`
    frozen 14-lanthanide bundle rows vs the archive rows they claim to be.

Guards (imported by the fold builders)
--------------------------------------
:func:`pair_isolation_check` and :func:`fold_isolation_check` raise ``AssertionError`` on a violation
by default and always return a machine-readable report.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

AB_CLASSES: tuple[str, ...] = ("A_EXACT_DATABASE_DUPLICATE", "B_SAME_MEASUREMENT_DIFF_PROV")
SAFE_SELF_CITATION = "10.1021/jacs.5c19738"

KEY_CATEGORICAL: tuple[str, ...] = ("extractant_system_key", "g19_metal", "g19_ox", "acid_primary", "solvent_key")
KEY_NUMERIC: tuple[str, ...] = ("acid_concentration_M", "extractant_primary_concentration_M", "temperature_C")
STRICT_CATEGORICAL: tuple[str, ...] = ("modifier_name", "complexant_signature", "holdback_smiles_canonical")
STRICT_NUMERIC: tuple[str, ...] = ("metal_concentration_M", "phase_ratio_org_aq", "modifier_concentration_M",
                                   "nitrate_concentration_M", "holdback_concentration_M")

COPY_TOLERANCE_LOG_D = 0.005
DOUBLE_DIGITISATION_TOLERANCE_LOG_D = 0.02

ID_COL = "canonical_measurement_id"
PUB_COL = "g19_publication_id"


# --------------------------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------------------------- #

def _col(df: pd.DataFrame, name: str, default: Any = None) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=object)
    return np.full(len(df), default, dtype=object)


def _is_missing(v: Any) -> bool:
    if v is None or v is pd.NA:
        return True
    return isinstance(v, float) and np.isnan(v)


def _as_list(value: Any) -> list[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [v for v in value if not _is_missing(v)]
    return [value]


def _short(t: Any, n: int = 160) -> str:
    s = " ".join(str(t).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def union_components(edges_a: Iterable[Any], edges_b: Iterable[Any],
                     nodes: Iterable[Any] = ()) -> dict[Any, Any]:
    """Union-find over an edge list: ``node -> component representative`` (smallest node by ``str``).

    ``nodes`` adds isolated nodes (they map to themselves)."""
    parent: dict[Any, Any] = {}

    def find(x: Any) -> Any:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for n in nodes:
        find(n)
    for a, b in zip(edges_a, edges_b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if str(ra) <= str(rb):
                parent[rb] = ra
            else:
                parent[ra] = rb
    return {x: find(x) for x in list(parent)}


# --------------------------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------------------------- #

def round_sig(values: Any, digits: int) -> np.ndarray:
    """Round to ``digits`` significant figures; NaN stays NaN, 0 stays 0 (gen11 ``_round_sig``)."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    out = np.full(x.shape, np.nan)
    out[np.isfinite(x) & (x == 0)] = 0.0
    finite = np.isfinite(x) & (x != 0)
    if finite.any():
        scale = np.power(10.0, np.floor(np.log10(np.abs(x[finite]))))
        out[finite] = np.round(x[finite] / scale, digits - 1) * scale
    return out


def _num_tokens(values: Any, digits: int) -> np.ndarray:
    r = round_sig(values, digits)
    return np.array(["NA" if not np.isfinite(v) else format(v, ".12g") for v in r], dtype=object)


def _cat_tokens(series: pd.Series) -> np.ndarray:
    vals = series.to_numpy(dtype=object)
    return np.array(["NA" if _is_missing(v) or (isinstance(v, str) and v == "") else str(v) for v in vals],
                    dtype=object)


def near_duplicate_key(df: pd.DataFrame, sig: int = 6, strict: bool = False, *,
                       include_metal_state: bool = True, include_structure: bool = True) -> pd.Series:
    """String key of the experimental cell at ``sig`` significant figures (``NA`` matches ``NA``).

    ``g19_ox`` is rounded like a number so ``3.0`` and ``3`` agree.  ``include_metal_state=False`` keeps
    the element but wildcards the oxidation state (used to find the same conditions recorded under two
    state labels).  Missing columns raise ``KeyError`` -- a key silently built from fewer fields would
    over-report.
    """
    if not isinstance(sig, (int, np.integer)) or not 1 <= int(sig) <= 15:
        raise ValueError(f"sig must be an integer in 1..15, got {sig!r}")
    parts: list[np.ndarray] = []
    for col in KEY_CATEGORICAL:
        if col == "extractant_system_key" and not include_structure:
            continue
        if col == "g19_ox":
            parts.append(_num_tokens(df[col], 6) if include_metal_state else np.full(len(df), "*", dtype=object))
            continue
        parts.append(_cat_tokens(df[col]))
    for col in KEY_NUMERIC:
        parts.append(_num_tokens(df[col], sig))
    if strict:
        for col in STRICT_CATEGORICAL:
            parts.append(_cat_tokens(df[col]))
        for col in STRICT_NUMERIC:
            parts.append(_num_tokens(df[col], sig))
    joined = ["|".join(v) for v in zip(*parts)] if len(df) else []
    return pd.Series(joined, index=df.index, dtype=object)


def key_id(keys: pd.Series | Iterable[str]) -> pd.Series:
    """A short stable hash of a key string (for CSVs; the key itself holds long SMILES)."""
    return pd.Series(list(keys), dtype=object).map(lambda k: "k_" + hashlib.sha1(str(k).encode()).hexdigest()[:12])


def _pairs_within_groups(labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Positional (i, j), i < j, for every pair of rows sharing a label (None/NaN labels skipped)."""
    codes, _ = pd.factorize(pd.Series(labels, dtype=object), use_na_sentinel=True)
    if len(codes) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    for block in np.split(order, boundaries):
        if len(block) < 2 or codes[block[0]] < 0:
            continue
        a, b = np.triu_indices(len(block), k=1)
        left.append(block[a])
        right.append(block[b])
    if not left:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(left), np.concatenate(right)


def near_duplicate_pairs(df: pd.DataFrame, sig: int = 6, strict: bool = False) -> pd.DataFrame:
    """Every pair of rows sharing :func:`near_duplicate_key` (all values, any log D).

    Columns: ``idx_a``/``idx_b`` (index labels, ``idx_a`` first in frame order), ``id_a``/``id_b``,
    ``pub_a``/``pub_b``, ``same_publication``, ``log_D_a``/``log_D_b``, ``abs_delta_log_D`` (NaN when
    either is missing), ``same_duplicate_group``, ``both_ab_group``, tiers, classes, ``key_id``,
    ``strict_key_equal`` (at the same ``sig``; NaN when the strict columns are absent), ``sig_figs``.
    """
    keys = near_duplicate_key(df, sig=sig, strict=strict)
    i, j = _pairs_within_groups(keys)
    return _enrich_pairs(df, i, j, keys, sig, strict)


def _enrich_pairs(df: pd.DataFrame, i: np.ndarray, j: np.ndarray, keys: pd.Series, sig: int,
                  strict: bool) -> pd.DataFrame:
    idx = df.index.to_numpy()
    logd = (pd.to_numeric(df["log_D"], errors="coerce").to_numpy(dtype=float) if "log_D" in df.columns
            else np.full(len(df), np.nan))
    ids = _col(df, ID_COL)
    pubs = _col(df, PUB_COL)
    dgrp = _col(df, "duplicate_group_id")
    dcls = _col(df, "duplicate_class")
    tier = _col(df, "g19_tier")
    kvals = keys.to_numpy(dtype=object)
    out = pd.DataFrame({
        "sig_figs": np.full(len(i), sig, dtype=int),
        "key_type": np.full(len(i), "strict" if strict else "base", dtype=object),
        "key_id": key_id(kvals[i]).to_numpy(dtype=object) if len(i) else np.array([], dtype=object),
        "idx_a": idx[i], "idx_b": idx[j], "id_a": ids[i], "id_b": ids[j],
        "pub_a": pubs[i], "pub_b": pubs[j], "log_D_a": logd[i], "log_D_b": logd[j],
        "tier_a": tier[i], "tier_b": tier[j], "duplicate_class_a": dcls[i], "duplicate_class_b": dcls[j],
    })
    out["same_publication"] = np.array([a == b for a, b in zip(pubs[i], pubs[j])], dtype=bool)
    out["abs_delta_log_D"] = np.abs(logd[i] - logd[j])
    same_group = np.array([(not _is_missing(a)) and a == b for a, b in zip(dgrp[i], dgrp[j])], dtype=bool)
    out["same_duplicate_group"] = same_group
    out["both_ab_group"] = same_group & np.array([c in AB_CLASSES for c in dcls[i]], dtype=bool)
    needed = set(STRICT_CATEGORICAL) | set(STRICT_NUMERIC)
    if strict:
        out["strict_key_equal"] = np.ones(len(i), dtype=bool)
    elif needed.issubset(df.columns):
        skeys = near_duplicate_key(df, sig=sig, strict=True).to_numpy(dtype=object)
        out["strict_key_equal"] = np.array(skeys[i] == skeys[j], dtype=bool)
    else:
        out["strict_key_equal"] = np.full(len(i), np.nan)
    return out


def near_duplicate_pair_table(df: pd.DataFrame, sigs: tuple[int, ...] = (6, 3)) -> pd.DataFrame:
    """One row per pair that shares the base key at ANY of ``sigs``, with a flag per tier.

    Columns: ids, publications, tiers, classes, ``log_D_a/b``, ``abs_delta_log_D``, ``same_publication``,
    ``both_ab_group``, and per tier ``near_dup_sig<s>`` / ``strict_key_equal_sig<s>`` / ``key_id_sig<s>``.
    A pair equal at 6 s.f. is normally equal at 3 s.f. too, but not always (rounding boundaries), so the
    flags are computed, not implied.
    """
    per: list[pd.DataFrame] = []
    for s in sigs:
        p = near_duplicate_pairs(df, sig=s)
        per.append(p)
    base_cols = ["idx_a", "idx_b", "id_a", "id_b", "pub_a", "pub_b", "tier_a", "tier_b", "duplicate_class_a",
                 "duplicate_class_b", "log_D_a", "log_D_b", "abs_delta_log_D", "same_publication",
                 "same_duplicate_group", "both_ab_group"]
    allp = pd.concat([p[base_cols] for p in per], ignore_index=True).drop_duplicates(["idx_a", "idx_b"])
    allp = allp.sort_values(["idx_a", "idx_b"], kind="stable").reset_index(drop=True)
    for s, p in zip(sigs, per):
        tag = p.set_index(["idx_a", "idx_b"])
        mi = pd.MultiIndex.from_frame(allp[["idx_a", "idx_b"]])
        allp[f"near_dup_sig{s}"] = mi.isin(tag.index)
        allp[f"strict_key_equal_sig{s}"] = tag["strict_key_equal"].reindex(mi).to_numpy()
        allp[f"key_id_sig{s}"] = tag["key_id"].reindex(mi).to_numpy()
    allp["both_model"] = (allp["tier_a"] == "MODEL") & (allp["tier_b"] == "MODEL")
    return allp


# --------------------------------------------------------------------------------------------- #
# 1. exact duplicates
# --------------------------------------------------------------------------------------------- #

def _exp_num(ids: pd.Series) -> pd.Series:
    return pd.to_numeric(ids.astype("string").str.extract(r"(\d+)$", expand=False), errors="coerce")


def exact_duplicate_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Group table for every ``duplicate_group_id`` with more than one row, and semantics checks.

    Returns ``(groups, checks)``; ``checks`` maps a check name to ``{"pass": bool, ...counts}``.
    """
    d = df
    sizes = d.groupby("duplicate_group_id", sort=True)["duplicate_group_id"].transform("size")
    multi = d[sizes > 1].assign(_logd=pd.to_numeric(d.loc[sizes > 1, "log_D"], errors="coerce"),
                                _model=lambda x: (x["g19_tier"] == "MODEL") if "g19_tier" in x.columns else False)
    g = multi.groupby("duplicate_group_id", sort=True)
    rep_is_canon = multi.assign(_rc=multi["group_representative_id"].eq(multi[ID_COL]) & multi["is_canonical_row"]
                                ).groupby("duplicate_group_id")["_rc"].any()
    groups = pd.DataFrame({
        "duplicate_class": g["duplicate_class"].first(),
        "n_rows": g.size(),
        "n_canonical_rows": g["is_canonical_row"].sum().astype(int),
        "group_representative_id": g["group_representative_id"].first(),
        "representative_is_canonical_member": rep_is_canon,
        "n_identity_hashes": g["identity_hash"].nunique(),
        "n_publications": g[PUB_COL].nunique() if PUB_COL in multi.columns else np.nan,
        "n_entry_authors": g["entry_author"].nunique() if "entry_author" in multi.columns else np.nan,
        "n_log_D_finite": g["_logd"].count(),
        "log_D_min": g["_logd"].min(), "log_D_max": g["_logd"].max(),
        "n_model_tier": g["_model"].sum().astype(int),
    })
    groups["log_D_spread"] = groups["log_D_max"] - groups["log_D_min"]
    groups = groups.reset_index()
    if "g19_metal_state" in multi.columns:
        ms = g["g19_metal_state"].agg(lambda s: ";".join(sorted({str(v) for v in s if isinstance(v, str)})))
        groups["metal_states"] = groups["duplicate_group_id"].map(ms)
    if "duplicate_class_reason" in multi.columns:
        groups["duplicate_class_reason"] = groups["duplicate_group_id"].map(g["duplicate_class_reason"].first())
    groups = groups.sort_values(["duplicate_class", "n_rows", "duplicate_group_id"],
                                ascending=[True, False, True], kind="stable").reset_index(drop=True)

    checks: dict[str, Any] = {}
    canon = d["is_canonical_row"].astype(bool)
    noncanon = d[~canon]
    checks["noncanonical_only_in_AB"] = {
        "pass": bool(noncanon["duplicate_class"].isin(AB_CLASSES).all()),
        "n_noncanonical": int(len(noncanon)),
        "n_noncanonical_outside_AB": int((~noncanon["duplicate_class"].isin(AB_CLASSES)).sum())}
    ab = d[d["duplicate_class"].isin(AB_CLASSES)]
    canon_per = ab.groupby("duplicate_group_id")["is_canonical_row"].sum()
    checks["one_canonical_per_AB_group"] = {"pass": bool((canon_per == 1).all()), "n_AB_groups": int(len(canon_per)),
                                            "n_groups_violating": int((canon_per != 1).sum())}
    abc = ab[ab["is_canonical_row"].astype(bool)]
    checks["AB_canonical_is_representative"] = {
        "pass": bool((abc["group_representative_id"] == abc[ID_COL]).all()),
        "n_violating": int((abc["group_representative_id"] != abc[ID_COL]).sum())}
    num = _exp_num(d[ID_COL])
    lowest = num.groupby(d["duplicate_group_id"]).transform("min")
    rep_rows = d[d["group_representative_id"] == d[ID_COL]]
    checks["representative_is_lowest_archive_id"] = {
        "pass": bool((num[rep_rows.index] == lowest[rep_rows.index]).all()),
        "n_violating": int((num[rep_rows.index] != lowest[rep_rows.index]).sum())}
    non_ab = d[~d["duplicate_class"].isin(AB_CLASSES)]
    checks["non_AB_rows_all_canonical"] = {"pass": bool(non_ab["is_canonical_row"].astype(bool).all()),
                                           "n_violating": int((~non_ab["is_canonical_row"].astype(bool)).sum())}
    if "model_readiness" in d.columns:
        red = d["model_readiness"].eq("D_redundant_duplicate")
        checks["redundant_readiness_equals_noncanonical"] = {"pass": bool((red == ~canon).all()),
                                                             "n_violating": int((red != ~canon).sum())}
    hash_per_group = d.groupby("duplicate_group_id")["identity_hash"].nunique()
    group_per_hash = d.groupby("identity_hash")["duplicate_group_id"].nunique()
    checks["identity_hash_one_to_one_with_group"] = {
        "pass": bool((hash_per_group == 1).all() and (group_per_hash == 1).all()),
        "n_groups": int(len(hash_per_group)), "n_hashes": int(len(group_per_hash))}
    if "duplicate_group_size" in d.columns:
        ok = d.groupby("duplicate_group_id")["duplicate_group_id"].transform("size") == d["duplicate_group_size"]
        checks["duplicate_group_size_consistent"] = {"pass": bool(ok.all()), "n_violating": int((~ok).sum())}
    checks["class_constant_within_group"] = {
        "pass": bool((d.groupby("duplicate_group_id")["duplicate_class"].nunique() == 1).all())}
    ab_logd = pd.to_numeric(ab["log_D"], errors="coerce")
    ab_spread = (ab_logd.groupby(ab["duplicate_group_id"]).max() - ab_logd.groupby(ab["duplicate_group_id"]).min())
    checks["AB_groups_identical_log_D"] = {"pass": bool((ab_spread.fillna(0) <= 1e-12).all()),
                                           "max_spread": float(ab_spread.max()) if ab_spread.notna().any() else 0.0}
    if PUB_COL in d.columns:
        pubs_per = ab.groupby("duplicate_group_id")[PUB_COL].nunique()
        checks["AB_groups_spanning_publications"] = {"pass": True, "informational": True,
                                                     "n_groups": int((pubs_per > 1).sum())}
    if "g19_tier" in d.columns:
        bad = d["g19_tier"].eq("MODEL") & ~canon
        checks["model_tier_has_no_noncanonical_rows"] = {"pass": not bool(bad.any()), "n_violating": int(bad.sum())}
    counts = d["duplicate_class"].value_counts().to_dict()
    checks["class_counts_rows"] = {"pass": True, "informational": True, **{str(k): int(v) for k, v in counts.items()}}
    return groups, checks


# --------------------------------------------------------------------------------------------- #
# 2-4. cross-publication copies, double digitisation
# --------------------------------------------------------------------------------------------- #

def cross_publication_copies(df: pd.DataFrame, sig: int = 6, tol: float = COPY_TOLERANCE_LOG_D,
                             pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Near-duplicate pairs (``sig``) with ``|delta log D| <= tol`` from different publications."""
    p = near_duplicate_pairs(df, sig=sig) if pairs is None else pairs[pairs["sig_figs"] == sig]
    keep = (~p["same_publication"].astype(bool)) & p["abs_delta_log_D"].le(tol + 1e-12)
    return p[keep].reset_index(drop=True)


_LOC_RX = re.compile(r"^\s*(?:from\s+)?(?:fig(?:ure|urre)?s?|tables?|graph)\s*\.?\s*s?i?\s*\d+[a-z]?", re.IGNORECASE)


def location_token(df: pd.DataFrame) -> pd.Series:
    """Normalised figure/table label: ``data_location`` when present, else the leading ``fig2`` /
    ``table1`` token of ``comments_raw``; ``None`` when neither exists.  ``"Figure 3"`` and ``"fig3"``
    both become ``"fig3"``; ``"Figure S4A"`` and ``"figs4a"`` both become ``"figs4a"`` (never ``"fig4a"``);
    ``"Table S1"`` and ``"tables1"`` both become ``"tables1"``."""
    out: list[str | None] = []
    for loc, com in zip(_col(df, "data_location"), _col(df, "comments_raw")):
        text = loc if isinstance(loc, str) and loc.strip() else None
        if text is None and isinstance(com, str):
            m = _LOC_RX.match(com)
            text = m.group(0) if m else None
        if text is None:
            out.append(None)
            continue
        t = re.sub(r"^\s*from\s+", "", text.lower()).strip()
        t = re.sub(r"^(?:figure|figurre|fig)\.?\s*", "fig", t)
        t = re.sub(r"^table\.?\s*", "table", t)
        t = re.sub(r"^graph\.?\s*", "graph", t)
        out.append(re.sub(r"\s+", "", t))
    return pd.Series(out, index=df.index, dtype=object)


_DIGITISER_RX = re.compile(r"extraction:((?:graphreader|webplot|plotdigitizer)[0-9A-Za-z]*)", re.IGNORECASE)


def digitiser_tag(df: pd.DataFrame) -> pd.Series:
    """``graphreader1`` / ``webplotm`` ... from ``comments_raw`` (``None`` otherwise)."""
    vals = []
    for c in _col(df, "comments_raw"):
        m = _DIGITISER_RX.search(c) if isinstance(c, str) else None
        vals.append(m.group(1).lower() if m else None)
    return pd.Series(vals, index=df.index, dtype=object)


def double_digitisation(df: pd.DataFrame, sig: int = 3, tol: float = DOUBLE_DIGITISATION_TOLERANCE_LOG_D,
                        pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same publication, same ``sig`` key, ``|delta log D| <= tol``, different record, not one A/B group.

    Candidates only: the same criterion also catches independent replicates and titration points whose
    varying condition is outside the base key.  ``location_relation`` (same / different figure-table label,
    or unknown), ``digitiser_a/b``, ``strict_key_equal`` and ``both_canonical`` are attached for triage.
    """
    p = near_duplicate_pairs(df, sig=sig) if pairs is None else pairs[pairs["sig_figs"] == sig]
    keep = (p["same_publication"].astype(bool) & p["abs_delta_log_D"].le(tol + 1e-12)
            & ~p["both_ab_group"].astype(bool) & (p["id_a"] != p["id_b"]))
    out = p[keep].reset_index(drop=True)
    loc = location_token(df)
    tag = digitiser_tag(df)
    canon = (df["is_canonical_row"].astype(bool) if "is_canonical_row" in df.columns
             else pd.Series(True, index=df.index))
    ia, ib = out["idx_a"].to_numpy(), out["idx_b"].to_numpy()
    la = loc.loc[ia].to_numpy(dtype=object)
    lb = loc.loc[ib].to_numpy(dtype=object)
    out["location_a"], out["location_b"] = la, lb
    out["location_relation"] = np.array(
        ["LOCATION_UNKNOWN" if (a is None or b is None) else ("SAME_LOCATION" if a == b else "DIFFERENT_LOCATION")
         for a, b in zip(la, lb)], dtype=object)
    out["digitiser_a"] = tag.loc[ia].to_numpy(dtype=object)
    out["digitiser_b"] = tag.loc[ib].to_numpy(dtype=object)
    out["both_canonical"] = canon.loc[ia].to_numpy(dtype=bool) & canon.loc[ib].to_numpy(dtype=bool)
    return out


# --------------------------------------------------------------------------------------------- #
# 5. DOI / source multiplicity and publication groups
# --------------------------------------------------------------------------------------------- #

_DOI_PREFIX_RX = re.compile(r"^(?:https?://)?(?:www\.|dx\.)?doi\.org/|^doi[:\s]\s*", re.IGNORECASE)
_DOI_BODY_RX = re.compile(r"10\.\d{4,9}/[^\s,;]+")


def _norm_doi(token: Any) -> str:
    return _DOI_PREFIX_RX.sub("", str(token).strip().lower()).strip().rstrip("/.,;)")


def row_source_dois(df: pd.DataFrame) -> list[tuple[str, ...]]:
    """Per row: the archive-corrected primary DOI plus every other ``doi_all`` token, normalised, with
    the SAFE self-citation and the uncorrected spelling of the primary removed."""
    out: list[tuple[str, ...]] = []
    for doi_all, raw, corr in zip(_col(df, "doi_all"), _col(df, "doi_primary"), _col(df, "doi_primary_corrected")):
        toks: set[str] = set()
        has_corr = isinstance(corr, str) and bool(corr.strip())
        raw_n = _norm_doi(raw) if isinstance(raw, str) else None
        if has_corr:
            toks.add(_norm_doi(corr))
        for t in _as_list(doi_all):
            n = _norm_doi(t)
            if not n or SAFE_SELF_CITATION in n or (has_corr and raw_n is not None and n == raw_n):
                continue
            m = _DOI_BODY_RX.search(n)
            if m:
                toks.add(m.group(0).rstrip("/.,;)"))
        out.append(tuple(sorted(toks)))
    return out


def corrected_publication_id(df: pd.DataFrame) -> pd.Series:
    """``g19_publication_id`` recomputed with the archive's DOI correction applied first.

    Every ``doi_all`` element equal to ``doi_primary`` is replaced by ``doi_primary_corrected`` and the
    loader's own rule (:func:`gen19ct.data.load.publication_key`) is re-run, so the only difference from
    ``g19_publication_id`` is the repaired DOI spelling.
    """
    from gen19ct.data.load import publication_key  # local import keeps this module importable standalone

    out = []
    for doi_all, raw, corr, refs, sub in zip(_col(df, "doi_all"), _col(df, "doi_primary"),
                                             _col(df, "doi_primary_corrected"), _col(df, "reference_other"),
                                             _col(df, "sub_source_file")):
        toks = [corr if (isinstance(corr, str) and corr.strip() and isinstance(raw, str) and str(t) == raw) else t
                for t in _as_list(doi_all)]
        out.append(publication_key(np.array(toks, dtype=object), refs, sub)[0])
    return pd.Series(out, index=df.index, dtype=object, name="corrected_publication_id")


def _norm_title(t: Any) -> str | None:
    if not isinstance(t, str) or not t.strip():
        return None
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"<[^>]+>", "", t.lower())).strip() or None


def doi_roles(df: pd.DataFrame, compilation_min_co_cited: int = 2) -> dict[str, str]:
    """``doi -> PRIMARY_SOURCE | COMPILATION_OR_SECONDARY`` (INFERRED: a DOI co-cited, across all rows,
    with >= ``compilation_min_co_cited`` other DOIs is treated as a compilation/review)."""
    co: dict[str, set[str]] = {}
    for toks in row_source_dois(df):
        for t in toks:
            co.setdefault(t, set()).update(x for x in toks if x != t)
    return {t: ("COMPILATION_OR_SECONDARY" if len(c) >= compilation_min_co_cited else "PRIMARY_SOURCE")
            for t, c in co.items()}


def doi_source_multiplicity(df: pd.DataFrame) -> pd.DataFrame:
    """One row per source DOI, per non-DOI report reference, and per reference title.

    ``risk``:
    ``V1_SPLIT_SAME_SOURCE`` -- a DOI that is a primary source (see :func:`doi_roles`) sits in more than one
    ``g19_publication_id``: the same paper's rows can land on both sides of a V1 fold.
    ``COMPILATION_FANOUT`` -- a compilation DOI spread over several publication ids; safe only if its rows
    are not copies of the primary papers' rows (see cross-publication copies).
    ``SAME_TITLE_MULTIPLE_DOIS`` (corrected DOIs differ), ``SAME_TITLE_MULTIPLE_RAW_DOIS`` (only the raw
    spellings differ -- the archive's DOI repair), ``SAME_TITLE_MULTIPLE_PUBLICATION_IDS`` -- title analogues.
    ``MULTI_CONTEXT`` -- one publication id but several sub-source files / entry authors / DOI sets.
    ``NONE`` otherwise.
    """
    dois = row_source_dois(df)
    roles = doi_roles(df)
    pubs = _col(df, PUB_COL)
    studies = _col(df, "g19_study_id")
    subs = _col(df, "sub_source_file")
    authors = _col(df, "entry_author")
    cites = _col(df, "archive_citation_doi")
    raws = _col(df, "doi_primary")
    corrs = _col(df, "doi_primary_corrected")
    titles = _col(df, "reference_title")
    tiers = _col(df, "g19_tier")
    rows = []
    for r, toks in enumerate(dois):
        corr_n = _norm_doi(corrs[r]) if isinstance(corrs[r], str) else None
        raw_n = _norm_doi(raws[r]) if isinstance(raws[r], str) else None
        for t in toks:
            spelling = raw_n if (t == corr_n and raw_n is not None) else t
            rows.append((t, r, len(toks) - 1, spelling))
    long = pd.DataFrame(rows, columns=["token", "row", "n_other_dois", "spelling"])
    out_rows: list[dict[str, Any]] = []

    def _ctx(rs: np.ndarray) -> dict[str, Any]:
        pub_set = sorted({str(pubs[x]) for x in rs})
        return {
            "n_rows": int(len(rs)),
            "n_model_rows": int(sum(1 for x in rs if tiers[x] == "MODEL")),
            "n_g19_publication_ids": int(len(pub_set)),
            "g19_publication_ids": ";".join(pub_set[:40]) + (";..." if len(pub_set) > 40 else ""),
            "n_g19_study_ids": int(len({studies[x] for x in rs})),
            "n_sub_source_files": int(len({subs[x] for x in rs if isinstance(subs[x], str)})),
            "n_entry_authors": int(len({authors[x] for x in rs if isinstance(authors[x], str)})),
            "entry_authors": ";".join(sorted({str(authors[x]) for x in rs if isinstance(authors[x], str)})),
            "n_archive_citation_contexts": int(len({str(cites[x]) if isinstance(cites[x], str) else "none"
                                                    for x in rs})),
            "n_reference_titles": int(len({_norm_title(titles[x]) for x in rs if _norm_title(titles[x])})),
        }

    if len(long):
        for tok, grp in long.groupby("token", sort=True):
            rs = grp["row"].to_numpy()
            co = sorted({c for x in rs for c in dois[x] if c != tok})
            ctx = _ctx(rs)
            role = roles.get(tok, "PRIMARY_SOURCE")
            n_sets = len({dois[x] for x in rs})
            if ctx["n_g19_publication_ids"] > 1:
                risk = "V1_SPLIT_SAME_SOURCE" if role == "PRIMARY_SOURCE" else "COMPILATION_FANOUT"
            elif ctx["n_sub_source_files"] > 1 or ctx["n_entry_authors"] > 1 or n_sets > 1:
                risk = "MULTI_CONTEXT"
            else:
                risk = "NONE"
            out_rows.append({
                "kind": "doi", "source": tok, **ctx, "role": role,
                "n_doi_sets": int(n_sets), "n_co_cited_dois": int(len(co)),
                "n_rows_cited_alone": int((grp["n_other_dois"] == 0).sum()),
                "raw_spellings": ";".join(_short(s, 80) for s in sorted(set(grp["spelling"]))),
                "risk": risk,
            })
    ref_rows = []
    for r, refs in enumerate(_col(df, "reference_other")):
        for t in _as_list(refs):
            if isinstance(t, str) and t.strip():
                ref_rows.append((t.strip().lower(), r))
    if ref_rows:
        rl = pd.DataFrame(ref_rows, columns=["token", "row"])
        for tok, grp in rl.groupby("token", sort=True):
            ctx = _ctx(grp["row"].to_numpy())
            risk = ("V1_SPLIT_SAME_SOURCE" if ctx["n_g19_publication_ids"] > 1 else
                    "MULTI_CONTEXT" if ctx["n_sub_source_files"] > 1 or ctx["n_entry_authors"] > 1 else "NONE")
            out_rows.append({"kind": "report_reference", "source": tok, **ctx, "role": "REPORT", "risk": risk})
    trows = [(t, r) for r, t in enumerate(_norm_title(x) for x in titles) if t]
    if trows:
        tl = pd.DataFrame(trows, columns=["token", "row"])
        for tok, grp in tl.groupby("token", sort=True):
            rs = grp["row"].to_numpy()
            ctx = _ctx(rs)
            primary = sorted({_norm_doi(corrs[x]) for x in rs if isinstance(corrs[x], str)})
            raw_primary = sorted({_norm_doi(raws[x]) for x in rs if isinstance(raws[x], str)})
            if len(primary) > 1:
                risk = "SAME_TITLE_MULTIPLE_DOIS"
            elif len(raw_primary) > 1:
                risk = "SAME_TITLE_MULTIPLE_RAW_DOIS"
            elif ctx["n_g19_publication_ids"] > 1:
                risk = "SAME_TITLE_MULTIPLE_PUBLICATION_IDS"
            else:
                risk = "NONE"
            out_rows.append({"kind": "reference_title", "source": tok[:200], **ctx, "role": "TITLE",
                             "n_primary_dois": int(len(primary)), "primary_dois": ";".join(primary),
                             "n_raw_primary_dois": int(len(raw_primary)),
                             "raw_spellings": ";".join(_short(s, 80) for s in raw_primary),
                             "n_doi_sets": int(len({dois[x] for x in rs})), "risk": risk})
    out = pd.DataFrame(out_rows)
    if len(out):
        order = {"V1_SPLIT_SAME_SOURCE": 0, "SAME_TITLE_MULTIPLE_DOIS": 1, "SAME_TITLE_MULTIPLE_RAW_DOIS": 2,
                 "SAME_TITLE_MULTIPLE_PUBLICATION_IDS": 3, "COMPILATION_FANOUT": 4, "MULTI_CONTEXT": 5, "NONE": 6}
        out["_o"] = out["risk"].map(order)
        out = out.sort_values(["_o", "kind", "n_rows", "source"], ascending=[True, True, False, True],
                              kind="stable").drop(columns="_o").reset_index(drop=True)
    return out


#: Link rules of :func:`publication_link_components`, from least to most conservative.
LINK_RULES: tuple[str, ...] = ("corrected_doi", "primary_source_doi", "archive_duplicate_group",
                               "cross_publication_copy", "near_duplicate_key", "compilation_doi")


def publication_link_components(df: pd.DataFrame, copy_pairs: pd.DataFrame | None = None,
                                key_pairs: pd.DataFrame | None = None,
                                duplicate_group_mask: Iterable[bool] | pd.Series | None = None) -> pd.DataFrame:
    """Per ``g19_publication_id``: the merged fold group under cumulatively added link rules.

    ``group_corrected_doi``          ids that differ only by the archive's DOI repair are merged;
    ``group_primary_source_doi``     + ids sharing a PRIMARY_SOURCE DOI (:func:`doi_roles`);
    ``group_archive_duplicate_group`` + ids holding rows of one archive ``duplicate_group_id`` (any class: an
                                     E_VALUE_CONFLICT group spanning publications is one measurement
                                     identity, and :func:`fold_isolation_check` forbids splitting it).  Only
                                     rows where ``duplicate_group_mask`` is True link (default: every row);
    ``group_cross_publication_copy`` + ids joined by a ``copy_pairs`` row (``pub_a``/``pub_b``);
    ``group_near_duplicate_key``     + ids joined by a ``key_pairs`` row (a near-duplicate key, any value);
    ``group_compilation_doi``        + ids sharing ANY source DOI, compilations included.

    A group id is the smallest member id (deterministic).  ``copy_pairs`` / ``key_pairs`` are taken as
    given, so the caller decides the population (e.g. MODEL rows only) and the tolerance.
    """
    pubs = df[PUB_COL].astype(object)
    nodes = sorted(pubs.unique(), key=str)
    corr = corrected_publication_id(df)
    dois = row_source_dois(df)
    roles = doi_roles(df)
    edges: dict[str, tuple[list[Any], list[Any]]] = {r: ([], []) for r in LINK_RULES}

    def link_by(labels: Iterable[Any], rule: str) -> None:
        first: dict[Any, Any] = {}
        for p, lab in zip(pubs, labels):
            if _is_missing(lab):
                continue
            if lab in first and first[lab] != p:
                edges[rule][0].append(first[lab])
                edges[rule][1].append(p)
            else:
                first.setdefault(lab, p)

    link_by(corr, "corrected_doi")
    if "duplicate_group_id" in df.columns:
        dmask = np.ones(len(df), dtype=bool) if duplicate_group_mask is None else \
            np.asarray(pd.Series(list(duplicate_group_mask)).fillna(False), dtype=bool)
        if len(dmask) != len(df):
            raise ValueError("duplicate_group_mask must have one entry per row")
        link_by([g if m else None for g, m in zip(df["duplicate_group_id"].tolist(), dmask)], "archive_duplicate_group")
    prim_a, prim_b = [], []
    comp_a, comp_b = [], []
    seen_prim: dict[str, Any] = {}
    seen_any: dict[str, Any] = {}
    for p, toks in zip(pubs, dois):
        for t in toks:
            if roles.get(t) == "PRIMARY_SOURCE":
                if t in seen_prim and seen_prim[t] != p:
                    prim_a.append(seen_prim[t])
                    prim_b.append(p)
                seen_prim.setdefault(t, p)
            if t in seen_any and seen_any[t] != p:
                comp_a.append(seen_any[t])
                comp_b.append(p)
            seen_any.setdefault(t, p)
    edges["primary_source_doi"] = (prim_a, prim_b)
    edges["compilation_doi"] = (comp_a, comp_b)
    for rule, pairs in (("cross_publication_copy", copy_pairs), ("near_duplicate_key", key_pairs)):
        if pairs is not None and len(pairs):
            cross = pairs[pairs["pub_a"] != pairs["pub_b"]]
            edges[rule] = (cross["pub_a"].tolist(), cross["pub_b"].tolist())

    frame = pd.DataFrame({PUB_COL: nodes})
    stats = df.groupby(PUB_COL, sort=False).agg(
        n_rows=(PUB_COL, "size"),
        n_model_rows=("g19_tier", lambda s: int((s == "MODEL").sum())) if "g19_tier" in df.columns else (PUB_COL, "size"),
        g19_publication_status=("g19_publication_status", "first") if "g19_publication_status" in df.columns
        else (PUB_COL, "first"),
    )
    frame = frame.join(stats, on=PUB_COL)
    refs = df.groupby(PUB_COL)["g19_publication_refs"].first() if "g19_publication_refs" in df.columns else None
    if refs is not None:
        frame["g19_publication_refs"] = frame[PUB_COL].map(refs).map(lambda s: _short(s, 160))
    frame["corrected_publication_id"] = frame[PUB_COL].map(pd.Series(corr.to_numpy(), index=pubs.to_numpy())
                                                            .groupby(level=0).first())
    acc_a: list[Any] = []
    acc_b: list[Any] = []
    for rule in LINK_RULES:
        acc_a.extend(edges[rule][0])
        acc_b.extend(edges[rule][1])
        comp = union_components(acc_a, acc_b, nodes)
        col = f"group_{rule}"
        frame[col] = frame[PUB_COL].map(comp)
        frame[f"n_publications_in_{col}"] = frame.groupby(col)[PUB_COL].transform("size")
        frame[f"n_model_rows_in_{col}"] = frame.groupby(col)["n_model_rows"].transform("sum")
        frame[f"n_edges_{rule}"] = len(edges[rule][0])
    return frame.sort_values(PUB_COL, kind="stable").reset_index(drop=True)


#: kinds of :func:`wildcard_copy_pairs`
WILDCARD_COPY_KINDS: tuple[str, ...] = ("STATE_WILDCARD", "STRUCTURE_WILDCARD")


def _is_decade(values: Any) -> np.ndarray:
    v = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (np.abs(v - np.round(v)) <= 1e-9)


def wildcard_copy_pairs(df: pd.DataFrame, sig: int = 6, tol: float = COPY_TOLERANCE_LOG_D) -> pd.DataFrame:
    """Value-matched row pairs that the registered near-duplicate key (and so ``fold_isolation_check`` and the copy
    grouping) cannot see, because the two records differ in a field the key contains (verification finding VL-04).

    ``STATE_WILDCARD``      same :func:`near_duplicate_key` with the oxidation state wildcarded
                            (``include_metal_state=False``, the :func:`metal_alias_split_risk` key), DIFFERENT state
                            labels (``Zr(IV)`` vs ``Zr(?)``, ``Pu(IV)`` vs ``Pu(VI)``), ``|delta log D| <= tol``;
    ``STRUCTURE_WILDCARD``  same key with ``extractant_system_key`` dropped (``include_structure=False``), the same
                            state label, DIFFERENT systems, ``|delta log D| <= tol``.

    Columns: ``kind``, ``idx_a``/``idx_b`` (index labels), ``id_a``/``id_b``, ``pub_a``/``pub_b``, ``same_publication``,
    ``state_a``/``state_b``, ``system_a``/``system_b``, ``log_D_a``/``log_D_b``, ``abs_delta_log_D``,
    ``same_duplicate_group``, ``D_raw_identical`` (the recorded ``D_raw`` strings are equal after stripping),
    ``decade_value`` (``log_D`` is an exact decade), ``same_location`` (same :func:`location_token`, both present), and
    ``strict_copy`` = ``STATE_WILDCARD``, or ``STRUCTURE_WILDCARD`` with ``D_raw_identical`` and not ``decade_value``
    (the bit-identical non-decade value that two independent extractants would not share by chance).  Whether a
    ``STRUCTURE_WILDCARD`` pair that is not strict is a copy is INFERRED, never asserted."""
    state = _state_label(df).to_numpy(dtype=object)
    system = _col(df, "extractant_system_key")
    y = np.asarray(pd.to_numeric(df["log_D"], errors="coerce"), dtype=float)
    ids = _col(df, ID_COL)
    pubs = _col(df, PUB_COL)
    dup = _col(df, "duplicate_group_id")
    draw = np.array([None if _is_missing(v) else str(v).strip() for v in _col(df, "D_raw")], dtype=object)
    loc = location_token(df).to_numpy(dtype=object) if ("data_location" in df.columns or "comments_raw" in df.columns)         else np.full(len(df), None, dtype=object)
    decade = _is_decade(y)
    frames = []
    for kind, keys in (("STATE_WILDCARD", near_duplicate_key(df, sig=sig, include_metal_state=False)),
                       ("STRUCTURE_WILDCARD", near_duplicate_key(df, sig=sig, include_structure=False))):
        i, j = _pairs_within_groups(keys)
        if not len(i):
            continue
        with np.errstate(invalid="ignore"):
            delta = np.abs(y[i] - y[j])
        keep = np.isfinite(delta) & (delta <= tol + 1e-12)
        if kind == "STATE_WILDCARD":
            keep &= state[i] != state[j]
        else:
            keep &= (state[i] == state[j]) & (system[i] != system[j])
        i, j, delta = i[keep], j[keep], delta[keep]
        same_raw = np.array([a is not None and a == b for a, b in zip(draw[i], draw[j])], dtype=bool)
        frames.append(pd.DataFrame({
            "kind": kind, "idx_a": df.index.to_numpy()[i], "idx_b": df.index.to_numpy()[j], "id_a": ids[i], "id_b": ids[j],
            "pub_a": pubs[i], "pub_b": pubs[j], "same_publication": pubs[i] == pubs[j], "state_a": state[i],
            "state_b": state[j], "system_a": system[i], "system_b": system[j], "log_D_a": y[i], "log_D_b": y[j],
            "abs_delta_log_D": delta,
            "same_duplicate_group": np.array([not _is_missing(a) and a == b for a, b in zip(dup[i], dup[j])], dtype=bool),
            "D_raw_identical": same_raw, "decade_value": decade[i] & decade[j],
            "same_location": np.array([a is not None and a == b for a, b in zip(loc[i], loc[j])], dtype=bool)}))
    cols = ["kind", "idx_a", "idx_b", "id_a", "id_b", "pub_a", "pub_b", "same_publication", "state_a", "state_b",
            "system_a", "system_b", "log_D_a", "log_D_b", "abs_delta_log_D", "same_duplicate_group", "D_raw_identical",
            "decade_value", "same_location", "strict_copy"]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["strict_copy"] = (out["kind"] == "STATE_WILDCARD") | (out["D_raw_identical"] & ~out["decade_value"])
    return out[cols]


# --------------------------------------------------------------------------------------------- #
# 6. metal aliases
# --------------------------------------------------------------------------------------------- #

def _state_label(df: pd.DataFrame) -> pd.Series:
    return pd.Series([s if isinstance(s, str) else (f"{e}(?)" if isinstance(e, str) else None)
                      for s, e in zip(_col(df, "g19_metal_state"), _col(df, "g19_metal"))],
                     index=df.index, dtype=object)


def metal_alias_split_risk(df: pd.DataFrame, sig: int = 6) -> pd.DataFrame:
    """Element vs element+state labels.

    ``scope="element"``: per element, every state label (``X(?)`` = unknown state) with row counts, the raw
    metal tokens, and how many publications record the element under more than one label.
    ``scope="publication_element"``: per (publication, element) with more than one state label -- rows and
    MODEL rows per label, and ``n_condition_keys_across_labels`` = condition keys (:func:`near_duplicate_key`
    with the oxidation state wildcarded) that occur under two labels, i.e. the same experiment recorded
    under different states.  ``risk`` is ``UNKNOWN_AND_KNOWN_STATE`` (a state-level leave-metal-out fold
    would keep the unknown-state rows of the hidden metal in train), ``MULTIPLE_KNOWN_STATES`` (different
    species; a problem only for an element-level fold), with suffix ``+SAME_CONDITIONS`` when a condition
    key is shared across labels.
    ``scope="export_file"``: per element, rows whose ``source_file`` stem (the per-metal export file) is a
    different element -- the export fan-out; a fold keyed on the file name would mislabel them.
    """
    d = df[df["g19_metal"].notna()]
    lab = _state_label(d)
    tier = d["g19_tier"] if "g19_tier" in d.columns else pd.Series(None, index=d.index, dtype=object)
    is_model = tier.eq("MODEL")
    raw = d["metal_raw"] if "metal_raw" in d.columns else pd.Series(None, index=d.index, dtype=object)
    pub = d[PUB_COL]
    rows: list[dict[str, Any]] = []
    for el, idx in d.groupby("g19_metal", sort=True).groups.items():
        labs = lab.loc[idx]
        vc = labs.value_counts()
        n_pub_multi = int(pd.DataFrame({"p": pub.loc[idx], "l": labs}).groupby("p")["l"].nunique().gt(1).sum())
        unknown = labs.str.endswith("(?)")
        known_labels = labs[~unknown].nunique()
        rows.append({
            "scope": "element", "g19_metal": el, "g19_publication_id": None,
            "state_labels": ";".join(f"{k}:{int(v)}" for k, v in sorted(vc.items())),
            "model_state_labels": ";".join(f"{k}:{int(v)}" for k, v in
                                           sorted(labs[is_model.loc[idx]].value_counts().items())),
            "n_state_labels": int(len(vc)), "n_rows": int(len(idx)), "n_model_rows": int(is_model.loc[idx].sum()),
            "n_rows_unknown_state": int(unknown.sum()),
            "n_model_rows_unknown_state": int((is_model.loc[idx] & unknown).sum()),
            "metal_raw_tokens": ";".join(sorted({str(x) for x in raw.loc[idx].dropna()})),
            "n_publications": int(pub.loc[idx].nunique()),
            "n_publications_multi_label": n_pub_multi,
            "risk": ("UNKNOWN_AND_KNOWN_STATE" if unknown.any() and (~unknown).any() else
                     "MULTIPLE_KNOWN_STATES" if known_labels > 1 else "NONE"),
        })
    wild = near_duplicate_key(d, sig=sig, include_metal_state=False)
    frame = pd.DataFrame({"p": pub, "e": d["g19_metal"], "l": lab, "k": wild, "m": is_model})
    for (p, el), grp in frame.groupby(["p", "e"], sort=True):
        if grp["l"].nunique() < 2:
            continue
        per_key = grp.groupby("k")["l"].nunique()
        shared_keys = per_key[per_key > 1].index
        mgrp = grp[grp["m"]]
        mper = mgrp.groupby("k")["l"].nunique()
        mshared = mper[mper > 1].index
        unknown = grp["l"].str.endswith("(?)")
        base = "UNKNOWN_AND_KNOWN_STATE" if unknown.any() and (~unknown).any() else "MULTIPLE_KNOWN_STATES"
        vc = grp["l"].value_counts()
        mvc = mgrp["l"].value_counts()
        rows.append({
            "scope": "publication_element", "g19_metal": el, "g19_publication_id": p,
            "state_labels": ";".join(f"{k}:{int(v)}" for k, v in sorted(vc.items())),
            "model_state_labels": ";".join(f"{k}:{int(v)}" for k, v in sorted(mvc.items())),
            "n_state_labels": int(len(vc)), "n_rows": int(len(grp)), "n_model_rows": int(grp["m"].sum()),
            "n_rows_unknown_state": int(unknown.sum()),
            "n_model_rows_unknown_state": int((grp["m"] & unknown).sum()),
            "n_publications": 1, "n_publications_multi_label": 1,
            "n_condition_keys_across_labels": int(len(shared_keys)),
            "n_rows_in_shared_condition_keys": int(grp["k"].isin(shared_keys).sum()),
            "n_model_condition_keys_across_labels": int(len(mshared)),
            "n_model_rows_in_shared_condition_keys": int(mgrp["k"].isin(mshared).sum()),
            "risk": base + ("+SAME_CONDITIONS" if len(shared_keys) else ""),
        })
    if "source_file" in d.columns:
        stem = d["source_file"].astype("string").str.replace(r"\.csv$", "", regex=True)
        other = stem.ne(d["g19_metal"].astype("string")).fillna(True)
        for el, idx in d.groupby("g19_metal", sort=True).groups.items():
            o = other.loc[idx]
            rows.append({
                "scope": "export_file", "g19_metal": el, "g19_publication_id": None,
                "state_labels": ";".join(f"{k}:{int(v)}" for k, v in sorted(stem.loc[idx][o].value_counts().items())),
                "n_rows": int(len(idx)), "n_model_rows": int(is_model.loc[idx].sum()),
                "n_rows_file_is_other_metal": int(o.sum()),
                "n_model_rows_file_is_other_metal": int((o & is_model.loc[idx]).sum()),
                "risk": "EXPORT_FILE_IS_NOT_METAL" if o.any() else "NONE",
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- #
# 7. metadata -> target leakage
# --------------------------------------------------------------------------------------------- #

_NUM_RX = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w])")


def _tokens(text: Any) -> list[tuple[float, int, int]]:
    """``(value, significant_digits, decimals)`` for each standalone number in ``text``."""
    if not isinstance(text, str):
        return []
    out = []
    for m in _NUM_RX.finditer(text):
        s = m.group(0)
        try:
            v = float(s)
        except ValueError:
            continue
        mant = re.sub(r"[eE].*$", "", s).lstrip("+-")
        digits = re.sub(r"[^0-9]", "", mant).lstrip("0")
        dec = len(mant.split(".", 1)[1]) if "." in mant else 0
        out.append((v, len(digits), dec))
    return out


def _match_flags(tokens: list[tuple[float, int, int]], d: float, logd: float) -> tuple[bool, bool, bool]:
    exact = sf3 = log2 = False
    if not np.isfinite(d):
        return exact, sf3, log2
    d3 = round_sig([d], 3)[0]
    for v, nd, dec in tokens:
        if not exact and nd >= 3 and abs(v - d) <= 1e-6 * max(abs(d), 1e-300):
            exact = True
        if not sf3 and nd >= 3 and round_sig([v], 3)[0] == d3:
            sf3 = True
        if not log2 and dec >= 2 and np.isfinite(logd) and abs(v - logd) <= 0.005:
            log2 = True
    return exact, sf3, log2


def target_in_text(df: pd.DataFrame, columns: tuple[str, ...] = ("comments_raw", "data_location", "ini_comp_raw",
                                                                 "sub_source_file"),
                   seed: int = 0, n_examples: int = 3) -> pd.DataFrame:
    """Does the target appear inside provenance text?  Per text column and match type:

    ``D_value_exact`` (a token equal to D within 1e-6 relative, token with >= 3 significant digits),
    ``D_value_3sf`` (equal at 3 s.f., token >= 3 significant digits), ``log_D_2dp`` (a token with >= 2
    decimals within 0.005 of log D), ``D_raw_substring`` (the raw D string, >= 4 characters, occurs
    verbatim).  ``n_rows_permuted`` repeats the test with D / log D / D_raw permuted across rows
    (``seed``) -- the coincidence rate for the same texts.
    """
    logd = pd.to_numeric(df["log_D"], errors="coerce").to_numpy(dtype=float)
    dval = pd.to_numeric(df["D_value"], errors="coerce").to_numpy(dtype=float)
    draw = _col(df, "D_raw")
    tier = _col(df, "g19_tier")
    valid = np.flatnonzero(np.isfinite(logd) & np.isfinite(dval))
    rng = np.random.default_rng(seed)
    perm = valid[rng.permutation(len(valid))]
    kinds = ("D_value_exact", "D_value_3sf", "log_D_2dp", "D_raw_substring")
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        texts = df[col].to_numpy(dtype=object)
        res = {k: np.zeros(len(valid), dtype=bool) for k in kinds}
        resp = {k: np.zeros(len(valid), dtype=bool) for k in kinds}
        for n, r in enumerate(valid):
            t = texts[r]
            if not isinstance(t, str):
                continue
            toks = _tokens(t)
            q = perm[n]
            res["D_value_exact"][n], res["D_value_3sf"][n], res["log_D_2dp"][n] = _match_flags(toks, dval[r], logd[r])
            resp["D_value_exact"][n], resp["D_value_3sf"][n], resp["log_D_2dp"][n] = _match_flags(toks, dval[q],
                                                                                                   logd[q])
            for store, target in ((res, draw[r]), (resp, draw[q])):
                if isinstance(target, str) and len(target.strip()) >= 4 and target.strip() in t:
                    store["D_raw_substring"][n] = True
        for k in kinds:
            hit = res[k]
            ex_rows = valid[hit][:n_examples]
            rows.append({
                "check": "target_in_text", "grouping": col, "match_type": k,
                "n_rows": int(len(valid)), "n_rows_matched": int(hit.sum()),
                "n_model_rows_matched": int(sum(1 for r in valid[hit] if tier[r] == "MODEL")),
                "n_rows_matched_permuted": int(resp[k].sum()),
                "examples": " || ".join(f"D={dval[r]:.6g} logD={logd[r]:.4g} text={_short(texts[r])}" for r in ex_rows),
            })
    return pd.DataFrame(rows)


def loo_group_mean_r2(y: np.ndarray, labels: pd.Series) -> dict[str, float]:
    """Leave-one-out group-mean statistic of ``y`` (singletons fall back to the LOO global mean).

    Descriptive, not a fitted model: how much of ``y`` a grouping would "explain" if its label were used
    as a feature, with the row itself excluded from its own group mean.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return {"n_rows": 0, "n_groups": 0, "n_singleton_rows": 0, "eta2_in_sample": float("nan"),
                "loo_r2": float("nan"), "loo_mae": float("nan"), "loo_mae_global_mean": float("nan")}
    codes, _ = pd.factorize(pd.Series(labels).astype("string").fillna("<NA>"))
    sums = np.bincount(codes, weights=y)
    cnts = np.bincount(codes)
    tot = y.sum()
    c = cnts[codes]
    glob = (tot - y) / max(n - 1, 1)
    pred = np.where(c > 1, (sums[codes] - y) / np.maximum(c - 1, 1), glob)
    sst = float(((y - y.mean()) ** 2).sum())
    fitted = sums[codes] / cnts[codes]
    return {
        "n_rows": int(n), "n_groups": int(len(cnts)), "n_singleton_rows": int((c == 1).sum()),
        "eta2_in_sample": 1 - float(((y - fitted) ** 2).sum()) / sst if sst > 0 else float("nan"),
        "loo_r2": 1 - float(((y - pred) ** 2).sum()) / sst if sst > 0 else float("nan"),
        "loo_mae": float(np.abs(y - pred).mean()),
        "loo_mae_global_mean": float(np.abs(y - glob).mean()),
    }


def provenance_groupings(df: pd.DataFrame, sig: int = 6) -> dict[str, tuple[str, pd.Series]]:
    """Named groupings ``name -> (role, labels)`` used by :func:`provenance_group_predictability`."""
    g: dict[str, tuple[str, pd.Series]] = {}
    exp = pd.to_numeric(df["source_record_id"], errors="coerce")
    line = pd.to_numeric(df["source_line_number"], errors="coerce")
    add = df["addition_date"].astype("string")
    pub = df[PUB_COL].astype("string")
    loc = location_token(df).astype("string")
    src = df["source_file"].astype("string")
    g["source_file"] = ("provenance", src)
    g["entry_author"] = ("provenance", df["entry_author"].astype("string"))
    g["addition_date_day"] = ("provenance", add.str.slice(0, 10))
    g["addition_date_minute"] = ("provenance", add.str.slice(0, 16))
    g["source_line_block20"] = ("provenance", src + ":" + (line // 20).astype("Int64").astype("string"))
    g["exp_id_block20"] = ("provenance", (exp // 20).astype("Int64").astype("string"))
    g["export_fanout_size"] = ("provenance", df["export_fanout_size"].astype("string"))
    g["reference_year"] = ("provenance", df["reference_year"].astype("string"))
    g["g19_publication_id"] = ("provenance", pub)
    g["g19_study_id"] = ("provenance", df["g19_study_id"].astype("string"))
    g["publication_x_location"] = ("provenance", pub + "@" + loc.fillna("NA"))
    g["publication_x_addition_minute"] = ("provenance", pub + "@" + add.str.slice(0, 16))
    g["publication_x_source_line_block20"] = ("provenance", pub + "@" + g["source_line_block20"][1])
    g["series_id"] = ("chemistry_reference", df["series_id"].astype("string"))
    cell = (_state_label(df).astype("string").fillna("NA") + "@" + df["extractant_system_key"].astype("string"))
    g["metal_state_x_extractant_system"] = ("chemistry_reference", cell)
    g[f"near_duplicate_key_sig{sig}"] = ("chemistry_reference", near_duplicate_key(df, sig=sig).astype("string"))
    g["publication_x_metal_state_x_extractant_system"] = ("chemistry_reference", pub + "@" + cell)
    return g


def provenance_group_predictability(df: pd.DataFrame, sig: int = 6, population: str = "all") -> pd.DataFrame:
    """LOO group-mean R^2 of ``log_D`` for provenance groupings vs chemistry reference groupings.

    ``excess_over_best_chemistry`` = a grouping's ``loo_r2`` minus the best chemistry-reference
    ``loo_r2``; a provenance grouping with a clearly positive excess would be "suspicious" (it explains the
    target beyond what the chemistry of the same rows explains).
    """
    y = pd.to_numeric(df["log_D"], errors="coerce")
    keep = y.notna().to_numpy()
    d = df[keep]
    yy = y[keep].to_numpy(dtype=float)
    rows = []
    for name, (role, labels) in provenance_groupings(d, sig=sig).items():
        rows.append({"check": "provenance_group_predictability", "population": population, "grouping": name,
                     "role": role, **loo_group_mean_r2(yy, labels)})
    out = pd.DataFrame(rows)
    best_chem = out.loc[out["role"] == "chemistry_reference", "loo_r2"].max()
    pub_r2 = out.loc[out["grouping"] == "g19_publication_id", "loo_r2"].max()
    out["loo_r2_minus_publication"] = out["loo_r2"] - pub_r2
    out["excess_over_best_chemistry"] = out["loo_r2"] - best_chem
    return out


def serial_target_correlation(df: pd.DataFrame, population: str = "all") -> pd.DataFrame:
    """Lag-1 structure of ``log_D`` along the archive record id (``source_record_id`` order).

    Consecutive records of the same publication are expected to correlate (series).  Consecutive records of
    *different* publications should not beyond what shared chemistry explains -- ``same_cell_fraction``
    gives that chemistry context.
    """
    exp = pd.to_numeric(df["source_record_id"], errors="coerce")
    y = pd.to_numeric(df["log_D"], errors="coerce")
    d = df.assign(_exp=exp, _y=y)
    d = d[d["_exp"].notna() & d["_y"].notna()].sort_values("_exp", kind="stable")
    a, b = d.iloc[:-1], d.iloc[1:]
    same_pub = a[PUB_COL].to_numpy() == b[PUB_COL].to_numpy()
    cell_a = (_state_label(a).astype("string").fillna("NA") + "@" + a["extractant_system_key"].astype("string")
              .fillna("NA")).to_numpy()
    cell_b = (_state_label(b).astype("string").fillna("NA") + "@" + b["extractant_system_key"].astype("string")
              .fillna("NA")).to_numpy()
    ya, yb = a["_y"].to_numpy(), b["_y"].to_numpy()
    rows = []
    for name, m in (("all_consecutive", np.ones(len(ya), dtype=bool)), ("same_publication", same_pub),
                    ("different_publication", ~same_pub)):
        k = int(m.sum())
        r = float(np.corrcoef(ya[m], yb[m])[0, 1]) if k > 2 else float("nan")
        rows.append({"check": "serial_target_correlation", "population": population, "grouping": name,
                     "role": "provenance", "n_rows": k, "lag1_pearson_r": r,
                     "mean_abs_delta_log_D": float(np.abs(ya[m] - yb[m]).mean()) if k else float("nan"),
                     "same_cell_fraction": float((cell_a[m] == cell_b[m]).mean()) if k else float("nan")})
    return pd.DataFrame(rows)


def metadata_target_leak(df: pd.DataFrame, seed: int = 0, sig: int = 6, population: str = "all") -> pd.DataFrame:
    """Concatenation of :func:`target_in_text`, :func:`provenance_group_predictability` and
    :func:`serial_target_correlation`, distinguished by the ``check`` column."""
    tit = target_in_text(df, seed=seed)
    tit.insert(1, "population", population)
    parts = [tit, provenance_group_predictability(df, sig=sig, population=population),
             serial_target_correlation(df, population=population)]
    return pd.concat(parts, ignore_index=True, sort=False)


# --------------------------------------------------------------------------------------------- #
# 8. V1 burden of near duplicates
# --------------------------------------------------------------------------------------------- #

def cross_boundary_burden(df: pd.DataFrame, pairs: pd.DataFrame, flag: str | None = None,
                          tol: float | None = None, key_col: str | None = None) -> dict[str, Any]:
    """How much would have to be grouped so that no selected near-duplicate pair crosses a V1 boundary.

    ``pairs`` is a pair table (``id_a``/``id_b``/``pub_a``/``pub_b``/``abs_delta_log_D``), restricted to
    rows where ``flag`` is true (when given) and ``abs_delta_log_D <= tol`` (when given).  Only pairs from
    different publications count.  Returns the pair / row / key / publication counts, the row-level answer
    (distinct rows in such pairs = rows needing a shared fold group with a row of another publication), and
    the publication-level answer (union-find components of publications; rows in merged components).
    """
    p = pairs
    if flag is not None:
        p = p[p[flag].astype(bool)]
    if tol is not None:
        p = p[p["abs_delta_log_D"].le(tol + 1e-12)]
    cross = p[p["pub_a"] != p["pub_b"]]
    rows_involved = set(cross["id_a"]).union(cross["id_b"])
    pubs_involved = set(cross["pub_a"]).union(cross["pub_b"])
    all_pubs = sorted(df[PUB_COL].astype(str).unique())
    comp = union_components(cross["pub_a"].astype(str), cross["pub_b"].astype(str), all_pubs)
    comp_s = pd.Series(comp)
    sizes = comp_s.value_counts()
    multi = sizes[sizes > 1]
    pub_rows = df[PUB_COL].astype(str).value_counts()
    comp_rows = pub_rows.groupby(comp_s.reindex(pub_rows.index)).sum()
    if key_col is None:
        key_col = "key_id" if "key_id" in cross.columns else None
    n_keys = int(cross[key_col].nunique()) if key_col is not None and len(cross) else 0
    return {
        "n_pairs_cross_publication": int(len(cross)),
        "n_rows_in_cross_publication_pairs": int(len(rows_involved)),
        "n_keys": n_keys,
        "n_publications_involved": int(len(pubs_involved)),
        "n_publications_total": int(len(all_pubs)),
        "n_merged_components": int(len(multi)),
        "n_publication_groups_after_merge": int(len(sizes)),
        "largest_component_n_publications": int(multi.max()) if len(multi) else 1,
        "largest_component_n_rows": int(comp_rows.max()) if len(comp_rows) else 0,
        "n_rows_in_merged_components": int(comp_rows[multi.index].sum()) if len(multi) else 0,
        "n_rows_total": int(len(df)),
    }


# --------------------------------------------------------------------------------------------- #
# 9. frozen bundle vs archive
# --------------------------------------------------------------------------------------------- #

def bundle_overlap(bundle: pd.DataFrame, archive: pd.DataFrame, gen6: pd.DataFrame | None = None,
                   rel_tol: float = 1e-6) -> pd.DataFrame:
    """One row per bundle row: does its ``safe_exp_id`` join an archive record, and does it agree?

    ``bundle`` needs ``safe_exp_id``, ``metal_symbol``, ``metal_ox``, ``log_D``, ``canonical_smiles`` and the
    ``cond__acid_concentration_M`` / ``cond__extractant_concentration_M`` / ``cond__temperature_C`` columns;
    ``gen6`` (optional) is :func:`gen19ct.data.load.bundle_publication_map`.
    """
    b = bundle.copy()
    b["exp_id"] = pd.to_numeric(b["safe_exp_id"].astype("string").str.split(":").str[-1], errors="coerce").astype("Int64")
    cols = ["g19_bundle_exp_id", ID_COL, "g19_metal", "g19_ox", "log_D", "extractant_primary_smiles",
            "extractant_smiles_canonical", "acid_concentration_M", "extractant_primary_concentration_M",
            "temperature_C", "g19_tier", "model_readiness", "duplicate_class", "duplicate_group_id",
            "is_canonical_row", PUB_COL, "metal_category"]
    a = archive[[c for c in cols if c in archive.columns]].rename(columns={"log_D": "archive_log_D"})
    j = b.merge(a, left_on="exp_id", right_on="g19_bundle_exp_id", how="left", validate="many_to_one")
    out = pd.DataFrame({
        "safe_exp_id": j["safe_exp_id"].astype(str), "exp_id": j["exp_id"],
        "joined": j[ID_COL].notna(), ID_COL: j[ID_COL],
        "bundle_metal_symbol": j["metal_symbol"], "archive_metal": j["g19_metal"],
        "bundle_metal_ox": j["metal_ox"], "archive_ox": j["g19_ox"],
        "bundle_log_D": j["log_D"], "archive_log_D": j["archive_log_D"],
    })
    out["metal_agree"] = (out["bundle_metal_symbol"].astype(object) == out["archive_metal"].astype(object))
    out["ox_agree"] = pd.to_numeric(out["bundle_metal_ox"], errors="coerce").eq(
        pd.to_numeric(out["archive_ox"], errors="coerce"))
    out["abs_delta_log_D"] = (pd.to_numeric(out["bundle_log_D"], errors="coerce")
                              - pd.to_numeric(out["archive_log_D"], errors="coerce")).abs()

    def _close(x: pd.Series, y: pd.Series) -> np.ndarray:
        xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        both_nan = ~np.isfinite(xv) & ~np.isfinite(yv)
        close = np.isfinite(xv) & np.isfinite(yv) & (np.abs(xv - yv) <= rel_tol * np.maximum(np.abs(yv), 1e-12))
        return close | both_nan

    out["acid_M_agree"] = _close(j["cond__acid_concentration_M"], j["acid_concentration_M"])
    out["extractant_M_agree"] = _close(j["cond__extractant_concentration_M"], j["extractant_primary_concentration_M"])
    out["temperature_agree"] = _close(j["cond__temperature_C"], j["temperature_C"])
    smi_b = j["canonical_smiles"].to_numpy(dtype=object)
    prim = j["extractant_primary_smiles"].to_numpy(dtype=object)
    allsm = j["extractant_smiles_canonical"].to_numpy(dtype=object)
    out["smiles_equals_archive_primary"] = [isinstance(s, str) and s == p for s, p in zip(smi_b, prim)]
    out["smiles_in_archive_system"] = [isinstance(s, str) and s in [str(x) for x in _as_list(al)]
                                       for s, al in zip(smi_b, allsm)]
    for c in ("g19_tier", "model_readiness", "duplicate_class", "duplicate_group_id", "is_canonical_row", PUB_COL):
        out[c] = j[c].to_numpy(dtype=object) if c in j.columns else None
    grp = out.loc[out["joined"], "duplicate_group_id"]
    out["n_bundle_rows_in_duplicate_group"] = out["duplicate_group_id"].map(grp.value_counts()).astype("Int64")
    if gen6 is not None:
        m = gen6.set_index("safe_exp_id")["publication_id"]
        out["gen6_publication_id"] = out["safe_exp_id"].map(m)
        out["publication_id_identical"] = out["gen6_publication_id"] == out[PUB_COL]
    return out


# --------------------------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------------------------- #

def pair_isolation_check(pairs: pd.DataFrame, folds: Mapping[Any, Any] | pd.Series,
                         member_cols: tuple[str, str] = ("idx_a", "idx_b"),
                         raise_on_violation: bool = True) -> pd.DataFrame:
    """A derived pair row must have both members in the same fold.

    ``folds`` maps a member id to its fold label (``"train"``/``"test"``, or a fold number).  Returns the
    violating pair rows with ``fold_a``, ``fold_b`` and ``reason`` (``CROSSES_FOLDS`` or ``UNASSIGNED``);
    raises ``AssertionError`` when there is any and ``raise_on_violation``.
    """
    if isinstance(folds, pd.Series):
        fmap = folds
    else:
        fmap = pd.Series(list(folds.values()), index=pd.Index(list(folds.keys()), dtype=object), dtype=object)
    if fmap.index.has_duplicates:
        raise ValueError("folds assigns a member more than once")
    a_col, b_col = member_cols
    fa = pairs[a_col].map(fmap)
    fb = pairs[b_col].map(fmap)
    unassigned = fa.isna() | fb.isna()
    crosses = ~unassigned & (fa.astype(object) != fb.astype(object))
    bad_mask = unassigned | crosses
    bad = pairs[bad_mask].copy()
    bad["fold_a"] = fa[bad_mask]
    bad["fold_b"] = fb[bad_mask]
    bad["reason"] = np.where(unassigned[bad_mask], "UNASSIGNED", "CROSSES_FOLDS")
    if raise_on_violation and len(bad):
        raise AssertionError(f"pair isolation violated: {int(crosses.sum())} pair(s) cross folds, "
                             f"{int(unassigned.sum())} pair(s) have an unassigned member; first: "
                             f"{bad.head(3)[[a_col, b_col, 'fold_a', 'fold_b', 'reason']].to_dict('records')}")
    return bad.reset_index(drop=True)


_LEVELS = ("V1", "V2", "V3", "V4", "V5", "V6")


def _levels(level: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(level, str):
        lv = _LEVELS if level.lower() == "all" else tuple(x.strip().upper() for x in level.split("+") if x.strip())
    else:
        lv = tuple(str(x).upper() for x in level)
    bad = [x for x in lv if x not in _LEVELS]
    if bad or not lv:
        raise ValueError(f"unknown level(s) {bad or level!r}; expected a subset of {_LEVELS} or 'all'")
    return lv


def _components(key: Any, component_map: Mapping[str, str] | None) -> set[str]:
    parts = str(key).split("|")
    return {component_map.get(p, p) for p in parts} if component_map else set(parts)


def fold_isolation_check(train_idx: Iterable[Any], test_idx: Iterable[Any], df: pd.DataFrame,
                         level: str | Iterable[str], *, near_dup_sig: int | None = 6,
                         near_dup_value_tol: float | None = None, element_level: bool = False,
                         component_aware: bool = False, component_map: Mapping[str, str] | None = None,
                         hidden_families: Iterable[str] | None = None,
                         family_map: Mapping[str, Any] | None = None,
                         raise_on_violation: bool = True, n_examples: int = 5) -> dict[str, Any]:
    """Assert that a train/test split isolates what ``level`` says it hides.

    Always: train and test share no index label, every label exists in ``df``, no archive
    ``duplicate_group_id`` (any class: A/B, C replicate, E value conflict, F insufficient information) has
    members on both sides, and (unless ``near_dup_sig is None``) no :func:`near_duplicate_key` value crosses
    the boundary.  With
    ``near_dup_value_tol`` only keys where some train row and some test row agree within that
    ``|delta log D|`` count (the "identical conditions AND values" reading of brief section 12).
    ``V1``: no ``g19_publication_id`` on both sides.
    ``V2``: no ``g19_metal_state`` on both sides, and no *alias*: an element present on both sides where
    either side holds rows of that element with an unknown state.  ``element_level=True`` forbids the element
    on both sides whatever the state.  Train rows of a test element with a different known state from the
    same publication are reported under ``warnings``.
    ``V5``: no (metal state, ``extractant_system_key``) cell on both sides, with the same unknown-state alias
    rule at the (element, extractant system) level; train rows of a hidden state whose extractant system
    shares a component structure with a hidden cell (e.g. TODGA|DHOA while TODGA is hidden) are reported
    under ``warnings`` (the V5-cell-only reading).  ``component_aware=True`` (the registered V5 hiding) makes
    that the violation ``V5_hidden_state_in_component_sharing_system``: in another system sharing a component
    with the hidden system, no train row of the hidden STATE and no unknown-state X(?) row of its element --
    the same state-level rule as in the hidden cell's own system.  Train rows of the element's OTHER known
    states there are allowed (a different species; element-level transfer is V2) and counted under
    ``warnings`` as ``V5_other_known_state_of_hidden_element_in_component_sharing_system_rows``.
    ``V6``: the Pr/Nd double cell -- the V5 checks with ``component_aware`` forced on.
    ``V3``: no test system on the train side (``V3_shared_system``) and no train system sharing a component
    with a test system (``V3_component_sharing_system``).
    ``V4``: needs ``hidden_families`` and ``family_map`` (component key -> family, or an iterable of
    families): no train system containing a component of a hidden family
    (``V4_train_system_contains_hidden_family``).
    ``component_map`` maps a component SMILES to the identity used for "shares a component" (default the
    SMILES itself; a stereo-free parent-key map gives the parent-structure sensitivity).  V7 (condition
    regions) has no level: the region is defined by the fold builder, which must assert it separately.

    Returns ``{"ok", "levels", "n_train", "n_test", "violations": {name: n}, "examples": {...},
    "warnings": {...}}``; raises ``AssertionError`` on any violation when ``raise_on_violation``.
    """
    levels = _levels(level)
    tr = pd.Index(list(train_idx))
    te = pd.Index(list(test_idx))
    viol: dict[str, int] = {}
    ex: dict[str, list[Any]] = {}
    warn: dict[str, int] = {}

    def record(name: str, items: Iterable[Any]) -> None:
        items = sorted({str(x) for x in items})
        viol[name] = len(items)
        ex[name] = items[:n_examples]

    if tr.has_duplicates or te.has_duplicates:
        warn["duplicate_labels_in_split"] = int(tr.duplicated().sum() + te.duplicated().sum())
    record("index_overlap", tr.intersection(te))
    record("index_not_in_frame", list(tr.difference(df.index)) + list(te.difference(df.index)))
    trd = df.loc[tr.intersection(df.index).unique()]
    ted = df.loc[te.intersection(df.index).unique()]

    def _set(frame: pd.DataFrame, col: str) -> set:
        return {v for v in frame[col].tolist() if isinstance(v, str)}

    if "duplicate_group_id" in df.columns:
        record("duplicate_group_shared", _set(trd, "duplicate_group_id") & _set(ted, "duplicate_group_id"))
    if "V1" in levels:
        record("V1_shared_publication", _set(trd, PUB_COL) & _set(ted, PUB_COL))
    if "V2" in levels:
        tr_unknown_el = {e for e, s in zip(trd["g19_metal"], trd["g19_metal_state"])
                         if isinstance(e, str) and not isinstance(s, str)}
        te_unknown_el = {e for e, s in zip(ted["g19_metal"], ted["g19_metal_state"])
                         if isinstance(e, str) and not isinstance(s, str)}
        both = _set(trd, "g19_metal") & _set(ted, "g19_metal")
        record("V2_shared_metal_state", _set(trd, "g19_metal_state") & _set(ted, "g19_metal_state"))
        record("V2_unknown_state_alias", {e for e in both if e in tr_unknown_el or e in te_unknown_el})
        if element_level:
            record("V2_shared_element", both)
        else:
            te_pe = set(zip(ted[PUB_COL], ted["g19_metal"]))
            te_states = _set(ted, "g19_metal_state")
            warn["V2_other_known_state_same_publication_rows"] = int(sum(
                1 for p, e, s in zip(trd[PUB_COL], trd["g19_metal"], trd["g19_metal_state"])
                if isinstance(s, str) and (p, e) in te_pe and s not in te_states))
    if "V3" in levels:
        te_sys = _set(ted, "extractant_system_key")
        te_comp = set().union(*[_components(k, component_map) for k in te_sys]) if te_sys else set()
        tr_sys = _set(trd, "extractant_system_key")
        record("V3_shared_system", tr_sys & te_sys)
        record("V3_component_sharing_system", {k for k in tr_sys - te_sys if _components(k, component_map) & te_comp})
    if "V4" in levels:
        if hidden_families is None or family_map is None:
            raise ValueError("level V4 needs hidden_families and family_map")
        hid = set(hidden_families)

        def fams(key: str) -> set:
            out: set = set()
            for c in str(key).split("|"):
                v = family_map.get(c, ())
                out.update({v} if isinstance(v, str) else set(v))
            return out
        record("V4_train_system_contains_hidden_family",
               {k for k in _set(trd, "extractant_system_key") if fams(k) & hid})
    if "V5" in levels or "V6" in levels:
        comp_aware = component_aware or "V6" in levels

        def cells(frame: pd.DataFrame) -> set:
            return {(s, k) for s, k in zip(frame["g19_metal_state"], frame["extractant_system_key"])
                    if isinstance(s, str) and isinstance(k, str)}

        def el_cells(frame: pd.DataFrame, unknown_only: bool) -> set:
            return {(e, k) for e, s, k in zip(frame["g19_metal"], frame["g19_metal_state"],
                                              frame["extractant_system_key"])
                    if isinstance(e, str) and isinstance(k, str) and (not unknown_only or not isinstance(s, str))}

        te_cells = cells(ted)
        record("V5_shared_cell", cells(trd) & te_cells)
        alias = (el_cells(trd, True) & el_cells(ted, False)) | (el_cells(trd, False) & el_cells(ted, True))
        record("V5_unknown_state_alias", alias)
        hidden_components: dict[str, set[str]] = {}
        for s, k in te_cells:
            hidden_components.setdefault(s, set()).update(k.split("|"))
        warn["V5_train_rows_sharing_component_with_hidden_cell"] = int(sum(
            1 for s, k in zip(trd["g19_metal_state"], trd["extractant_system_key"])
            if isinstance(s, str) and isinstance(k, str) and (s, k) not in te_cells
            and hidden_components.get(s, set()) & set(k.split("|"))))
        if comp_aware:
            # the registered state-level rule, applied in every system sharing a component with a hidden cell:
            # no train row of the hidden STATE and no X(?) row of its element there; other known states of the
            # element may stay (counted under warnings)
            hidden_by_elem: dict[str, list[tuple[str, str, set[str]]]] = {}
            for s, k in te_cells:
                hidden_by_elem.setdefault(str(s).split("(")[0], []).append((s, k, _components(k, component_map)))
            bad, other_state = set(), 0
            for i, e, s, k in zip(trd.index, trd["g19_metal"], trd["g19_metal_state"], trd["extractant_system_key"]):
                if not (isinstance(e, str) and isinstance(k, str)) or e not in hidden_by_elem:
                    continue
                kc = _components(k, component_map)
                sharing = [hs for hs, hk, hc in hidden_by_elem[e] if k != hk and kc & hc]
                if not sharing:
                    continue
                if not isinstance(s, str) or s in sharing:
                    bad.add(i)
                else:
                    other_state += 1
            record("V5_hidden_state_in_component_sharing_system", bad)
            warn["V5_other_known_state_of_hidden_element_in_component_sharing_system_rows"] = other_state
    if near_dup_sig is not None:
        both_frames = pd.concat([trd, ted])
        keys = near_duplicate_key(both_frames, sig=near_dup_sig).to_numpy(dtype=object)
        ktr, kte = keys[: len(trd)], keys[len(trd):]
        shared = set(ktr) & set(kte)
        name = f"near_duplicate_key_sig{near_dup_sig}_shared"
        if near_dup_value_tol is not None and shared:
            ytr = pd.to_numeric(trd["log_D"], errors="coerce").to_numpy(dtype=float)
            yte = pd.to_numeric(ted["log_D"], errors="coerce").to_numpy(dtype=float)
            by_tr: dict[str, list[float]] = {}
            for k, v in zip(ktr, ytr):
                if k in shared and np.isfinite(v):
                    by_tr.setdefault(k, []).append(v)
            close = set()
            for k, v in zip(kte, yte):
                if k in by_tr and np.isfinite(v) and np.min(np.abs(np.asarray(by_tr[k]) - v)) <= near_dup_value_tol + 1e-12:
                    close.add(k)
            shared = close
            name += f"_within_{near_dup_value_tol:g}"
        viol[name] = len(shared)
        ex[name] = key_id(sorted(shared)[:n_examples]).tolist() if shared else []
    ok = all(v == 0 for v in viol.values())
    report = {"ok": ok, "levels": list(levels), "n_train": int(len(tr)), "n_test": int(len(te)),
              "near_dup_sig": near_dup_sig, "near_dup_value_tol": near_dup_value_tol, "element_level": element_level,
              "component_aware": bool(component_aware or "V6" in levels), "component_map": component_map is not None,
              "violations": viol, "examples": ex, "warnings": warn}
    if raise_on_violation and not ok:
        raise AssertionError("fold isolation violated: " + ", ".join(f"{k}={v}" for k, v in viol.items() if v))
    return report
