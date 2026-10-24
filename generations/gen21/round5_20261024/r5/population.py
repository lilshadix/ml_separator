"""E-CT sibling population (CLAUDE_CODE_PROMPT.md section 6.1, with the section-5.1 bit-identity rule).

Adapted (copied, never run in place) from round5_design_20261023/probes/scripts/P1_ct_build.py
(LF SHA-256 afd7ae542f24df305c6f174fd3e816948b28298720d543772b685ded9d7138d0), pool section only, and the OFF_PUB /
OFF_XEXT pools of probes/scripts/P1_ct_diag.py (LF SHA-256 4234aa3d51a49690d75288f4c2b4907692806d6a5955b17111fa9cc2c8f66b2d).

**Metadata only.**  ``build_pools`` sees a whitelisted metadata frame (cell, publication, extractant SMILES, chemotype,
``metals`` and the ``cond__`` fields) and a coverage mask derived from the ``metals`` metadata string; it never sees a
log D, a replicate sd or a prediction.  The only label-derived input is the optional set of bit-identity links
(``r5.dupes.bit_identity``), which removes links, never cells (section 5.1).

Definitions (frozen; section 6.1):
  sibling of target t: any other bench cell (all 521, not only C3) with the same ``publication_id``, the same extractant
      SMILES and a finite V1 log D at every metal t has.  V1 pool: every such sibling.
  CLEAN: t and the sibling outside the round-3 V2-1 / V2-2 / V2-3 lists (r3.datav2), and the sibling's recorded cond__
      metadata differs from t's in at least one field family other than contact time / metal concentration (phase
      ratio is not recorded, so a phase-ratio-only sibling shows as "no recorded difference" and is dropped).
  BIT: CLEAN minus the links flagged by the section-5.1 bit-identity rule; n_sib = |BIT|.  The same links are also
      removed from the near, placebo and OFF pools.
  near: the CLEAN (BIT) siblings at minimum condition distance |dlog10 acid| + |dlog10 extractant conc| + 1[diluent
      differs] (concentrations floored at 1e-4 M; one side missing adds 1, both missing add 0); exact ties kept together.
  placebo pools (non-V2 covering cells): plc_chem same chemotype, other publications; plc_xext same extractant, other
      publications; plc_pub same publication, other extractants.
  OFF pools (C3 cells only, they need B1's OOF residual): off_sibs = the CLEAN siblings that are C3; off_all = every
      other non-V2 C3 cell of the publication x extractant; off_pub = every other non-V2 C3 cell of the publication;
      off_xext = off_pub with another extractant.
Strata (``strata``): PRIMARY = BIT, n_sib >= 2, without C5-BTBP; sensitivities WITH_C5, GE1, EQ1 (the harm rule),
P1's V1 stratum (V1 pool, n >= 2, without C5) and every row of P1's population table with and without the bit rule.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import paths

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/probes/scripts/P1_ct_build.py":
        "afd7ae542f24df305c6f174fd3e816948b28298720d543772b685ded9d7138d0",
    "generations/gen21/round5_design_20261023/probes/scripts/P1_ct_diag.py":
        "4234aa3d51a49690d75288f4c2b4907692806d6a5955b17111fa9cc2c8f66b2d",
}

C5 = paths.C5
FAMS_TRIVIAL = frozenset({"contact_time", "metal_conc"})
BASE_COLS = ("cell_id", "publication_id", "extractant", "chemotype", "metals")
FORBIDDEN_PREFIX = ("logD__", "repsd__")
LN = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")


# ------------------------------------------------------------------------------------------------------------
# metadata helpers (P1_ct_build.py)
# ------------------------------------------------------------------------------------------------------------
def metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The whitelisted metadata columns of the bench frame: identity, ``metals`` and every cond__ field."""
    cols = list(BASE_COLS) + [c for c in frame.columns if c.startswith("cond__")]
    f = frame[cols].reset_index(drop=True).copy()
    assert_metadata_only(f)
    return f


def assert_metadata_only(f: pd.DataFrame) -> None:
    bad = [c for c in f.columns if c.startswith(FORBIDDEN_PREFIX) or c in ("y", "B1", "B0", "D", "log_D", "D_raw")]
    if bad:
        raise AssertionError(f"population builder must be metadata-only; label columns present: {bad[:5]}")


def finite_from_metals(f: pd.DataFrame) -> np.ndarray:
    """Coverage mask (cells x 14 lanthanides) from the ``metals`` metadata string ('La;Ce;...')."""
    out = np.zeros((len(f), len(LN)), bool)
    idx = {m: k for k, m in enumerate(LN)}
    for i, s in enumerate(f["metals"].astype(str)):
        for m in s.split(";"):
            m = m.strip()
            if m:
                out[i, idx[m]] = True
    return out


def v2_lists() -> dict:
    """Round-3 V2 lists (next_research_20261019/r3/datav2.py): V2-1 re-imports, V2-2 dropped NaN-contact-time
    members (value = the kept member), V2-3 hidden-condition merges, V2-5 floor readings."""
    from r3 import datav2 as V2
    return {"V2-1": set(V2.V2_1_CELLS), "V2-2": dict(V2.V2_2_DROP), "V2-3": set(V2.V2_3_CELLS),
            "V2-5": dict(V2.V2_5_FLOOR)}


def v2drop_set() -> set:
    L = v2_lists()
    return set(L["V2-1"]) | set(L["V2-2"]) | set(L["V2-3"])


def fam(c: str) -> str:
    if c.startswith("cond__acid__"):
        return "acid_type"
    if c == "cond__acid_concentration_M":
        return "acid_conc"
    if c.startswith("cond__additive__"):
        return "additive"
    if c.startswith("cond__diluent__"):
        return "diluent"
    return {"cond__contact_time_min": "contact_time", "cond__extractant_concentration_M": "extr_conc",
            "cond__metal_concentration_mM": "metal_conc", "cond__temperature_C": "temperature"}[c]


def change_class(fams_list) -> str:
    """Cell-level condition-change type from the union of its siblings' differing field families."""
    U = set().union(*fams_list) if fams_list else set()
    U = U - FAMS_TRIVIAL
    if U == {"acid_conc"}:
        return "acid_conc_only"
    if U and U <= {"acid_conc", "acid_type"}:
        return "acid_type(+conc)"
    if U == {"diluent"}:
        return "diluent_only"
    if U == {"extr_conc"}:
        return "extr_conc_only"
    if U == {"temperature"}:
        return "temperature_only"
    return "mixed"


class CondMeta:
    """cond__ comparisons of P1_ct_build.py on a metadata frame."""

    def __init__(self, f: pd.DataFrame):
        self.f = f
        self.cols = [c for c in f.columns if c.startswith("cond__")]
        self.F = f[self.cols].to_numpy(float)
        self.fams = [fam(c) for c in self.cols]
        self.dil = np.array([c.startswith("cond__diluent__") for c in self.cols])
        self.ia = self.cols.index("cond__acid_concentration_M")
        self.il = self.cols.index("cond__extractant_concentration_M")

    def diff_fams(self, i: int, j: int) -> frozenset:
        x, y = self.F[i], self.F[j]
        nx, ny = np.isnan(x), np.isnan(y)
        with np.errstate(invalid="ignore"):
            d = (nx != ny) | (~nx & ~ny & ~np.isclose(x, y))
        return frozenset(self.fams[k] for k in np.where(d)[0])

    def cond_dist(self, i: int, j: int) -> float:
        def lg(v):
            return math.log10(max(float(v), 1e-4))
        d = 0.0
        for k in (self.ia, self.il):
            x, y = self.F[i, k], self.F[j, k]
            if np.isnan(x) and np.isnan(y):
                continue
            if np.isnan(x) or np.isnan(y):
                d += 1.0
                continue
            d += abs(lg(x) - lg(y))
        dx, dy = np.nan_to_num(self.F[i, self.dil]), np.nan_to_num(self.F[j, self.dil])
        return d + (0.0 if np.allclose(dx, dy) else 1.0)


# ------------------------------------------------------------------------------------------------------------
# pools
# ------------------------------------------------------------------------------------------------------------
def _link(a: str, b: str) -> frozenset:
    return frozenset((a, b))


def build_pools(f: pd.DataFrame, finite: np.ndarray, c3, v2drop, bit_links=()) -> dict:
    """Per C3 target: the V1 / CLEAN / BIT sibling pools, near siblings, placebo and OFF pools (lists of cell ids),
    the differing cond__ families per V1 sibling ('NONE' = no recorded difference), the change class over the CLEAN
    (``cls_clean``) and BIT (``cls``) siblings, and the BIT-removed links.  ``f`` must be a
    metadata frame (``metadata_frame``); ``bit_links`` is an iterable of flagged (cell_a, cell_b) pairs."""
    assert_metadata_only(f)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f["publication_id"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    chem = f["chemotype"].astype(str).to_numpy()
    cm = CondMeta(f)
    v2drop = set(v2drop)
    c3 = sorted(set(c3))
    c3set = set(c3)
    bits = {_link(a, b) for a, b in bit_links}
    okp = np.array([c not in v2drop for c in ids])
    out = {}
    for t in c3:
        i = pos[t]
        met = np.where(finite[i])[0]
        cover = np.all(finite[:, met], axis=1)
        cover[i] = False
        v1 = [ids[j] for j in np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]]
        clean, fams = [], {}
        for s in v1:
            dfm = cm.diff_fams(i, pos[s])
            fams[s] = ",".join(sorted(dfm)) or "NONE"
            if t not in v2drop and s not in v2drop and bool(dfm - FAMS_TRIVIAL):
                clean.append(s)
        removed = [s for s in clean if _link(t, s) in bits]
        bit = [s for s in clean if _link(t, s) not in bits]

        def near_of(pool):
            if not pool:
                return []
            dd = np.array([cm.cond_dist(i, pos[s]) for s in pool])
            return [pool[q] for q in np.where(np.isclose(dd, dd.min(), atol=1e-9))[0]]

        def drop_bits(pool):
            return [s for s in pool if _link(t, s) not in bits]
        plc_chem = [ids[j] for j in np.where((chem == chem[i]) & (pub != pub[i]) & cover & okp)[0]]
        plc_xext = [ids[j] for j in np.where((ext == ext[i]) & (pub != pub[i]) & cover & okp)[0]]
        plc_pub = [ids[j] for j in np.where((pub == pub[i]) & (ext != ext[i]) & cover & okp)[0]]
        tok = t not in v2drop
        off_all = [c for c in c3 if tok and c != t and pub[pos[c]] == pub[i] and ext[pos[c]] == ext[i] and c not in v2drop]
        off_pub = [c for c in c3 if c != t and c not in v2drop and pub[pos[c]] == pub[i]]
        off_xext = [c for c in off_pub if ext[pos[c]] != ext[i]]
        P = {"publication": pub[i], "extractant": ext[i], "chemotype": chem[i], "c5": pub[i] == C5,
             "t_v2drop": t in v2drop, "v1": v1, "clean": clean, "bit": bit, "bit_removed": removed,
             "near_clean": near_of(clean), "near": near_of(bit),
             "off_sibs_clean": [s for s in clean if s in c3set], "off_sibs": [s for s in bit if s in c3set],
             "sib_fams": fams, "cls_clean": change_class([set(fams[s].split(",")) for s in clean]) if clean else "",
             "cls": change_class([set(fams[s].split(",")) for s in bit]) if bit else ""}
        for nm, pool in (("plc_chem", plc_chem), ("plc_xext", plc_xext), ("plc_pub", plc_pub), ("off_all", off_all),
                         ("off_pub", off_pub), ("off_xext", off_xext)):
            P[nm + "_clean"] = pool
            P[nm] = drop_bits(pool)
        out[t] = P
    return out


# ------------------------------------------------------------------------------------------------------------
# strata and counts
# ------------------------------------------------------------------------------------------------------------
def strata(pools: dict) -> dict:
    """Target lists per stratum.  ``*_NOBIT`` rows are P1's CLEAN rule without the bit-identity check (P1's
    reproduction target, a sensitivity); V1 rows use the V1 pool (no CLEAN, no bit rule)."""
    def sel(pool, lo, hi=None, c5=None):
        out = []
        for t, P in sorted(pools.items()):
            n = len(P[pool])
            if n < lo or (hi is not None and n > hi):
                continue
            if c5 is not None and P["c5"] != c5:
                continue
            out.append(t)
        return out
    S = {"PRIMARY": sel("bit", 2, c5=False), "WITH_C5": sel("bit", 2), "GE1": sel("bit", 1, c5=False),
         "GE1_WITH_C5": sel("bit", 1), "EQ1": sel("bit", 1, 1, c5=False),
         "PRIMARY_NOBIT": sel("clean", 2, c5=False), "WITH_C5_NOBIT": sel("clean", 2), "GE1_NOBIT": sel("clean", 1, c5=False),
         "GE1_WITH_C5_NOBIT": sel("clean", 1), "EQ1_NOBIT": sel("clean", 1, 1, c5=False),
         "V1_GE1": sel("v1", 1), "V1_GE2_NOC5": sel("v1", 2, c5=False)}
    return S


def counts(cells, pools: dict, rows_per_seed: dict) -> dict:
    """COUNTS only: cells, publications, publication x chemotype units, rows per seed per design, the change-class
    mix and the BIT-removed links inside the stratum.  ``rows_per_seed``: {design: {seed: {cell: n_rows}}}."""
    cells = sorted(cells)
    pubs = sorted({pools[c]["publication"] for c in cells})
    units = sorted({(pools[c]["publication"], pools[c]["chemotype"]) for c in cells})
    rps = {}
    for d, bys in rows_per_seed.items():
        per = {str(s): int(sum(m.get(c, 0) for c in cells)) for s, m in bys.items()}
        rps[d] = {"per_seed": per, "distinct": sorted(set(per.values()))}
    removed = sorted({tuple(sorted((c, s))) for c in cells for s in pools[c]["bit_removed"]})
    return {"cells": len(cells), "publications": len(pubs), "pub_x_chemotype_units": len(units),
            "rows_per_seed": rps, "c5_cells": int(sum(pools[c]["c5"] for c in cells)),
            "n_sib_bit": {int(k): int(v) for k, v in sorted(pd.Series([len(pools[c]["bit"]) for c in cells],
                                                                       dtype=int).value_counts().items())},
            "change_class": {str(k): int(v) for k, v in sorted(pd.Series([pools[c]["cls"] or "NONE" for c in cells],
                                                                          dtype=object).value_counts().items())},
            "bit_flagged_sibling_links": [list(x) for x in removed], "n_bit_flagged_sibling_links": len(removed),
            "targets_with_a_bit_flagged_sibling": sorted(c for c in cells if pools[c]["bit_removed"])}


def rows_per_seed_from_ref(designs=paths.DESIGNS) -> dict:
    """{design: {seed: {cell: n scored rows}}} from the round-4 REF parquets, reading the row-identity columns only."""
    out = {}
    for d in designs:
        t = pd.read_parquet(paths.R4PRED / f"REF_{d}.parquet", columns=["split_seed", "cell_id", "k"])
        if not (t.k == 3).all():
            raise AssertionError(f"REF_{d}: expected k = 3 rows only")
        out[d] = {int(s): g.groupby("cell_id").size().astype(int).to_dict() for s, g in t.groupby("split_seed")}
    return out


def c3_cells_from_ref(designs=paths.DESIGNS) -> list:
    sets = [set(pd.read_parquet(paths.R4PRED / f"REF_{d}.parquet", columns=["cell_id"]).cell_id.astype(str)) for d in designs]
    if any(s != sets[0] for s in sets):
        raise AssertionError("REF parquets of BP and BPG hold different C3 cells")
    return sorted(sets[0])


def pools_table(pools: dict, S: dict) -> pd.DataFrame:
    """One metadata row per C3 target: pool sizes, change class and stratum membership (no label, no score)."""
    rows = []
    for t, P in sorted(pools.items()):
        r = {"cell_id": t, "publication": P["publication"], "chemotype": P["chemotype"], "c5": P["c5"],
             "t_v2drop": P["t_v2drop"], "n_v1": len(P["v1"]), "n_clean": len(P["clean"]), "n_sib": len(P["bit"]),
             "n_bit_removed": len(P["bit_removed"]), "bit_removed": ";".join(P["bit_removed"]), "cls": P["cls"],
             "cls_clean": P["cls_clean"], "n_near": len(P["near"])}
        for nm in ("plc_chem", "plc_xext", "plc_pub", "off_sibs", "off_all", "off_pub", "off_xext"):
            r["n_" + nm] = len(P[nm])
        for k, v in S.items():
            r["in_" + k] = t in set(v)
        rows.append(r)
    return pd.DataFrame(rows)


def links_table(pools: dict) -> pd.DataFrame:
    """Every V1 sibling link (target, sibling) with its differing cond__ families and CLEAN / BIT status."""
    rows = []
    for t, P in sorted(pools.items()):
        cl, bt = set(P["clean"]), set(P["bit"])
        for s in P["v1"]:
            rows.append({"target": t, "sibling": s, "publication": P["publication"], "clean": s in cl, "bit": s in bt,
                         "bit_removed": s in set(P["bit_removed"]), "diff": P["sib_fams"].get(s, ""),
                         "target_v2drop": P["t_v2drop"]})
    return pd.DataFrame(rows)


def duplicate_cell_sets() -> dict:
    """Duplicate sensitivities of the E-A-row items (brief 5.3, 9.3; s0-fidelity-8): noV2 = the round-3 V2-1 / V2-2 /
    V2-3 drop set; noBITDUP = the second cell of every Stage-0 flagged bit-identity link (results/stage0/
    dupscan_links.csv, flag_bit); V2-3 = the hidden-merge cells (11 among the C3 cells), reported apart as a
    condition-aware stratum while request S1 is open (DATA_CONTRACT 1)."""
    from . import paths as _P
    v = v2_lists()
    out = {"noV2": set(map(str, v2drop_set())), "V2-3": set(map(str, v["V2-3"]))}
    d = pd.read_csv(_P.RES / "stage0" / "dupscan_links.csv")
    flag = d.flag_bit.astype(str).str.lower() == "true"
    out["noBITDUP"] = set(d[flag].cell_b.astype(str))
    return out

