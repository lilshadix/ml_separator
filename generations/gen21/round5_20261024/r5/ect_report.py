"""E-CT report tables (CLAUDE_CODE_PROMPT.md section 6.1; conventions 6.0, 9.1-9.3): comparators on identical rows,
placebos, the read ladder, dose-response, decompositions, the mechanism contrast, the read-alignment identity, the
inference of 6.1, influence rows, duplicate and V2-5 sensitivities, the descriptive annex rows, the laboratory
annotation, the report rules and the section-9.3 per-item table.  Every row carries its section-9.1 label.

Copied and adapted (never run in place) from round5_design_20261023/probes/scripts/P1_ct_analyze.py
(LF SHA-256 d9283ec094966eef325a11a833eeede001142d9e150cd0f575951c3b77fc346f) and P1_ct_diag.py
(LF SHA-256 4234aa3d51a49690d75288f4c2b4907692806d6a5955b17111fa9cc2c8f66b2d); shift-1 bins from
ideas2/scripts/shift1_skA.py (LF SHA-256 0396f409c54a764d87a8ddd10615acf48d09f461865c207351839b60064456e4).

Conventions: R^2 pooled within each split seed, then the seed mean (round 4); arms named E_* are exact expected squared
errors and enter as SSE; every comparison is restricted to the rows on which all of its arms exist (identical rows),
and the row count is reported.  Unit = publication x chemotype; c_u = mean_s (SSE_ref,u,s - SSE_cand,u,s) / SST_s, so
sum_u c_u is the pooled contrast exactly.  Exact one-sided sign-flips over units (full enumeration up to 22 units),
the chemotype-block flip (floor 1 / 2^G_chem), SIGN, IM t on the per-unit relative SSE and on the log ratio, the
leave-one-publication-out jackknife with t(P - 1), the top-3 share and the equal-weight per-publication relative SSE.
Cluster-level inference is "not estimable" (3 cpg clusters in PRIMARY).  No E-CT value is ever placed next to E-A's
0.452: B1 on the same rows is the comparator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as CM
from . import ect as E
from . import paths
from r4 import metrics as M4  # noqa: E402

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/probes/scripts/P1_ct_analyze.py":
        "d9283ec094966eef325a11a833eeede001142d9e150cd0f575951c3b77fc346f",
    "generations/gen21/round5_design_20261023/probes/scripts/P1_ct_diag.py":
        "4234aa3d51a49690d75288f4c2b4907692806d6a5955b17111fa9cc2c8f66b2d",
    "generations/gen21/round5_design_20261023/ideas2/scripts/shift1_skA.py":
        "0396f409c54a764d87a8ddd10615acf48d09f461865c207351839b60064456e4",
}

SEEN, SENS, CC, DIAG, SMOKE = ("EXPLORATORY (seen)", "SENSITIVITY", "CONSISTENCY CHECK", "DIAGNOSTIC", "TECHNICAL SMOKE")
EXACT_MAX = 22
NOT_NEXT_TO = "E-CT value: never placed next to E-A's 0.452; B1 on the same rows is the comparator"
OFS_LAB = "SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*)"
# stratum -> (row flag, arm prefix of the pool arms, label)
STRATA = {
    "PRIMARY": ("in_PRIMARY", "", SEEN), "WITH_C5": ("in_WITH_C5", "", SENS), "GE1": ("in_GE1", "", SENS),
    "GE1_WITH_C5": ("in_GE1_WITH_C5", "", SENS), "EQ1": ("in_EQ1", "", CC),
    "PRIMARY_NOBIT": ("in_PRIMARY_NOBIT", "NB_", SENS), "WITH_C5_NOBIT": ("in_WITH_C5_NOBIT", "NB_", SENS),
    "GE1_NOBIT": ("in_GE1_NOBIT", "NB_", SENS), "EQ1_NOBIT": ("in_EQ1_NOBIT", "NB_", SENS),
    "V1_GE1": ("in_V1_GE1", "V1_", SENS), "V1_GE2_NOC5": ("in_V1_GE2_NOC5", "V1_", SENS),
}
POOL_ARMS = ("COPY", "SH0", "SHK2_r4", "SIB_r2", "SIB_r3", "SIB_r4", "SH_r2", "SH_r3", "SH_r4", "OFF", "OFF_ALL")
CE_PAIRS, GD_PAIRS = ("La/Ce", "Ce/Pr"), ("Eu/Gd", "Gd/Tb")
PAIR_ORDER = ["La/Ce", "Ce/Pr", "Pr/Nd", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu"]
NBINS = ([0, 1, 2, 4, 7, 10 ** 6], ["1", "2", "3-4", "5-7", ">=8"])
DX_BINS = ([-0.01, 0.15, 0.30, 0.60, 5.0], ["<=0.15", "0.15-0.30", "0.30-0.60", ">0.60"])
P1_SEEN = {"BPG": (0.177, " with 13 of 14"), "BP": (0.174, "")}   # P1's seen PRIMARY SH - B1 without the bit rule
#   (BPG units up from brief 6.1's report rules; BP from P1_condition_transfer.md, which gives no BP unit count)


# ------------------------------------------------------------------------------------------------------------
# metrics on identical rows (E_* arms are expected squared errors)
# ------------------------------------------------------------------------------------------------------------
def err2(d: pd.DataFrame, arm: str) -> np.ndarray:
    if arm.startswith("E_"):
        return d[arm].to_numpy(float)
    return (d.y.to_numpy(float) - d[arm].to_numpy(float)) ** 2


def _codes(d: pd.DataFrame) -> np.ndarray:
    return pd.factorize(d.split_seed, sort=True)[0]


def _sst(d: pd.DataFrame) -> np.ndarray:
    c, y = _codes(d), d.y.to_numpy(float)
    n = np.bincount(c)
    mu = np.bincount(c, y) / n
    return np.bincount(c, (y - mu[c]) ** 2)


def r2_seeds(d: pd.DataFrame, arm: str) -> np.ndarray:
    if len(d) == 0:
        return np.array([])
    return 1.0 - np.bincount(_codes(d), err2(d, arm)) / _sst(d)


def r2(d, arm) -> float:
    v = r2_seeds(d, arm)
    return float(v.mean()) if len(v) else float("nan")


def mse(d, arm) -> float:
    c = _codes(d)
    return float((np.bincount(c, err2(d, arm)) / np.bincount(c)).mean()) if len(d) else float("nan")


def relsse(d, cand, ref) -> float:
    c = _codes(d)
    return float((1.0 - np.bincount(c, err2(d, cand)) / np.bincount(c, err2(d, ref))).mean()) if len(d) else float("nan")


def mae(d, arm) -> float:
    if arm.startswith("E_") or not len(d):
        return float("nan")
    c = _codes(d)
    return float((np.bincount(c, np.abs(d.y.to_numpy(float) - d[arm].to_numpy(float))) / np.bincount(c)).mean())


def bal_mae(d, arm) -> float:
    if arm.startswith("E_") or not len(d):
        return float("nan")
    return CM.bal_mae(d, arm)


def complete(d: pd.DataFrame, arms) -> pd.DataFrame:
    arms = [a for a in dict.fromkeys(arms)]
    ok = np.ones(len(d), bool)
    for a in arms:
        ok &= np.isfinite(d[a].to_numpy(float))
    return d[ok]


def describe(d: pd.DataFrame) -> dict:
    ns = d.split_seed.nunique()
    if not len(d):
        return {"cells": 0, "publications": 0, "units": 0, "rows_per_seed": 0.0, "cpg_clusters": 0}
    first = d[d.split_seed == d.split_seed.min()]
    cl = first.groupby("cluster_cpg").size()
    return {"cells": int(d.cell_id.nunique()), "publications": int(d.publication.nunique()),
            "units": int(d.pubchem.nunique()), "rows_per_seed": round(len(d) / ns, 1), "cpg_clusters": int(len(cl)),
            "largest_cluster_share": round(float(cl.max() / cl.sum()), 3), "kish_cpg": round(float(M4.kish(first, "cluster_cpg")), 2)}


def arm(prefix: str, name: str, cols) -> str:
    return prefix + name if prefix and name in POOL_ARMS and prefix + name in cols else name


def shared(a: str) -> bool:
    """Arms that do not depend on the sibling pool (valid in every pool variant's stratum)."""
    return a.startswith(("B1_", "B0_")) or a == "G14Z"


def pool_ok(prefix: str, a: str) -> bool:
    return (a.startswith(prefix) or shared(a)) if prefix else True


def stratum(R: pd.DataFrame, name: str) -> pd.DataFrame:
    flag = STRATA[name][0] if name in STRATA else name
    return R[R[flag].astype(bool)]


def arm_table(d: pd.DataFrame, arms, *, ref: str = "B1_r4", stratum_name: str = "", label: str = SEEN,
              extra: dict | None = None) -> pd.DataFrame:
    """R^2 (seed mean, min, max, per seed), MSE, MAE, balanced MAE, relative SSE vs ref, on the rows where every arm
    and ref exist."""
    arms = [a for a in arms if a in d.columns]
    dd = complete(d, list(arms) + [ref])
    info = describe(dd)
    rows = []
    for a in arms:
        rs = r2_seeds(dd, a)
        rows.append({"stratum": stratum_name, "arm": a, "R2": float(rs.mean()) if len(rs) else np.nan,
                     "R2_min_seed": float(rs.min()) if len(rs) else np.nan, "R2_max_seed": float(rs.max()) if len(rs) else np.nan,
                     "R2_per_seed": ";".join(f"{v:.4f}" for v in rs), "MSE": mse(dd, a), "MAE": mae(dd, a),
                     "MAE_balanced": bal_mae(dd, a), f"relSSE_vs_{ref}": relsse(dd, a, ref),
                     "coverage_of_stratum_rows": round(len(dd) / len(d), 4) if len(d) else np.nan, **info,
                     **(extra or {}), "label": label})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
# inference
# ------------------------------------------------------------------------------------------------------------
def unit_contrib(d: pd.DataFrame, cand: str, ref: str, unit=("publication", "chemotype")) -> pd.DataFrame:
    sst = dict(zip(sorted(d.split_seed.unique()), _sst(d)))
    ns = d.split_seed.nunique()
    e = d[list(unit) + ["split_seed"]].copy()
    e["ec"], e["er"] = err2(d, cand), err2(d, ref)
    e["c"] = (e.er - e.ec) / e.split_seed.map(sst).to_numpy(float) / ns
    g = e.groupby(list(unit)).agg(c=("c", "sum"), sse_c=("ec", "sum"), sse_r=("er", "sum"), rows=("c", "size"))
    g["sse_c"], g["sse_r"], g["rows"] = g.sse_c / ns, g.sse_r / ns, g.rows / ns
    g["dSSE"] = g.sse_r - g.sse_c                               # section 9.2: seed means; win = dSSE > 0, tie = exact 0
    g["win"], g["loss"], g["tie"] = g.dSSE > 0, g.dSSE < 0, g.dSSE == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        g["rho"] = 1.0 - g.sse_c / g.sse_r
        g["logratio"] = np.log(g.sse_r / g.sse_c)
    return g.reset_index()


def signflip_p(values, groups=None, *, key: str = "ECT", exact_max: int = EXACT_MAX) -> dict:
    """One-sided sign-flip p for sum > 0: exact enumeration (chunked) when G <= exact_max, else r5.common's random
    4,999 flips seeded by paths.seed_for(key).  ``groups`` flips whole blocks together."""
    x = np.asarray(values, float)
    if groups is not None:
        x = pd.Series(x).groupby(np.asarray(groups)).sum().to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("non-finite unit value")
    G = len(x)
    if G == 0:
        return {"p": float("nan"), "G": 0, "exact": True}
    T, tol = float(x.sum()), 1e-10 * float(np.abs(x).sum())
    if G <= exact_max:
        tot, ge, bits = 1 << G, 0, np.arange(G, dtype=np.int64)
        for a in range(0, tot, 1 << 16):
            ints = np.arange(a, min(tot, a + (1 << 16)), dtype=np.int64)
            S = 1.0 - 2.0 * ((ints[:, None] >> bits) & 1)
            ge += int(((S @ x) >= T - tol).sum())
        return {"p": ge / tot, "G": G, "exact": True}
    r = CM.sf_pool(x, key, exact_max=0)
    return {"p": r["p"], "G": G, "exact": False, "seed": r["seed"]}


def infer(d: pd.DataFrame, cand: str, ref: str, *, stratum_name: str, label: str, key: str | None = None) -> tuple[dict, pd.DataFrame]:
    """The section-6.1 inference of cand - ref on the rows where both exist; (summary row, unit table)."""
    dd = complete(d, [cand, ref])
    key = key or f"ECT|{stratum_name}|{cand}-{ref}"
    info = describe(dd)
    base = {"stratum": stratum_name, "cand": cand, "ref": ref, **info, "label": label}
    if dd.publication.nunique() < 2 or not len(dd):
        return {**base, "note": "fewer than 2 publications: not computed"}, pd.DataFrame()
    u = unit_contrib(dd, cand, ref)
    c = u.c.to_numpy(float)
    cz = np.where(np.abs(c) <= 1e-12, 0.0, c)
    tot = float(c.sum())
    pub_u = unit_contrib(dd, cand, ref, unit=("publication",))
    sign = CM.sign_test(cz)
    imr, iml = CM.im_t(u.rho.to_numpy(float)), CM.im_t(u.logratio.to_numpy(float))
    lo = CM.lopo(dd, lambda x: r2(x, cand) - r2(x, ref), by="publication")
    sf_u = signflip_p(c, key=key + "|unit")
    sf_c = signflip_p(c, u.chemotype.to_numpy(), key=key + "|chem")
    ds = r2_seeds(dd, cand) - r2_seeds(dd, ref)
    row = {**base, "R2_cand": r2(dd, cand), "R2_ref": r2(dd, ref), "dR2": float(ds.mean()),
           "dR2_per_seed": ";".join(f"{v:+.4f}" for v in ds), "relSSE": relsse(dd, cand, ref),
           "MSE_cand": mse(dd, cand), "MSE_ref": mse(dd, ref), "G_units": int(len(u)),
           "units_pos": int((c > 1e-6).sum()), "units_neg": int((c < -1e-6).sum()), "units_zero_1e-6": int((np.abs(c) <= 1e-6).sum()),
           "SIGN_W/L/T": f"{sign['W']}/{sign['L']}/{sign['T']}", "SIGN_p": sign["p"],
           "p_signflip_unit": sf_u["p"], "signflip_unit_exact": sf_u["exact"],
           "p_signflip_chemotype_blocks": sf_c["p"], "G_chemotypes": sf_c["G"],
           "IMt_rho_t": imr["t"], "IMt_rho_p": imr["p"], "IMt_log_t": iml["t"], "IMt_log_p": iml["p"],
           "top3_share": CM.top3_share(c), "eqw_pub_relSSE": float(pub_u.rho.mean()), "pubs_pos": int((pub_u.c > 0).sum()),
           "P": int(len(pub_u)), "lopo_se": lo["se"], "lopo_lo": lo["lo"], "lopo_hi": lo["hi"], "lopo_min": lo["min"],
           "lopo_min_without": lo["min_by"], "lopo_max": lo["max"], "lopo_max_without": lo["max_by"],
           "cluster_inference": "not estimable" if info.get("cpg_clusters", 0) <= 4 else "see cpg clusters",
           "contribution_total": tot}
    u = u.sort_values("c", ascending=False)
    u.insert(0, "contrast", f"{cand} - {ref}")
    u.insert(0, "stratum", stratum_name)
    u["label"] = label
    return row, u


# ------------------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------------------
MAIN_ARMS = ["B1_r4", "B0_r4", "SH_r4", "SIB_r4", "SHK2_r4", "OFF", "OFF_ALL", "NEAR_r4", "NEARSH_r4", "COPY", "SH0",
             "G14Z", "E_RAND1_raw_r4", "E_RAND1_sh_r4", "E_RAND1_raw_r0"]


def t_populations(R) -> pd.DataFrame:
    rows = []
    for nm, (_, pre, lab) in STRATA.items():
        d = stratum(R, nm)
        rows.append({"stratum": nm, "pool": {"": "BIT (CLEAN + bit rule)", "NB_": "CLEAN without bit rule",
                                             "V1_": "V1 (every covering sibling)"}[pre], **describe(d),
                     "c5_cells": int(d[d.c5].cell_id.nunique()) if len(d) else 0, "label": lab})
    for k in E.VARIANTS_R4:
        d = R[R[f"in_PRIMARY_{k}"]]
        if k == "V25":
            d = d[d.v25_row_ok]
        rows.append({"stratum": f"PRIMARY_{k}", "pool": k, **describe(d), "c5_cells": 0, "label": SENS})
    return pd.DataFrame(rows)


def t_comparators(R) -> pd.DataFrame:
    out = []
    for nm, (_, pre, lab) in STRATA.items():
        d = stratum(R, nm)
        if not len(d):
            continue
        arms = [a for a in (arm(pre, a, R.columns) for a in MAIN_ARMS) if pool_ok(pre, a)]
        out.append(arm_table(d, arms, stratum_name=nm, label=lab))
    return pd.concat(out, ignore_index=True)


def t_placebos(R) -> tuple[pd.DataFrame, list, list]:
    """Cross-publication and same-publication placebos at the arm's own weight (and count-matched), with the
    duplicate sensitivity (s0-items-1): the same placebos on pools with offset / near-identity links removed, read
    against OFS_SH_r4 (pool variant 'OFS')."""
    out, inf, units = [], [], []
    for nm in ("PRIMARY", "WITH_C5", "GE1"):
        d0 = stratum(R, nm)
        for pre, sh, off, lab in (("", "SH_r4", "OFF", STRATA[nm][2]), ("OFS_", "OFS_SH_r4", "OFS_OFF_ALL", OFS_LAB)):
            if sh not in R.columns:
                continue
            variant = {"pool_variant": "primary (bit rule)" if not pre else "OFS (offset / near-identity removed)"}
            for p in E.PB_POOLS:
                pb = f"{pre}PB_{p}"
                d = d0[d0[pb].notna() & d0[sh].notna()]
                if not len(d):
                    continue
                out.append(arm_table(d, ["B1_r4", sh, off, pb, pb + "_raw"], stratum_name=nm, label=lab,
                                     extra={"placebo": pb, "rows_with_pool_share": round(len(d) / len(d0), 4), **variant}))
                if d.publication.nunique() >= 2:
                    r, u = infer(d, sh, pb, stratum_name=nm, label=lab)
                    inf.append(r)
                    units.append(u)
                if p in E.CM_POOLS:
                    cm = f"E_{pre}PB_{p}_CM"
                    dc = d0[d0[cm].notna() & d0[sh].notna()]
                    if len(dc):
                        out.append(arm_table(dc, ["B1_r4", sh, pb, cm, cm + "_raw"], stratum_name=nm, label=lab,
                                             extra={"placebo": cm + " (count-matched)",
                                                    "rows_with_pool_share": round(len(dc) / len(d0), 4), **variant}))
                        if dc.publication.nunique() >= 2:
                            r, u = infer(dc, sh, cm, stratum_name=nm, label=lab)
                            inf.append(r)
                            units.append(u)
    return pd.concat(out, ignore_index=True), inf, units


LADDER_ARMS = [(0, ["COPY", "SH0", "G14Z", "E_RAND1_raw_r0"]), (2, ["B1_r2", "B0_r2", "SIB_r2", "SH_r2"]),
               (3, ["B1_r3", "B0_r3", "SIB_r3", "SH_r3"]), (4, ["B1_r4", "B0_r4", "SIB_r4", "SH_r4", "OFF"])]


def t_ladder(R) -> pd.DataFrame:
    rows = []
    for nm in ("PRIMARY", "WITH_C5", "GE1", "EQ1", "PRIMARY_NOBIT", "V1_GE1", "V1_GE2_NOC5"):
        _, pre, lab = STRATA[nm]
        d0 = stratum(R, nm)
        lad = [(rd, [a2 for a2 in (arm(pre, a, R.columns) for a in arms) if pool_ok(pre, a2)])
               for rd, arms in LADDER_ARMS + ([(4, ["SHK2_r4"])] if pre else [])]
        d = complete(d0, [a for _, arms in lad for a in arms])
        for rd, arms in lad:
            for a2 in arms:
                rows.append({"stratum": nm, "reads": rd, "arm": a2, "R2": r2(d, a2), "MSE": mse(d, a2),
                             "relSSE_vs_B1_r4": relsse(d, a2, "B1_r4"), **describe(d), "label": lab})
    return pd.DataFrame(rows)


def t_subm(R) -> pd.DataFrame:
    rows = []
    base = R[R.in_GE1_WITH_C5 & (R.n_sib >= 5)]
    for nm, d0, lab in (("n_sib>=5, without C5", base[~base.c5], SEEN), ("n_sib>=5, with C5", base, SENS)):
        arms = [f"E_SUB{m}_{k}_r4" for m in E.SUB_M for k in ("raw", "sh")] + ["SIB_r4", "SH_r4", "B1_r4", "OFF"]
        d = complete(d0, arms)
        cells = d.drop_duplicates("cell_id")
        for m in list(E.SUB_M) + ["all"]:
            raw, sh = (f"E_SUB{m}_raw_r4", f"E_SUB{m}_sh_r4") if m != "all" else ("SIB_r4", "SH_r4")
            rows.append({"population": nm, "m": str(m), "R2_raw_template": r2(d, raw), "R2_SH_m": r2(d, sh),
                         "MSE_raw": mse(d, raw), "MSE_SH_m": mse(d, sh), "R2_B1_r4": r2(d, "B1_r4"), "R2_OFF": r2(d, "OFF"),
                         **describe(d), "n_sib_distribution": cells.n_sib.value_counts().sort_index().to_dict(),
                         "cells_per_publication": cells.groupby("publication").size().to_dict(), "label": lab})
    return pd.DataFrame(rows)


def t_nsib(R) -> pd.DataFrame:
    """Between-cell n_sib strata (descriptive): GE1 without C5, P1's pool without the bit rule, and C5-BTBP in."""
    rows = []
    for nm, d0, lab in (("GE1 (CLEAN + bit, no C5)", stratum(R, "GE1"), SEEN), ("GE1_NOBIT", stratum(R, "GE1_NOBIT"), SENS),
                        ("GE1_WITH_C5 (CLEAN + bit, with C5)", stratum(R, "GE1_WITH_C5"), SENS)):
        pre = "NB_" if "NOBIT" in nm else ""
        nn = d0.n_sib if not pre else d0.n_NB
        bins = pd.cut(nn, NBINS[0], labels=NBINS[1])
        for b, d in d0.groupby(bins, observed=True):
            sh, sib, off = arm(pre, "SH_r4", R.columns), arm(pre, "SIB_r4", R.columns), arm(pre, "OFF", R.columns)
            rows.append({"population": nm, "n_sib": str(b), **describe(d), "R2_B1": r2(d, "B1_r4"), "R2_SH": r2(d, sh),
                         "R2_SIB": r2(d, sib), "R2_OFF": r2(d, off), "relSSE_SH_vs_B1": relsse(d, sh, "B1_r4"),
                         "relSSE_SH_vs_OFF": relsse(d, sh, off), "label": lab + " (descriptive between-cell strata)"})
    return pd.DataFrame(rows)


def t_pairtype(R) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, excl = [], []
    arms = ["B1_r4", "SH_r4", "OFF", "SIB_r4"]
    for nm in ("PRIMARY", "WITH_C5", "PRIMARY_NOBIT"):
        _, pre, lab = STRATA[nm]
        a4 = [arm(pre, a, R.columns) for a in arms]
        d = complete(stratum(R, nm), a4)
        ns = d.split_seed.nunique()
        e = pd.DataFrame({"pair": d.A + "/" + d.B})
        for a in a4:
            e[a] = err2(d, a)
        g = e.groupby("pair")[a4].sum() / ns
        g["rows_per_seed"] = e.groupby("pair").size() / ns
        g["gain_SH_vs_B1"] = g[a4[0]] - g[a4[1]]
        g["share_gain_SH_vs_B1"] = g.gain_SH_vs_B1 / g.gain_SH_vs_B1.sum()
        g["gain_SH_vs_OFF"] = g[a4[2]] - g[a4[1]]
        tot = g.gain_SH_vs_OFF.sum()
        g["share_gain_SH_vs_OFF"] = g.gain_SH_vs_OFF / tot if abs(tot) > 1e-12 else np.nan
        g = g.reindex([o for o in PAIR_ORDER if o in g.index] + [o for o in g.index if o not in PAIR_ORDER])
        g = g.reset_index().rename(columns={a: f"SSE_per_seed_{a}" for a in a4})
        g.insert(0, "stratum", nm)
        g["label"] = lab
        out.append(g)
        pair = d.A + "/" + d.B
        ce, gd = pair.isin(CE_PAIRS).to_numpy(), pair.isin(GD_PAIRS).to_numpy()
        share_ce = float(g.loc[g.pair.isin(CE_PAIRS), "gain_SH_vs_B1"].sum() / g.gain_SH_vs_B1.sum())
        share_gd = float(g.loc[g.pair.isin(GD_PAIRS), "gain_SH_vs_B1"].sum() / g.gain_SH_vs_B1.sum())
        for lb, m in (("all pairs", np.ones(len(d), bool)), ("excluding Ce pairs", ~ce), ("excluding Gd pairs", ~gd),
                      ("excluding Ce and Gd pairs", ~ce & ~gd)):
            dd = d[m]
            excl.append({"stratum": nm, "rows": lb, **describe(dd), **{f"R2_{a}": r2(dd, a) for a in a4},
                         "SH-B1": r2(dd, a4[1]) - r2(dd, a4[0]), "SH-OFF": r2(dd, a4[1]) - r2(dd, a4[2]),
                         "Ce_share_of_SH-B1_gain": share_ce, "Gd_share_of_SH-B1_gain": share_gd, "label": lab})
    return pd.concat(out, ignore_index=True), pd.DataFrame(excl)


def t_condclass(R) -> pd.DataFrame:
    rows = []
    for nm in ("PRIMARY", "GE1", "WITH_C5"):
        lab = STRATA[nm][2]
        d0 = complete(stratum(R, nm), ["B1_r4", "SH_r4", "SIB_r4", "OFF", "COPY"])
        ns = d0.split_seed.nunique()
        tot = float((err2(d0, "B1_r4") - err2(d0, "SH_r4")).sum() / ns)
        for c, d in d0.groupby("cls"):
            g = float((err2(d, "B1_r4") - err2(d, "SH_r4")).sum() / ns)
            rows.append({"stratum": nm, "class": c, **describe(d), "R2_B1": r2(d, "B1_r4"), "R2_SH": r2(d, "SH_r4"),
                         "R2_SIB": r2(d, "SIB_r4"), "R2_OFF": r2(d, "OFF"), "R2_COPY": r2(d, "COPY"),
                         "SSEgain_SH_vs_B1_per_seed": g,
                         "SSEgain_SH_vs_OFF_per_seed": float((err2(d, "OFF") - err2(d, "SH_r4")).sum() / ns),
                         "share_of_SH-B1_gain": g / tot if tot else np.nan,
                         "publications_in_class": ";".join(sorted(d.publication.unique())),
                         "reported_separately": c in ("diluent_only", "extr_conc_only"), "label": lab})
    return pd.DataFrame(rows)


def t_alignment(R) -> pd.DataFrame:
    """SH = OFF exactly where every sibling was read at the target's metals in the same fold (DIAGNOSTIC)."""
    rows = []
    for nm in ("PRIMARY", "WITH_C5", "GE1", "PRIMARY_NOBIT"):
        _, pre, _ = STRATA[nm]
        sh, off = arm(pre, "SH_r4", R.columns), arm(pre, "OFF", R.columns)
        d = complete(stratum(R, nm), [sh, off, "B1_r4"])
        gs = d.geom_same.to_numpy(bool) if not pre else np.zeros(len(d), bool)
        ad = np.abs(d[sh].to_numpy(float) - d[off].to_numpy(float))
        q = d[~gs]
        rows.append({"stratum": nm, "publication": "ALL", "share_rows_aligned": float(gs.mean()) if len(d) else np.nan,
                     "max_abs_SH-OFF_aligned": float(ad[gs].max()) if gs.any() else np.nan,
                     "mean_abs_SH-OFF_elsewhere": float(ad[~gs].mean()) if (~gs).any() else np.nan,
                     "rows_per_seed_not_aligned": len(q) / max(d.split_seed.nunique(), 1),
                     "cells_not_aligned": int(q.cell_id.nunique()), "pubs_not_aligned": ";".join(sorted(q.publication.unique())),
                     "R2_SH_not_aligned": r2(q, sh), "R2_OFF_not_aligned": r2(q, off), "R2_B1_not_aligned": r2(q, "B1_r4"),
                     "label": DIAG + " (read-alignment identity; geometry flag for the BIT pool only)"})
        if not pre:
            for p, g in d.groupby("publication"):
                m = g.geom_same.to_numpy(bool)
                rows.append({"stratum": nm, "publication": p, "share_rows_aligned": float(m.mean()),
                             "rows_per_seed": len(g) / g.split_seed.nunique(), "mean_n_sib": float(g.n_sib.mean()),
                             "mean_off_np": float(g.OFF_np.mean()),
                             "mean_abs_SH-OFF": float(np.abs(g.SH_r4 - g.OFF).mean()), "label": DIAG})
    return pd.DataFrame(rows)


def t_mechanism(R) -> tuple[pd.DataFrame, list, list]:
    """Registered mechanism contrast SH - OFF_XEXT and SH - PB_PUB (and count-matched PB_PUB) on rows where the
    comparator exists, pooled and per publication."""
    rows, inf, units = [], [], []
    for nm in ("PRIMARY", "WITH_C5", "GE1"):
        d0 = stratum(R, nm)
        for pre, sh, lab in (("", "SH_r4", STRATA[nm][2]), ("OFS_", "OFS_SH_r4", OFS_LAB)):
            if sh not in R.columns:
                continue
            variant = "primary (bit rule)" if not pre else "OFS (offset / near-identity removed)"
            ox, pp, cmx = f"{pre}OFF_XEXT", f"{pre}PB_PUB", f"E_{pre}PB_PUB_CM"
            fam = d0[(d0[f"n_{pre}off_xext"] > 0) | (d0[f"n_{pre}PB_PUB"] > 0)]
            n_fam = int(fam[["publication", "extractant"]].drop_duplicates().shape[0])
            n_pub_x = int(fam.publication.nunique())
            both = d0[d0[ox].notna() & d0[pp].notna()]
            for cmp, sub in ((ox, d0), (pp, d0), (cmx, d0), (ox, both), (pp, both)):
                rowset = f"rows where both {ox} and {pp} exist" if sub is both else f"rows where {cmp} exists"
                d = complete(sub, [sh, cmp, "B1_r4"])
                if not len(d):
                    continue
                ns = d.split_seed.nunique()
                sse = {a: float(err2(d, a).sum() / ns) for a in (sh, cmp, "B1_r4")}
                gain = sse["B1_r4"] - sse[sh]
                rows.append({"stratum": nm, "pool_variant": variant, "comparator": cmp, "rows": rowset,
                             "publication": "POOLED", **describe(d),
                             "R2_SH": r2(d, sh), "R2_cmp": r2(d, cmp), "R2_B1": r2(d, "B1_r4"),
                             "dR2_SH-cmp": r2(d, sh) - r2(d, cmp), "relSSE_SH_vs_cmp": relsse(d, sh, cmp),
                             "SSE_SH_per_seed": sse[sh], "SSE_cmp_per_seed": sse[cmp], "SSE_B1_per_seed": sse["B1_r4"],
                             "recovered_share": (sse["B1_r4"] - sse[cmp]) / gain if gain else np.nan,
                             "families_with_other_extractant_pool": n_fam, "publications_with_other_extractant_pool": n_pub_x,
                             "label": lab})
                for p, g in d.groupby("publication"):
                    s = {a: float(err2(g, a).sum() / ns) for a in (sh, cmp, "B1_r4")}
                    gg = s["B1_r4"] - s[sh]
                    rows.append({"stratum": nm, "pool_variant": variant, "comparator": cmp, "rows": rowset,
                                 "publication": p, **describe(g),
                                 "SSE_SH_per_seed": s[sh], "SSE_cmp_per_seed": s[cmp], "SSE_B1_per_seed": s["B1_r4"],
                                 "recovered_share": (s["B1_r4"] - s[cmp]) / gg if gg else np.nan, "label": lab})
                if sub is d0:
                    r, u = infer(d, sh, cmp, stratum_name=nm, label=lab + (" (registered mechanism contrast)" if not pre
                                                                           else " (mechanism contrast)"))
                    inf.append(r)
                    units.append(u)
    return pd.DataFrame(rows), inf, units


CONTRASTS = [("SH_r4", "B1_r4"), ("SH_r4", "OFF"), ("SH_r4", "OFF_ALL"), ("OFF", "B1_r4"), ("SIB_r4", "B1_r4"),
             ("COPY", "B1_r4"), ("SH0", "B1_r4"), ("SH_r3", "B1_r4"), ("SH_r2", "B1_r4"), ("SH_r2", "B1_r2"),
             ("SH_r3", "B1_r3"), ("SH_r4", "NEAR_r4"), ("SH_r4", "NEARSH_r4"), ("SH_r4", "SIB_r4"),
             ("SH_r4", "E_RAND1_sh_r4"), ("SIB_r2", "SIB_r4"), ("SIB_r2", "COPY"), ("SIB_r2", "B1_r4")]


def t_inference(R) -> tuple[list, list]:
    inf, units = [], []
    for nm in ("PRIMARY", "WITH_C5", "GE1", "GE1_WITH_C5", "EQ1", "PRIMARY_NOBIT", "WITH_C5_NOBIT", "V1_GE2_NOC5"):
        _, pre, lab = STRATA[nm]
        d = stratum(R, nm)
        for c, rf in CONTRASTS:
            c2, r2_ = arm(pre, c, R.columns), arm(pre, rf, R.columns)
            if pre and not (pool_ok(pre, c2) and pool_ok(pre, r2_) and (c2.startswith(pre) or r2_.startswith(pre))):
                continue
            if c2 not in R.columns or r2_ not in R.columns:
                continue
            row, u = infer(d, c2, r2_, stratum_name=nm, label=lab)
            inf.append(row)
            if (c, rf) in (("SH_r4", "B1_r4"), ("SH_r4", "OFF")):
                units.append(u)
    return inf, units


def t_influence(R) -> pd.DataFrame:
    rows = []
    d = complete(stratum(R, "PRIMARY"), ["SH_r4", "B1_r4", "OFF"])
    for cand, ref, pub in (("SH_r4", "B1_r4", E.PUB_ACIDTYPE), ("SH_r4", "OFF", E.PUB_ORNL), ("SH_r4", "B1_r4", E.PUB_ORNL),
                           ("SH_r4", "OFF", E.PUB_ACIDTYPE)):
        u = unit_contrib(d, cand, ref)
        tot = float(u.c.sum())
        sh = float(u[u.publication == pub].c.sum())
        dd = d[d.publication != pub]
        rows.append({"stratum": "PRIMARY", "contrast": f"{cand} - {ref}", "influence_of": pub, "dR2_all": r2(d, cand) - r2(d, ref),
                     "dR2_without": r2(dd, cand) - r2(dd, ref), "share_of_contrast": sh / tot if tot else np.nan,
                     "label": SEEN})
    w = complete(stratum(R, "WITH_C5"), ["SH_r4", "B1_r4", "OFF"])
    ww = w[w.cell_id != paths.C5_CELL]
    for ref in ("B1_r4", "OFF"):
        rows.append({"stratum": "WITH_C5", "contrast": f"SH_r4 - {ref}", "influence_of": paths.C5_CELL,
                     "cell_in_stratum": bool((w.cell_id == paths.C5_CELL).any()), "dR2_all": r2(w, "SH_r4") - r2(w, ref),
                     "dR2_without": r2(ww, "SH_r4") - r2(ww, ref), "label": SENS})
    return pd.DataFrame(rows)


def t_dupsens(R) -> pd.DataFrame:
    rows = []
    specs = [("PRIMARY (CLEAN + bit rule; primary)", "in_PRIMARY", "", SEEN),
             ("PRIMARY_NOBIT (P1's CLEAN; items 1 and 3 kept)", "in_PRIMARY_NOBIT", "NB_", SENS),
             ("PRIMARY_I1 (CLEAN minus the item-1 link only)", "in_PRIMARY_I1", "I1_", SENS),
             ("PRIMARY_SD (bit rule + shape-sd < 0.02 links dropped; label-based)", "in_PRIMARY_SD", "SD_", SENS),
             ("PRIMARY_OFS (bit rule + offset / near-identity links dropped)", "in_PRIMARY_OFS", "OFS_", SENS),
             ("PRIMARY_QNT (bit rule + quantised cells dropped as targets and siblings)", "in_PRIMARY_QNT", "QNT_", SENS),
             ("WITH_C5_NOBIT", "in_WITH_C5_NOBIT", "NB_", SENS), ("WITH_C5", "in_WITH_C5", "", SENS)]
    for nm, flag, pre, lab in specs:
        d = R[R[flag]]
        sh = pre + "SH_r4" if pre else "SH_r4"
        row, _ = infer(d, sh, "B1_r4", stratum_name=nm, label=lab)
        r = {k: row.get(k) for k in ("stratum", "cand", "ref", "cells", "publications", "units", "rows_per_seed", "R2_cand",
                                     "R2_ref", "dR2", "relSSE", "lopo_lo", "lopo_hi", "lopo_min", "lopo_min_without",
                                     "units_pos", "units_neg", "label")}
        off = pre + "OFF" if pre in ("", "NB_") else None
        if off and off in R.columns:
            dd = complete(d, [sh, off])
            r["SH-OFF"] = r2(dd, sh) - r2(dd, off)
        rows.append(r)
    # duplicate sensitivity of the control pools (s0-items-1, s0-items-2): contrasts read on OFS pools throughout
    d = R[R.in_PRIMARY_OFS]
    for cand, ref in (("OFS_SH_r4", "OFS_OFF_ALL"), ("OFS_SH_r4", "OFS_OFF_XEXT"), ("OFS_SH_r4", "OFS_PB_PUB"),
                      ("OFS_SH_r4", "E_OFS_PB_PUB_CM"), ("OFS_SH_r4", "OFS_PB_XEXT"), ("OFS_SH_r4", "OFS_PB_CHEM"),
                      ("OFS_SH_r4", "OFS_PB_LABX"), ("OFS_SH_r4", "OFS_NEAR_r4"), ("OFS_SH_r4", "OFS_NEARSH_r4")):
        if cand not in R.columns or ref not in R.columns:
            continue
        dd = complete(d, [cand, ref, "B1_r4"])
        if dd.publication.nunique() < 2:
            rows.append({"stratum": "PRIMARY_OFS (control pools)", "cand": cand, "ref": ref, **describe(dd),
                         "note": "fewer than 2 publications: not computed", "label": OFS_LAB})
            continue
        row, _ = infer(dd, cand, ref, stratum_name="PRIMARY_OFS (control pools)", label=OFS_LAB)
        rows.append({k: row.get(k) for k in ("stratum", "cand", "ref", "cells", "publications", "units", "rows_per_seed",
                                             "R2_cand", "R2_ref", "dR2", "relSSE", "lopo_lo", "lopo_hi", "lopo_min",
                                             "lopo_min_without", "units_pos", "units_neg", "label")})
    return pd.DataFrame(rows)


def t_v25(R) -> pd.DataFrame:
    d = R[R.in_PRIMARY_V25 & R.v25_row_ok]
    rows = []
    row, _ = infer(d, "V25_SH_r4", "B1_r4", stratum_name="PRIMARY_V25", label=SENS)
    rows.append({**{k: row.get(k) for k in ("stratum", "cand", "ref", "cells", "publications", "rows_per_seed", "R2_cand",
                                            "R2_ref", "dR2", "relSSE", "lopo_lo", "lopo_hi")}, "label": SENS})
    dd = complete(d[d.in_PRIMARY], ["SH_r4", "V25_SH_r4", "B1_r4"])
    rows.append({"stratum": "PRIMARY_V25 rows inside PRIMARY", "cand": "V25_SH_r4", "ref": "SH_r4", **describe(dd),
                 "R2_cand": r2(dd, "V25_SH_r4"), "R2_ref": r2(dd, "SH_r4"), "dR2": r2(dd, "V25_SH_r4") - r2(dd, "SH_r4"),
                 "relSSE": relsse(dd, "V25_SH_r4", "SH_r4"), "label": SENS})
    P = stratum(R, "PRIMARY")
    t = P.drop_duplicates("cell_id")
    rows.append({"stratum": "counts", "targets_with_changed_pool": int((t.n_V25 != t.n_sib).sum()),
                 "V2-5_targets_in_PRIMARY": int((t.t_floor.astype(str) != "").sum()),
                 "PRIMARY_rows_dropped_per_seed": (len(P) - int((P.in_PRIMARY_V25 & P.v25_row_ok).sum())) / P.split_seed.nunique(),
                 "label": SENS})
    # C5-BTBP in (brief 5.2): PRIMARY_V25 excludes C5; check that V2-5 leaves the C5 rows of WITH_C5 unchanged
    C = R[R.in_WITH_C5 & R.c5]
    tc = C.drop_duplicates("cell_id")
    v, s = C.V25_SH_r4.to_numpy(float), C.SH_r4.to_numpy(float)
    same = (bool(len(C)) and bool((tc.n_V25 == tc.n_sib).all()) and bool(C.v25_row_ok.all())
            and np.array_equal(v, s, equal_nan=True))
    rows.append({"stratum": "C5-BTBP rows of WITH_C5 (V2-5 check)", "cand": "V25_SH_r4", "ref": "SH_r4",
                 "cells": int(C.cell_id.nunique()), "rows_per_seed": len(C) / max(C.split_seed.nunique(), 1),
                 "targets_with_changed_pool": int((tc.n_V25 != tc.n_sib).sum()),
                 "V2-5_targets_in_C5": int((tc.t_floor.astype(str) != "").sum()),
                 "C5_rows_dropped_per_seed": (len(C) - int(C.v25_row_ok.sum())) / max(C.split_seed.nunique(), 1),
                 "max_abs_V25_SH_r4-SH_r4": float(np.nanmax(np.abs(v - s))) if np.isfinite(v - s).any() else np.nan,
                 "note": ("V2-5 leaves C5-BTBP unchanged (no pool changed, no row dropped, V25_SH_r4 = SH_r4 exactly on "
                          "every C5 row): with C5 in, the V2-5 stratum is PRIMARY_V25 plus these rows unchanged") if same
                 else "V2-5 changes C5-BTBP rows (see the counts)", "label": SENS})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
# descriptive annex rows (each marked PLANNED / EXCLUDED in REGISTRY.csv at the Stage-0 seal)
# ------------------------------------------------------------------------------------------------------------
SHIFT_ARMS = ["ACOPY_r0", "COPY", "AINT_r0", "SH_r2", "SH_r4", "B1_r4", "OFF"]


def t_shift1(R) -> pd.DataFrame:
    """shift-1 row: ACOPY@0, COPY@0, AINT@0, SH@2, SH@4, B1@4, OFF by |dlog10 acid| bin x interior / extrapolation x
    below / above, with and without pub_19783c039b and C5-BTBP; both distance metrics (dx_mean, dx_near)."""
    A = complete(R[R.in_GE1_WITH_C5 & (R.n_acid >= 1)], SHIFT_ARMS)
    if not len(A):
        return pd.DataFrame()
    # acid position (label-free metadata): recomputed from the target table columns the builder joined
    A = A.copy()
    rows = []
    subsets = {"all": A, "without C5": A[~A.c5], f"without {E.PUB_ORNL}": A[A.publication != E.PUB_ORNL],
               "without C5 and ORNL": A[~A.c5 & (A.publication != E.PUB_ORNL)]}
    for sname, S in subsets.items():
        for metric in ("dx_mean_abs", "dx_near"):
            bins = pd.cut(S[metric], DX_BINS[0], labels=DX_BINS[1])
            for keys, d in S.groupby([bins, S.acid_position], observed=True):
                rows.append(_shift_row(d, sname, metric, str(keys[0]), keys[1]))
            for b, d in S.groupby(bins, observed=True):
                rows.append(_shift_row(d, sname, metric, str(b), "any"))
        for pos_, d in S.groupby("acid_position"):
            rows.append(_shift_row(d, sname, "any", "any", pos_))
        rows.append(_shift_row(S, sname, "any", "any", "any"))
    # duplicate sensitivity (s0-items-2): ACOPY / AINT on the acid pool minus offset / near-identity links
    if "OFS_ACOPY_r0" in R.columns:
        O = complete(R[R.in_GE1_WITH_C5 & (R.n_OFS_acid >= 1)], OFS_SHIFT_ARMS)
        for sname, S in {"all": O, "without C5": O[~O.c5]}.items():
            for pos_, d in S.groupby("acid_position"):
                rows.append(_shift_row(d, sname, "any", "any", pos_, arms=OFS_SHIFT_ARMS, label=OFS_LAB))
            rows.append(_shift_row(S, sname, "any", "any", "any", arms=OFS_SHIFT_ARMS, label=OFS_LAB))
    return pd.DataFrame(rows)


OFS_SHIFT_ARMS = ["OFS_ACOPY_r0", "OFS_COPY", "OFS_AINT_r0", "OFS_SH_r4", "B1_r4"]


def _shift_row(d, sname, metric, b, pos_, *, arms=None, label=None) -> dict:
    out = {"subset": sname, "distance_metric": metric, "bin_decades": b, "position": pos_, **describe(d)}
    arms = SHIFT_ARMS if arms is None else arms
    for a in arms:
        out[f"R2_{a}"] = r2(d, a)
        out[f"MSE_{a}"] = mse(d, a)
    out["label"] = (label if label is not None else SEEN) + " (descriptive annex: shift-1)"
    if arms is not SHIFT_ARMS:
        out["pool_variant"] = "OFS (offset / near-identity removed)"
    return out


EREC_CORE = ["COPY", "SH_r2", "SH_r3", "SH_r4"]
EREC_NP = {"NP_LAD": ["NP_LAD_r2", "NP_LAD_r3", "NP_LAD_r4", "NP_LAD_r5"],
           "NP_DOPTQ": ["B1_r2", "B1_r3", "B1_r4", "NP_DOPTQ_r5"]}


def t_erec(R) -> pd.DataFrame:
    """recipe-1's E-REC table: credit of an own prior series by sibling count against the frozen best label-free
    no-prior policy (NP_LAD: recipe-2 ladder, legal-HULL fallback; NP_DOPTQ beside it), with and without
    pub_b0b3296ec5, on GE1 (without C5) and on GE1_WITH_C5 (C5-BTBP in, brief 5.2).  The 5-read rung is a different
    read budget."""
    rows = []
    for pop, flag, lab in (("GE1 (without C5)", "in_GE1", SEEN), ("GE1_WITH_C5 (with C5)", "in_GE1_WITH_C5", SENS)):
        G = R[R[flag]]
        if not len(G):
            continue
        first = G[G.split_seed == G.split_seed.min()]
        giant = first.groupby("cluster_cpg").size().idxmax()
        strata = {"n_own = 1": G[G.n_sib == 1], "n_own >= 2": G[G.n_sib >= 2], f"giant cluster ({giant})": G[G.cluster_cpg == giant],
                  "the rest": G[G.cluster_cpg != giant]}
        for sname, S0 in strata.items():
            for wname, S in (("with " + E.PUB_ACIDTYPE, S0), ("without " + E.PUB_ACIDTYPE, S0[S0.publication != E.PUB_ACIDTYPE])):
                for version, arms in (("rows with a legal 5th read (both policies)", EREC_CORE + EREC_NP["NP_LAD"] + EREC_NP["NP_DOPTQ"]),
                                      ("all rows, 2-4 reads", EREC_CORE + EREC_NP["NP_LAD"][:3] + EREC_NP["NP_DOPTQ"][:3])):
                    d = complete(S, arms)
                    r = {"population": pop, "stratum": sname, "pub_b0b3296ec5": wname, "rows": version, **describe(d)}
                    for a in arms:
                        r[f"R2_{a}"] = r2(d, a)
                    sh2 = r2(d, "SH_r2")
                    for pol, cols in EREC_NP.items():
                        cols = [c for c in cols if c in arms]
                        vals = [r2(d, c) for c in cols]
                        m = next((rr for rr, v in zip(range(2, 2 + len(cols)), vals) if v >= sh2), None)
                        r[f"reads_equivalent_SH@2_vs_{pol}"] = m if m is not None else f">{1 + len(cols)}"
                        r[f"SH@2-{pol}@4"] = sh2 - vals[2] if len(vals) > 2 else np.nan
                    for mm, col in ((1, "E_RAND1_sh_r2"), (2, "E_RAND2_sh_r2")):
                        dm = complete(d, [col])           # a subset of d: identical rows inside the row (brief 6.0)
                        r[f"R2_SH@2_m{mm}_random_own"] = r2(dm, col)
                        r[f"R2_SH_r2_on_m{mm}_rows"] = r2(dm, "SH_r2")
                        for pol, cols in EREC_NP.items():
                            for rr, c in zip((2, 3, 4), [c for c in cols if c in arms][:3]):
                                r[f"R2_{pol}@{rr}_on_m{mm}_rows"] = r2(dm, c)
                        r[f"rows_per_seed_m{mm}"] = describe(dm)["rows_per_seed"]
                    r["frozen_no_prior_policy"] = "NP_LAD (recipe-2 ladder La,Ho / La,Nd,Ho / La,Nd,Er,Lu / La,Nd,Gd,Er,Lu; legal HULL fallback; B1 form, fold beta)"
                    r["label"] = lab + " (descriptive annex: recipe-1 E-REC; 5 reads = a different read budget)"
                    rows.append(r)
    return pd.DataFrame(rows)


def t_m02(inf: list) -> pd.DataFrame:
    keep = {("SIB_r2", "SIB_r4"), ("SIB_r2", "COPY"), ("SIB_r2", "B1_r4")}
    rows = [dict(r, annex="M02 ladder row (SIB@2 vs SIB@4 and COPY)") for r in inf
            if (r.get("cand"), r.get("ref")) in keep and r.get("stratum") in ("PRIMARY", "WITH_C5", "GE1")]
    return pd.DataFrame(rows)


def t_m04(R) -> pd.DataFrame:
    """M04 line: 'avoid floor or saturated reference conditions; average at least 2' (descriptive)."""
    rows = []
    for nm in ("PRIMARY", "GE1", "WITH_C5"):
        lab = STRATA[nm][2] + " (descriptive annex: M04)"
        d0 = stratum(R, nm)
        d = complete(d0[(d0.n_INWIN >= 1) & (d0.n_OUTWIN >= 1)], ["B1_r4", "SH_r4", "INWIN_SH_r4", "OUTWIN_SH_r4"])
        rows.append({"stratum": nm, "line": "in-window vs floor / saturated references (rows with both)", **describe(d),
                     **{f"R2_{a}": r2(d, a) for a in ("B1_r4", "SH_r4", "INWIN_SH_r4", "OUTWIN_SH_r4")},
                     "relSSE_INWIN_vs_OUTWIN": relsse(d, "INWIN_SH_r4", "OUTWIN_SH_r4"),
                     "relSSE_INWIN_vs_SH": relsse(d, "INWIN_SH_r4", "SH_r4"), "label": lab})
        d = complete(d0[d0.n_sib >= 2], ["SH_r4", "E_RAND1_sh_r4", "E_RAND2_sh_r4", "E_RAND1_raw_r4", "E_RAND2_raw_r4"])
        rows.append({"stratum": nm, "line": "average at least 2 (one vs two random own siblings, 4 reads)", **describe(d),
                     **{f"R2_{a}": r2(d, a) for a in ("SH_r4", "E_RAND1_sh_r4", "E_RAND2_sh_r4", "E_RAND1_raw_r4", "E_RAND2_raw_r4")},
                     "relSSE_RAND2sh_vs_RAND1sh": relsse(d, "E_RAND2_sh_r4", "E_RAND1_sh_r4"), "label": lab})
    return pd.DataFrame(rows)


def t_m06(R) -> pd.DataFrame:
    """M06 annex: same campaign (other extractants, same publication) against other publications; the homologue
    channel (PB_PUBHOM) is recorded as a frozen lead with an external test only."""
    rows = []
    for nm in ("PRIMARY", "WITH_C5", "GE1"):
        lab = STRATA[nm][2] + " (descriptive annex: M06)"
        d0 = stratum(R, nm)
        for desc, arms in (("same campaign vs other publication (rows with PB_PUB and PB_XEXT)", ["B1_r4", "SH_r4", "PB_PUB", "PB_XEXT"]),
                           ("same campaign vs same chemotype elsewhere (rows with PB_PUB and PB_CHEM)", ["B1_r4", "SH_r4", "PB_PUB", "PB_CHEM"]),
                           ("homologue (same chemotype) vs other chemotype, same publication", ["B1_r4", "SH_r4", "PB_PUBHOM", "PB_PUBHET"]),
                           ("homologue channel alone (rows with PB_PUBHOM)", ["B1_r4", "SH_r4", "PB_PUBHOM", "PB_PUB"])):
            d = complete(d0, arms)
            r = {"stratum": nm, "comparison": desc, **describe(d)}
            for a in arms:
                r[f"R2_{a}"] = r2(d, a)
                r[f"relSSE_{a}_vs_B1"] = relsse(d, a, "B1_r4")
            r["homologue_channel"] = "frozen lead; external test only (IR M06 salvage)"
            r["label"] = lab
            rows.append(r)
        if "OFS_SH_r4" not in R.columns:
            continue
        for desc, arms in (("same campaign vs other publication, OFS pools", ["B1_r4", "OFS_SH_r4", "OFS_PB_PUB", "OFS_PB_XEXT"]),
                           ("same campaign vs same chemotype elsewhere, OFS pools", ["B1_r4", "OFS_SH_r4", "OFS_PB_PUB", "OFS_PB_CHEM"]),
                           ("homologue vs other chemotype, same publication, OFS pools",
                            ["B1_r4", "OFS_SH_r4", "OFS_PB_PUBHOM", "OFS_PB_PUBHET"]),
                           ("homologue channel alone, OFS pools", ["B1_r4", "OFS_SH_r4", "OFS_PB_PUBHOM", "OFS_PB_PUB"])):
            d = complete(d0, arms)
            r = {"stratum": nm, "pool_variant": "OFS (offset / near-identity removed)", "comparison": desc, **describe(d)}
            for a in arms:
                r[f"R2_{a}"] = r2(d, a)
                r[f"relSSE_{a}_vs_B1"] = relsse(d, a, "B1_r4")
            r["homologue_channel"] = "frozen lead; external test only (IR M06 salvage)"
            r["label"] = OFS_LAB + " (descriptive annex: M06)"
            rows.append(r)
    return pd.DataFrame(rows)


def t_lab(R) -> tuple[pd.DataFrame, list]:
    """Laboratory annotation (frozen PROVISIONAL key): laboratory counts, laboratory-level contributions to SH - B1,
    and same-laboratory other-publication templates (PB_LABX) where they exist."""
    rows, inf = [], []
    for nm in ("PRIMARY", "WITH_C5", "GE1"):
        lab = STRATA[nm][2] + " (laboratory key PROVISIONAL, U6 review pending)"
        d = complete(stratum(R, nm), ["SH_r4", "B1_r4"])
        if "lab_cons" not in d.columns or (d.lab_cons.astype(str) == "NA").all():
            rows.append({"stratum": nm, "status": "NOT RUN: frozen laboratory key missing", "label": lab})
            continue
        dd = d.copy()
        dd["lab_chem"] = dd.lab_cons.astype(str) + "|" + dd.chemotype.astype(str)
        u = unit_contrib(dd, "SH_r4", "B1_r4", unit=("lab_cons",))
        uc = unit_contrib(dd, "SH_r4", "B1_r4", unit=("lab_cons", "chemotype"))
        sf = signflip_p(uc.c.to_numpy(float), key=f"ECT|LAB|{nm}")
        rows.append({"stratum": nm, "status": "PROVISIONAL", "labs_conservative": int(dd.lab_cons.nunique()),
                     "labs_liberal": int(dd.lab_lib.nunique()), "lab_x_chemotype_units": int(len(uc)),
                     "publications": int(dd.publication.nunique()), "dR2_SH-B1": r2(dd, "SH_r4") - r2(dd, "B1_r4"),
                     "labs_pos": int((u.c > 1e-6).sum()), "labs_neg": int((u.c < -1e-6).sum()),
                     "p_signflip_lab_x_chemotype": sf["p"], "top3_lab_share": CM.top3_share(u.c.to_numpy(float)),
                     "wording": "same publication reads 'same study or campaign' until the key is reviewed", "label": lab})
        for r in u.itertuples(index=False):
            rows.append({"stratum": nm, "status": "per laboratory", "lab_cons": r.lab_cons, "c_SH-B1": r.c,
                         "rows_per_seed": r.rows, "rho": r.rho, "label": lab})
        dl = complete(stratum(R, nm), ["SH_r4", "B1_r4", "PB_LABX", "PB_LABX_raw"])
        if len(dl):
            rows.append({"stratum": nm, "status": "same laboratory, other publication (PB_LABX rows)", **describe(dl),
                         **{f"R2_{a}": r2(dl, a) for a in ("B1_r4", "SH_r4", "PB_LABX", "PB_LABX_raw")}, "label": lab})
            if dl.publication.nunique() >= 2:
                r_, _ = infer(dl, "SH_r4", "PB_LABX", stratum_name=nm, label=lab)
                inf.append(r_)
        if "OFS_PB_LABX" in R.columns:                   # duplicate sensitivity (s0-items-1)
            do = complete(stratum(R, nm), ["OFS_SH_r4", "B1_r4", "OFS_PB_LABX", "OFS_PB_LABX_raw"])
            if len(do):
                rows.append({"stratum": nm, "status": "same laboratory, other publication (OFS_PB_LABX rows)",
                             "pool_variant": "OFS (offset / near-identity removed)", **describe(do),
                             **{f"R2_{a}": r2(do, a) for a in ("B1_r4", "OFS_SH_r4", "OFS_PB_LABX", "OFS_PB_LABX_raw")},
                             "label": OFS_LAB})
                if do.publication.nunique() >= 2:
                    r_, _ = infer(do, "OFS_SH_r4", "OFS_PB_LABX", stratum_name=nm, label=OFS_LAB)
                    inf.append(r_)
    return pd.DataFrame(rows), inf


# ------------------------------------------------------------------------------------------------------------
# report rules (in-corpus consistency checks; no kill fires in-corpus)
# ------------------------------------------------------------------------------------------------------------
def _find(inf, stratum_name, cand, ref):
    for r in inf:
        if r.get("stratum") == stratum_name and r.get("cand") == cand and r.get("ref") == ref:
            return r
    return {}


def _mech_shares(mech: pd.DataFrame, cmp: str) -> pd.DataFrame:
    """Pooled and per-publication rows of one registered mechanism contrast (PRIMARY, bit-rule pools, the
    comparator's own rows)."""
    return mech[(mech.stratum == "PRIMARY") & (mech.comparator == cmp) & mech.rows.str.startswith(f"rows where {cmp} ")
                & (mech.pool_variant == "primary (bit rule)")]


def _shares_txt(m: pd.DataFrame) -> str:
    return "; ".join(f"{'pooled' if q.publication == 'POOLED' else q.publication} {q.recovered_share:.3f}"
                     for q in m.itertuples(index=False))


def t_rules(R, inf, excl: pd.DataFrame, mech: pd.DataFrame, dup: pd.DataFrame) -> pd.DataFrame:
    """The section-8 report rules.  1.1-R01 (SH - B1 on PRIMARY, no threshold) carries the label of its rows
    (EXPLORATORY (seen); its NOBIT / I1 rows SENSITIVITY, as DECISION_LOG section 7); the threshold rules are
    CONSISTENCY CHECK."""
    rows, labels = [], []
    p = _find(inf, "PRIMARY", "SH_r4", "B1_r4")
    rows.append({"rule": "SH - B1 on PRIMARY with its LOPO interval", "value": p.get("dR2"),
                 "interval": f"[{p.get('lopo_lo', np.nan):+.4f}, {p.get('lopo_hi', np.nan):+.4f}]",
                 "units_up": f"{p.get('units_pos')} of {p.get('G_units')}", "threshold": "none (report rule)",
                 "outcome": "reported", "comparator_note": f"B1 on the same rows = {p.get('R2_ref', np.nan):.4f}; {NOT_NEXT_TO}"})
    labels.append(SEEN)
    for _, r in dup.iterrows():
        if r["stratum"].startswith(("PRIMARY_NOBIT", "PRIMARY_I1")):
            rows.append({"rule": f"SH - B1 duplicate sensitivity: {r['stratum']}", "value": r["dR2"],
                         "interval": f"[{r['lopo_lo']:+.4f}, {r['lopo_hi']:+.4f}]",
                         "units_up": f"{r['units_pos']:.0f} of {r['units']:.0f}", "threshold": "none", "outcome": "reported"})
            labels.append(SENS)
    h = _find(inf, "EQ1", "SH_r4", "B1_r4")
    v = h.get("relSSE", np.nan)
    rows.append({"rule": "harm rule (ii): at n_sib = 1, SH not worse than B1 by more than 5 % relative SSE", "value": v,
                 "threshold": ">= -0.05", "cells": h.get("cells"), "publications": h.get("publications"),
                 "outcome": ("passes" if v >= -0.05 else "fails: the deployed rule depends on sibling count") if np.isfinite(v) else "not computed"})
    ce = excl[(excl.stratum == "PRIMARY") & (excl.rows == "all pairs")]
    cs = float(ce["Ce_share_of_SH-B1_gain"].iloc[0]) if len(ce) else np.nan
    rows.append({"rule": "artefact flag (iii): >= 50 % of the SH - B1 gain on Ce pairs", "value": cs, "threshold": ">= 0.50",
                 "Gd_share": float(ce["Gd_share_of_SH-B1_gain"].iloc[0]) if len(ce) else np.nan,
                 "outcome": ("triggered: reword as 'transfer of a laboratory fingerprint'" if cs >= 0.5 else "not triggered")
                 if np.isfinite(cs) else "not computed"})
    mx, mb = _mech_shares(mech, "OFF_XEXT"), _mech_shares(mech, "PB_PUB")
    mp, pp = mx[mx.publication == "POOLED"], mb[mb.publication == "POOLED"]
    nf = int(mp.families_with_other_extractant_pool.iloc[0]) if len(mp) else 0
    npub = int(mp.publications_with_other_extractant_pool.iloc[0]) if len(mp) else 0
    rows.append({"rule": "mechanism wording (SH - OFF_XEXT, SH - PB_PUB per publication)",
                 "value": float(mp["dR2_SH-cmp"].iloc[0]) if len(mp) else np.nan,
                 "families_with_other_extractant_pool": nf,
                 "SH-PB_PUB": float(pp["dR2_SH-cmp"].iloc[0]) if len(pp) else np.nan,
                 "recovered_share_OFF_XEXT": _shares_txt(mx), "recovered_share_PB_PUB": _shares_txt(mb),
                 "families": f"{nf} families (publication x extractant) in {npub} publications",
                 "threshold": ">= 4 families with an other-extractant pool AND a threshold set before data (none registered in-corpus)",
                 "outcome": "default wording: 'same-campaign information (extractant transfer and campaign fingerprint not separated)'"})
    for cand, ref, txt in (("SH_r4", "SIB_r4", "SH vs SIB"), ("SH_r4", "NEAR_r4", "SH vs NEAR")):
        r = _find(inf, "PRIMARY", cand, ref)
        rows.append({"rule": f"claim wording: {txt} (units W/L/T, SIGN p)", "value": r.get("dR2"),
                     "units": r.get("SIGN_W/L/T"), "SIGN_p": r.get("SIGN_p"), "threshold": "fixed wording",
                     "outcome": "claim reads 'own-series template + 3-4 reads', not 'K = 1.78'"})
    r = _find(inf, "PRIMARY", "SH_r2", "B1_r4")
    rows.append({"rule": "SH with 2 reads (lead; excluded from claims, section 4.3 item 4)", "value": r.get("dR2"),
                 "units": f"{r.get('units_pos')} of {r.get('G_units')}", "top3_share": r.get("top3_share"),
                 "threshold": "fixed", "outcome": "stays excluded from claims"})
    labels += [CC] * (len(rows) - len(labels))          # the harm, artefact, mechanism, claim and 2-read rules
    out = pd.DataFrame(rows)
    out["label"] = labels
    return out


# ------------------------------------------------------------------------------------------------------------
# everything for one design, the per-item table, the reproduction targets, the smoke surrogate
# ------------------------------------------------------------------------------------------------------------
def all_tables(R: pd.DataFrame, design: str) -> dict:
    T = {}
    T["populations"] = t_populations(R)
    T["comparators"] = t_comparators(R)
    T["placebos"], inf_pb, u_pb = t_placebos(R)
    T["ladder"] = t_ladder(R)
    T["subm"] = t_subm(R)
    T["nsib"] = t_nsib(R)
    T["pairtype"], T["pairexcl"] = t_pairtype(R)
    T["condclass"] = t_condclass(R)
    T["alignment"] = t_alignment(R)
    T["mechanism"], inf_m, u_m = t_mechanism(R)
    inf, u_i = t_inference(R)
    T["influence"] = t_influence(R)
    T["dupsens"] = t_dupsens(R)
    T["v25"] = t_v25(R)
    T["annex_shift1"] = t_shift1(R)
    T["annex_erec"] = t_erec(R)
    T["annex_m04"] = t_m04(R)
    T["annex_m06"] = t_m06(R)
    T["lab"], inf_l = t_lab(R)
    allinf = inf + inf_pb + inf_m + inf_l
    T["annex_m02"] = t_m02(allinf)
    T["inference"] = pd.DataFrame(allinf)
    T["units"] = pd.concat([u for u in u_i + u_pb + u_m if len(u)], ignore_index=True) if any(len(u) for u in u_i + u_pb + u_m) else pd.DataFrame()
    T["rules"] = t_rules(R, allinf, T["pairexcl"], T["mechanism"], T["dupsens"])
    for k, v in T.items():
        if len(v):
            v.insert(0, "design", design)
    return T


def fmt_p(v) -> str:
    """A p-value for the per-item table: scientific notation below 0.001 (r5.common prints every float with .4f)."""
    v = float(v)
    return "NaN" if not np.isfinite(v) else (f"{v:.2e}" if v < 1e-3 else f"{v:.4f}")


def unit_tests_txt(p) -> str:
    """The section-6.1 descriptive tests of one inference row, for the 'Unit W/L/T and the descriptive tests' field."""
    return (f"{p['cand']} - {p['ref']} on {p['stratum']}, {p['G_units']:.0f} publication x chemotype units: SIGN W/L/T "
            f"{p['SIGN_W/L/T']}, SIGN p {fmt_p(p['SIGN_p'])}; IM-t on the per-unit relative SSE t {p['IMt_rho_t']:.3f} "
            f"(p {fmt_p(p['IMt_rho_p'])}), on the log SSE ratio t {p['IMt_log_t']:.3f} (p {fmt_p(p['IMt_log_p'])}); "
            f"equal-weight per-publication relative SSE {p['eqw_pub_relSSE']:.4f} ({p['P']:.0f} publications)")


def per_item(T: dict, design: str, *, seal_tag: str, leakage: str, seconds: float) -> dict:
    inf = T["inference"]
    p = inf[(inf.stratum == "PRIMARY") & (inf.cand == "SH_r4") & (inf.ref == "B1_r4")].iloc[0]
    w = inf[(inf.stratum == "WITH_C5") & (inf.cand == "SH_r4") & (inf.ref == "B1_r4")].iloc[0]
    nb = inf[(inf.stratum == "PRIMARY_NOBIT") & (inf.cand == "NB_SH_r4") & (inf.ref == "B1_r4")]
    comp = T["comparators"]
    met = comp[comp.stratum == "PRIMARY"][["arm", "R2", "MSE", "relSSE_vs_B1_r4", "R2_per_seed", "rows_per_seed", "label"]]
    un = T["units"]
    un = un[(un.stratum == "PRIMARY") & (un.contrast == "SH_r4 - B1_r4")][["publication", "chemotype", "c", "rows", "rho", "logratio"]] \
        if len(un) else pd.DataFrame()
    nb_txt = (f" (reproduced here without the bit rule: PRIMARY_NOBIT {nb.dR2.iloc[0]:+.4f}, {nb.units_pos.iloc[0]:.0f} of "
              f"{nb.G_units.iloc[0]:.0f} units up)") if len(nb) else ""
    return {
        "item": "1.1 E-CT core: condition transfer (M01; P1; PC P1)",
        "population": {k: p[k] for k in ("cells", "publications", "units", "rows_per_seed", "cpg_clusters", "kish_cpg")},
        "design": design, "read_budget": "4 reads (code k = 3) for every scored row; ladder rungs 0 / 2 / 3 reads on the same rows; NP 5 reads = a different read budget",
        "arms": "B1_r2-4, B0_r2-4, SIB_r2-4, SH_r2-4 (K = 1.78), SHK2, COPY, SH0, OFF, OFF_ALL, OFF_PUB, OFF_XEXT, NEAR, NEARSH, RAND1, RAND2, SUB m, PB_CHEM/XEXT/PUB (+ CM), PB_LABX, PB_PUBHOM/HET, ACOPY, AINT, NP_LAD, NP_DOPTQ_r5; variants NB_, V1_, I1_, SD_, OFS_, QNT_, V25_, INWIN_, OUTWIN_",
        "metrics": met,
        "per_seed": f"SH_r4 - B1_r4 per seed (PRIMARY): {p['dR2_per_seed']}",
        "units": un if len(un) else unit_tests_txt(p),
        "units_tests": unit_tests_txt(p),            # printed in the field row beside the unit sub-table (per_item_table)
        "lopo_top3": {"dR2": p["dR2"], "lopo_lo": p["lopo_lo"], "lopo_hi": p["lopo_hi"], "lopo_min": p["lopo_min"],
                      "lopo_min_without": p["lopo_min_without"], "top3_share": p["top3_share"],
                      "p_signflip_unit": fmt_p(p["p_signflip_unit"]),
                      "p_signflip_chemotype_blocks": fmt_p(p["p_signflip_chemotype_blocks"])},
        "c5": {"C5 out (PRIMARY) SH-B1": p["dR2"], "C5 in (WITH_C5) SH-B1": w["dR2"]},
        "duplicates": T["dupsens"][["stratum", "cand", "ref", "dR2", "lopo_lo", "lopo_hi", "rows_per_seed", "label"]],
        "leakage": leakage, "compute": f"{seconds:.0f} s (build + intervention + tables); seal {seal_tag}",
        "label": f"{SEEN}; {NOT_NEXT_TO}",
        "changed": (f"With the frozen section-5.1 bit-identity rule, PRIMARY SH_r4 - B1_r4 is {p['dR2']:+.4f} (LOPO "
                    f"[{p['lopo_lo']:+.4f}, {p['lopo_hi']:+.4f}]) with {p['units_pos']:.0f} of {p['G_units']:.0f} units up, "
                    f"against P1's seen {design} {P1_SEEN[design][0]:+.3f}{P1_SEEN[design][1]}{nb_txt}; SH and its "
                    "comparators are now "
                    "sealed code (r5/ect.py) that runs unchanged on new data, and every in-corpus number stays "
                    "EXPLORATORY (seen), nothing confirmed in-corpus."),
        "next_step": "C-ECT on new laboratories (section 7.5; data-gated: U3 dry run on a corpus-linked laboratory, then the ledger).",
    }


def per_item_table(item: dict, path=None, *, title: str | None = None) -> str:
    """r5.common.per_item_table, with ``item['units_tests']`` printed in the 'Unit W/L/T and the descriptive tests'
    field row beside the per-unit sub-table (the common renderer writes only "see ... below" for a sub-table).  Writes
    LF markdown to ``path`` when given and returns the text."""
    txt = CM.per_item_table(item, None, title=title)
    name = dict(CM.ITEM_FIELDS)["units"]
    if isinstance(item["units"], pd.DataFrame) and item.get("units_tests"):
        txt = txt.replace(f"| {name} | see '{name}' below |",
                          f"| {name} | {CM._fmt(item['units_tests'])}; per-unit contributions c in '{name}' below |", 1)
    if path is not None:
        paths.write_text_lf(path, txt)
    return txt


REPRO = {  # section 6.0 reproduction targets of item 1.1 (BPG); values to 4 decimals
    "E6_V1_GE1": {"stratum": "V1_GE1", "counts": (149, None, None, 1111.0),
                  "values": {"V1_SIB_r4": 0.5734, "V1_SHK2_r4": 0.5946, "B1_r4": 0.4489}},
    "P1_PRIMARY_NOBIT": {"stratum": "PRIMARY_NOBIT", "counts": (99, 12, 14, 690.0),
                         "values": {"B1_r4": 0.5309, "NB_SH_r4": 0.7082, "NB_SIB_r4": 0.6987, "NB_OFF": 0.6963}},
}
REPRO_TOL = 5e-5 + 1e-12
B1_TOL = 8.9e-16                 # brief 6.0: B1 from the fold betas within 8.9e-16 (seen 8.88e-16)


def reproduction(R: pd.DataFrame, checks: dict, design: str) -> dict:
    """TECHNICAL SMOKE: the item's section-6.0 reproduction targets.  A mismatch stops the item."""
    out = {"label": SMOKE, "design": design, "b1_maxabs": checks["b1_maxabs"], "b1_ok": checks["b1_maxabs"] <= B1_TOL,
           "y_maxabs": checks["y_maxabs"], "readset_mismatch": checks["readset_mismatch"], "targets": {}}
    ok = out["b1_ok"] and checks["readset_mismatch"] == 0 and checks["y_maxabs"] == 0.0
    for name, spec in REPRO.items():
        d = stratum(R, spec["stratum"])
        arms = list(spec["values"])
        dd = complete(d, arms)
        info = describe(dd)
        got_c = (info["cells"], info["publications"], info["units"], info["rows_per_seed"])
        c_ok = all(e is None or e == g for e, g in zip(spec["counts"], got_c))
        vals = {a: r2(dd, a) for a in arms}
        rec = {"counts_expected": list(spec["counts"]), "counts_found": list(got_c), "counts_ok": bool(c_ok),
               "values_found": vals}
        if design == "BPG":
            v_ok = {a: bool(abs(vals[a] - v) <= REPRO_TOL) for a, v in spec["values"].items()}
            rec.update({"values_expected": spec["values"], "values_ok": v_ok})
            ok = ok and c_ok and all(v_ok.values())
        else:
            rec["note"] = "no BP value target registered (BPG decides); counts must match"
            ok = ok and c_ok
        out["targets"][name] = rec
    out["passed"] = bool(ok)
    return out


def surrogate(R: pd.DataFrame, design: str) -> pd.DataFrame:
    """TECHNICAL SMOKE frame: y and every prediction / expectation column replaced by seeded noise (NaN pattern kept),
    so the table code runs on real row structure without any real error being computed."""
    S = R.copy()
    rng = np.random.default_rng(paths.seed_for(f"ECT|SMOKE_TABLES|{design}"))
    keep = {"dZ", "k", "k_max", "n_metals", "draw", "ia", "ib", "dx_mean_abs", "dx_near", "fold", "split_seed"}
    S["y"] = rng.normal(0.0, 0.2, len(S))
    for c in S.columns:
        if c in keep or c == "y" or c.startswith("n_") or c.endswith("_np") or S[c].dtype.kind != "f":
            continue
        v = S[c].to_numpy(float)
        noise = (np.abs(rng.normal(0.0, 0.05, len(S))) if c.startswith("E_") else S.y.to_numpy() + rng.normal(0.0, 0.15, len(S)))
        S[c] = np.where(np.isfinite(v), noise, np.nan)
    return S
