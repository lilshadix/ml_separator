"""Duplicate and near-copy scans of V1 (CLAUDE_CODE_PROMPT.md section 5.1; I2 rule 15 and its "Duplicate and
provenance candidates" list).

Every scan here reads V1 log D values, so every flag is a sensitivity and a provenance question, never an edit to V1
[IR rule 7].  Only the frozen bit-identity rule changes an analysis pool, and it removes links, not cells.

* ``bit_identity`` (FROZEN, section 5.1): over every unordered pair of the 521 bench cells, within and across
  publications and extractants: shared = metals finite in both; n_ident = shared metals whose V1 log D are equal as
  float64; a link is flagged when n_ident >= 3 and n_ident >= 0.5 * n_shared.
* ``offset_scan`` (sensitivity; items 8, 9 and the near-identity items 2, 5, 6, 7): for pairs sharing >= 6 metals,
  d = log D_a - log D_b on the shared metals in Z order.  OFFSET: the longest run of >= 6 consecutive shared metals with
  sample sd(d) <= 0.03 and |mean d| >= 0.05 (a constant-shift copy on a segment; for equal length the run with the
  smallest sd).  NEAR_ID: >= 6 shared metals, max |d| <= 0.05, not bit-flagged.  ``integer_shift``: the OFFSET mean is
  within 0.005 of a non-zero integer (decade-shift copies such as V2-1 DMDPhPDA copy B).
* ``quant_scan`` (sensitivity; items 3, 9, 10, 11): per cell, from D = 10^logD: the finest decimal step used (q_cell),
  significant digits, the log half-step q_cell / (2 D ln 10) per metal; Q_TIES = >= 3 metals with an identical
  log D; Q_ADJEQ = >= 1 strict-adjacent pair with identical log D; Q_COARSE = q_cell >= 0.1 and median half-step
  >= 0.01 log; Q_SMALLINT = >= 3 strict-adjacent pairs and >= 75 % of their D ratios equal p/q with p, q <= 10
  (relative 1e-6).
* ``near_copy_donors`` (lab-1 rule, label-free: uses only the target's reads; I2 lab-1): a donor is a near-copy of
  target t when it is in the same medium (same acid type, same diluent, |dlog10 acid| <= 0.15) and its log D lies
  within 0.15 of t's at every one of t's 4 read metals.  Copied from
  round5_design_20261023/ideas2/scripts/lab1skepB_nearcopy.py
  (LF SHA-256 1f5434f6c23e7933416c9a91f0a28fd9aef39ac4c7941f4bdb1bedb2c84cb902).  Metal-concentration duplicates
  (operationalised here, no source implements it: same publication and extractant, cond__ differing only in metal
  concentration, >= 3 shared finite metals) are grouped by ``metal_conc_duplicates``; a donor pool keeps one member
  per group (most finite metals, then the smallest cell id).  Pairs that share no metal are split series, reported
  apart.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import population as POP

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/ideas2/scripts/lab1skepB_nearcopy.py":
        "1f5434f6c23e7933416c9a91f0a28fd9aef39ac4c7941f4bdb1bedb2c84cb902",
}

BIT_MIN_IDENT = 3
BIT_MIN_FRAC = 0.5
OFF_MIN_LEN = 6
OFF_SD_MAX = 0.03
OFF_MIN_SHIFT = 0.05
NEARID_MAXABS = 0.05
NC_TOL = 0.15
NC_DACID = 0.15
Q_TIE_MIN = 3
Q_COARSE_STEP = 0.1
Q_COARSE_HALF = 0.01
Q_SMALLINT_FRAC = 0.75
Q_SMALLINT_PQ = 10
LN = POP.LN

# known items of section 5.1 (I2 "Duplicate and provenance candidates"); cell prefixes resolve against the bench
KNOWN_BIT = {
    "item1": ("44012938119033a5", "5169c377777d282b", 13, 14),
    "item3": ("d4c1e23a091d0960", "e4ee526bda3920c2", 3, 6),
    "item4": ("259e68bb986bb484", "26cbed1ed70886fe", 8, 8),
    "b9_pair1": ("247f57741141188a", "62e7cdd6fa003391", 8, 8),
    "b9_pair2": ("5a6a2f3bf13a7d55", "c61ecb085d8df371", 8, 8),
    "b9_pair3": ("5df28757679cae26", "f8ed87d3cf7f4281", 8, 8),
}
KNOWN_NOT_FLAGGED = {  # within-publication links with 2-5 identical metals that the rule must leave alone
    "af971a0b699f29ae~item1a": ("af971a0b699f29ae", "44012938119033a5"),
    "af971a0b699f29ae~item1b": ("af971a0b699f29ae", "5169c377777d282b"),
}
KNOWN_SENS = {  # sensitivity items for the offset / near-id / quant scans.  Forms: "pairs" (cell or prefix pairs);
    # "cell_partner_pubs" (links of a named cell to any cell of the named publications); "pub_pairs" (cross-publication
    # links between two publications); "pubs" (cells whose quantisation flags are reported)
    "item2": {"pairs": [("652810", "440129")], "scan": "NEAR_ID|OFFSET"},
    "item5": {"pairs": [("37b0cdc532627e1d", "fadca46ff2b140c2")], "scan": "NEAR_ID"},
    "item6": {"pairs": [("2c83afeaea6f67e0", "b1d9860b27f6967a")], "scan": "NEAR_ID"},
    "item7": {"pub_pairs": [("pub_af0c74798c", "pub_d68ed8c8cf")],
              "cell_partner_pubs": {"4fcbabe856178b53": ["pub_af0c74798c", "pub_d68ed8c8cf"]}, "scan": "NEAR_ID"},
    "item8": {"cell_partner_pubs": {"1b71b86063cd1faf": ["pub_6a0ef8528c"], "354fe634113a078e": ["pub_723ceb6c2c"]},
              "scan": "OFFSET"},
    "item9": {"pairs": [("3ca8bfeb40f3602a", "cf100c942f2bcf49"), ("13bdb1b842aedadd", "cf100c942f2bcf49")],
              "pubs": ["pub_e9799b75f3"], "scan": "OFFSET|Q_COARSE"},       # sc068 cells x the sc070 cell
    "item10": {"pubs": ["pub_0fd6ab4e40"], "scan": "Q_ADJEQ"},
    "item11": {"pubs": ["pub_a49146c8fd"], "scan": "Q_SMALLINT"},
}


def resolve(prefix: str, ids) -> str:
    hit = [c for c in ids if c.startswith(prefix)]
    if len(hit) != 1:
        raise KeyError(f"cell prefix {prefix!r} matches {len(hit)} cells")
    return hit[0]


def pair_index(n: int):
    I, J = np.triu_indices(n, 1)
    return I.astype(np.int64), J.astype(np.int64)


# ------------------------------------------------------------------------------------------------------------
# frozen bit-identity rule
# ------------------------------------------------------------------------------------------------------------
def bit_identity(Y: np.ndarray, ids, *, min_ident: int = BIT_MIN_IDENT, min_frac: float = BIT_MIN_FRAC,
                 keep_min_ident: int = 2) -> pd.DataFrame:
    """All pairs with n_ident >= keep_min_ident: cell_a < cell_b (table order of ``ids``), n_shared, n_ident,
    frac_ident, ident_metals and flag_bit (the frozen rule).  Equality is exact float64 equality."""
    Y = np.asarray(Y, float)
    ids = [str(c) for c in ids]
    I, J = pair_index(len(ids))
    fin = np.isfinite(Y)
    sh = fin[I] & fin[J]
    eq = sh & (Y[I] == Y[J])
    ns, ne = sh.sum(1), eq.sum(1)
    keep = np.where(ne >= keep_min_ident)[0]
    flag = (ne >= min_ident) & (ne >= min_frac * ns)
    rows = []
    for p in keep:
        rows.append({"cell_a": ids[I[p]], "cell_b": ids[J[p]], "n_shared": int(ns[p]), "n_ident": int(ne[p]),
                     "frac_ident": float(ne[p] / ns[p]), "flag_bit": bool(flag[p]),
                     "ident_metals": ",".join(LN[m] for m in np.where(eq[p])[0])})
    return pd.DataFrame(rows, columns=["cell_a", "cell_b", "n_shared", "n_ident", "frac_ident", "flag_bit",
                                       "ident_metals"])


def flagged_links(bt: pd.DataFrame) -> list:
    return [(r.cell_a, r.cell_b) for r in bt.itertuples(index=False) if r.flag_bit]


# ------------------------------------------------------------------------------------------------------------
# constant-offset and near-identity scan
# ------------------------------------------------------------------------------------------------------------
def _windows(D: np.ndarray, min_len: int, sd_max: float):
    """Longest run (length >= min_len) of consecutive columns of D with sample sd <= sd_max, per row; for equal
    length the smallest sd.  Returns (length, start, mean, sd) arrays (length 0 = none)."""
    P, n = D.shape
    c1 = np.concatenate([np.zeros((P, 1)), np.cumsum(D, 1)], 1)
    c2 = np.concatenate([np.zeros((P, 1)), np.cumsum(D * D, 1)], 1)
    blen = np.zeros(P, int)
    bst = np.zeros(P, int)
    bmean = np.full(P, np.nan)
    bsd = np.full(P, np.nan)
    for L in range(n, min_len - 1, -1):
        starts = np.arange(0, n - L + 1)
        S1 = c1[:, starts + L] - c1[:, starts]
        S2 = c2[:, starts + L] - c2[:, starts]
        mean = S1 / L
        sd = np.sqrt(np.clip((S2 - L * mean * mean) / (L - 1), 0.0, None))
        ok = sd <= sd_max
        sdm = np.where(ok, sd, np.inf)
        k = np.argmin(sdm, 1)
        hit = np.isfinite(sdm[np.arange(P), k]) & (blen == 0)
        blen[hit] = L
        bst[hit] = starts[k[hit]]
        bmean[hit] = mean[hit, k[hit]]
        bsd[hit] = sd[hit, k[hit]]
    return blen, bst, bmean, bsd


def closest_window(Y: np.ndarray, i: int, j: int, lengths=(OFF_MIN_LEN, 8)) -> dict:
    """Diagnostic for a named pair the scan did not flag: over the shared metals, the all-shared mean / sd / max |d|
    and, per window length L, the minimum-sd run of L consecutive shared metals (no threshold applied)."""
    Y = np.asarray(Y, float)
    m = np.where(np.isfinite(Y[i]) & np.isfinite(Y[j]))[0]
    d = Y[i, m] - Y[j, m]
    out = {"n_shared": int(len(m)), "mean": float(d.mean()) if len(m) else None,
           "sd": float(d.std(ddof=1)) if len(m) > 1 else None, "max_abs": float(np.abs(d).max()) if len(m) else None}
    for L in lengths:
        if len(m) < L:
            continue
        sds = [(float(d[s:s + L].std(ddof=1)), s) for s in range(len(m) - L + 1)]
        sd, s = min(sds)
        out[f"L{L}"] = {"segment": f"{LN[m[s]]}-{LN[m[s + L - 1]]}", "mean": float(d[s:s + L].mean()), "sd": sd}
    return out


def offset_scan(Y: np.ndarray, ids, bit_flagged=(), *, min_len: int = OFF_MIN_LEN, sd_max: float = OFF_SD_MAX,
                min_shift: float = OFF_MIN_SHIFT, nearid_max: float = NEARID_MAXABS) -> pd.DataFrame:
    """OFFSET and NEAR_ID links (see the module docstring); only flagged pairs are returned."""
    Y = np.asarray(Y, float)
    ids = [str(c) for c in ids]
    bits = {frozenset(x) for x in bit_flagged}
    fin = np.isfinite(Y)
    I, J = pair_index(len(ids))
    sh = fin[I] & fin[J]
    ns = sh.sum(1)
    sel = np.where(ns >= min_len)[0]
    keys = np.packbits(sh[sel], axis=1)
    keyv = keys[:, 0].astype(np.int64) * 256 + keys[:, 1].astype(np.int64)
    rows = []
    for kv in np.unique(keyv):
        g = sel[keyv == kv]
        mets = np.where(sh[g[0]])[0]
        D = Y[I[g]][:, mets] - Y[J[g]][:, mets]
        blen, bst, bmean, bsd = _windows(D, min_len, sd_max)
        mx = np.abs(D).max(1)
        off = (blen >= min_len) & (np.abs(bmean) >= min_shift)
        nid = mx <= nearid_max
        for q in np.where(off | nid)[0]:
            a, b = ids[I[g[q]]], ids[J[g[q]]]
            isbit = frozenset((a, b)) in bits
            o = bool(off[q])
            seg = (f"{LN[mets[bst[q]]]}-{LN[mets[bst[q] + blen[q] - 1]]}" if blen[q] else "")
            rows.append({"cell_a": a, "cell_b": b, "n_shared": int(len(mets)), "off_len": int(blen[q]), "off_segment": seg,
                         "off_mean": float(bmean[q]) if blen[q] else np.nan, "off_sd": float(bsd[q]) if blen[q] else np.nan,
                         "flag_offset": o, "integer_shift": bool(o and abs(bmean[q] - round(bmean[q])) < 0.005
                                                                and round(bmean[q]) != 0),
                         "nearid_maxabs": float(mx[q]), "flag_nearid": bool(nid[q] and not isbit)})
    out = pd.DataFrame(rows, columns=["cell_a", "cell_b", "n_shared", "off_len", "off_segment", "off_mean", "off_sd",
                                      "flag_offset", "integer_shift", "nearid_maxabs", "flag_nearid"])
    return out[out.flag_offset | out.flag_nearid].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------------
# quantisation scan
# ------------------------------------------------------------------------------------------------------------
def _decimals(D: float, dps=range(-3, 9), tol: float = 1e-9):
    """Smallest decimal places dp with D on the 10^-dp grid (relative tol); None when no grid fits."""
    for dp in dps:
        s = 10.0 ** dp
        if abs(D - round(D * s) / s) <= tol * D:
            return dp
    return None


def _small_ratio(r: float, pq: int = Q_SMALLINT_PQ, tol: float = 1e-6) -> bool:
    for q in range(1, pq + 1):
        p = round(r * q)
        if 1 <= p <= pq and abs(r - p / q) <= tol * r:
            return True
    return False


def quant_scan(Y: np.ndarray, ids) -> pd.DataFrame:
    """Per-cell quantisation metrics and flags (module docstring)."""
    Y = np.asarray(Y, float)
    rows = []
    for i, c in enumerate(ids):
        y = Y[i]
        mets = np.where(np.isfinite(y))[0]
        v = y[mets]
        Dv = 10.0 ** v
        dps = [_decimals(float(x)) for x in Dv]
        known = [d for d in dps if d is not None]
        dp_cell = max(known) if len(known) == len(dps) and known else None
        q = 10.0 ** (-dp_cell) if dp_cell is not None else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            half = np.log10((Dv + q / 2) / (Dv - q / 2)) / 2 if dp_cell is not None else np.full(len(Dv), np.nan)
        sig = [int(np.floor(np.log10(x))) + d + 1 for x, d in zip(Dv, dps) if d is not None]
        _, cnt = np.unique(v, return_counts=True)
        adj = [(a, b) for a, b in zip(mets[:-1], mets[1:]) if b - a == 1 and not (LN[a] == "Nd" and LN[b] == "Sm")]
        adjeq = sum(1 for a, b in adj if y[a] == y[b])
        si = [_small_ratio(float(10.0 ** (y[a] - y[b]))) or _small_ratio(float(10.0 ** (y[b] - y[a]))) for a, b in adj]
        med_half = float(np.nanmedian(half)) if len(half) and np.isfinite(half).any() else np.nan
        rows.append({"cell_id": str(c), "n_finite": int(len(mets)), "dp_cell": dp_cell, "q_cell": q,
                     "sig_min": int(min(sig)) if sig else None, "sig_max": int(max(sig)) if sig else None,
                     "half_median": med_half, "half_max": float(np.nanmax(half)) if np.isfinite(half).any() else np.nan,
                     "max_tie": int(cnt.max()) if len(cnt) else 0, "n_adjacent": len(adj), "n_adjeq": int(adjeq),
                     "n_smallint": int(sum(si)),
                     "Q_TIES": bool(len(cnt) and cnt.max() >= Q_TIE_MIN), "Q_ADJEQ": adjeq >= 1,
                     "Q_COARSE": bool(dp_cell is not None and q >= Q_COARSE_STEP and med_half >= Q_COARSE_HALF),
                     "Q_SMALLINT": bool(len(adj) >= 3 and sum(si) >= Q_SMALLINT_FRAC * len(adj)),
                     "Q_UNRESOLVED": dp_cell is None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
# lab-1 near-copy donors (label-free) and metal-concentration duplicates
# ------------------------------------------------------------------------------------------------------------
def medium_keys(f: pd.DataFrame):
    """(acid_id, diluent_id, log10 acid) per cell, as lab1skepB_nearcopy.py."""
    acid_cols = [c for c in f.columns if c.startswith("cond__acid__")]
    dil_cols = [c for c in f.columns if c.startswith("cond__diluent__")]
    Aa = f[acid_cols].fillna(0).to_numpy(float)
    Dd = f[dil_cols].fillna(0).to_numpy(float)
    acid_id = Aa.argmax(1) + 1000 * (Aa.max(1) > 0)
    dil_id = Dd.argmax(1) + 1000 * (Dd.max(1) > 0)
    lacid = np.log10(np.clip(f["cond__acid_concentration_M"].astype(float).to_numpy(), 1e-6, None))
    return acid_id, dil_id, lacid


def near_copy_mask(y_reads: np.ndarray, Ydon_reads: np.ndarray, same_med: np.ndarray, tol: float = NC_TOL) -> np.ndarray:
    """lab-1 rule: donors in the same medium whose log D lies within tol of the target's at every read metal (a
    missing donor value at a read metal is not 'within')."""
    with np.errstate(invalid="ignore"):
        close = np.all(np.abs(Ydon_reads - y_reads[None, :]) <= tol, axis=1)
    return same_med & close


def metal_conc_links(f: pd.DataFrame, finite: np.ndarray) -> pd.DataFrame:
    """Pairs of cells with the same publication and extractant whose cond__ metadata differ only in metal
    concentration (recorded in both), with the number of shared finite metals (metadata only)."""
    cm = POP.CondMeta(f)
    ids = f["cell_id"].astype(str).tolist()
    km = cm.cols.index("cond__metal_concentration_mM")
    key = f["publication_id"].astype(str) + "|" + f["extractant"].astype(str)
    rows = []
    for _, g in f.groupby(key):
        for i, j in combinations(g.index.tolist(), 2):
            if cm.diff_fams(i, j) == frozenset({"metal_conc"}) and np.isfinite(cm.F[i, km]) and np.isfinite(cm.F[j, km]):
                rows.append({"cell_a": ids[i], "cell_b": ids[j], "publication": f.at[i, "publication_id"],
                             "mconc_a": float(cm.F[i, km]), "mconc_b": float(cm.F[j, km]),
                             "n_shared": int((finite[i] & finite[j]).sum())})
    return pd.DataFrame(rows, columns=["cell_a", "cell_b", "publication", "mconc_a", "mconc_b", "n_shared"])


def metal_conc_duplicates(f: pd.DataFrame, finite: np.ndarray, min_shared: int = 3) -> list:
    """Groups (sorted lists) of >= 2 cells linked by ``metal_conc_links`` with >= min_shared shared metals: a repeated
    measurement of one condition at another metal loading.  Pairs sharing no metal are split series (complementary
    halves of one series), not duplicates.  The first member of a group is the kept representative (most finite
    metals, then the smallest cell id)."""
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    lk = metal_conc_links(f, finite)
    lk = lk[lk.n_shared >= min_shared]
    par = {c: c for c in set(lk.cell_a) | set(lk.cell_b)}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in zip(lk.cell_a, lk.cell_b):
        par[find(a)] = find(b)
    comp = {}
    for c in par:
        comp.setdefault(find(c), []).append(c)
    return sorted(sorted(m, key=lambda c: (-int(finite[pos[c]].sum()), c)) for m in comp.values() if len(m) >= 2)


def near_copy_scan(Y: np.ndarray, f: pd.DataFrame, read_sets: dict, v2drop, min_donor_metals: int = 6) -> pd.DataFrame:
    """Apply the lab-1 rule to every (design, seed, C3 target, own-publication other-extractant donor with >= 6
    finite metals, neither V2-dropped).  ``read_sets``: {(design, seed, cell): [metal index, ...]}.  Returns one row
    per (target, donor) with the number of read sets scanned and flagged."""
    Y = np.asarray(Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f["publication_id"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    acid_id, dil_id, lacid = medium_keys(f)
    v2drop = set(v2drop)
    okd = np.array([c not in v2drop for c in ids]) & (np.isfinite(Y).sum(1) >= min_donor_metals)
    acc = {}
    for (design, seed, t), rd in sorted(read_sets.items()):
        if t in v2drop:
            continue
        i = pos[t]
        dX = np.where(okd & (pub == pub[i]) & (ext != ext[i]) & (np.arange(len(ids)) != i))[0]
        if not len(dX):
            continue
        rd = list(rd)
        same_med = (acid_id[dX] == acid_id[i]) & (dil_id[dX] == dil_id[i]) & (np.abs(lacid[dX] - lacid[i]) <= NC_DACID)
        nc = near_copy_mask(Y[i, rd], Y[dX][:, rd], same_med)
        for j, flag in zip(dX, nc):
            a = acc.setdefault((t, ids[j]), [0, 0, set()])
            a[0] += 1
            a[1] += int(flag)
            if flag:
                a[2].add(f"{design}:{seed}")
    rows = [{"target": t, "donor": d, "publication": pub[pos[t]], "same_medium": bool(
        (acid_id[pos[d]] == acid_id[pos[t]]) & (dil_id[pos[d]] == dil_id[pos[t]])
        & (abs(lacid[pos[d]] - lacid[pos[t]]) <= NC_DACID)), "n_readsets": a[0], "n_flagged": a[1],
             "flagged_in": ";".join(sorted(a[2]))} for (t, d), a in sorted(acc.items())]
    return pd.DataFrame(rows, columns=["target", "donor", "publication", "same_medium", "n_readsets", "n_flagged",
                                       "flagged_in"])


def pair_class(a: str, b: str, pub: dict, ext: dict) -> str:
    sp = "within_pub" if pub[a] == pub[b] else "cross_pub"
    se = "same_ext" if ext[a] == ext[b] else "cross_ext"
    return f"{sp}|{se}"
