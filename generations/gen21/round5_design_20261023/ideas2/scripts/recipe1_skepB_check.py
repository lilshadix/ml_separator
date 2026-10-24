#!/usr/bin/env python
"""Skeptic B check of idea recipe-1 (reads-equivalent of an own prior series).  EXPLORATORY: all 229 C3 cells
have been seen in rounds 1-4 and in the round-5 probes; nothing here is a test.

Reads only: the V1 bench (gen20k.data.load_bench), ideas2/recipe_out/recipe_cells.parquet and recipe_pairs.parquet
(written by recipe_build.py), and probes/P4_corpus_registry.csv (author lists, keyed by DOI; publication_id is
'pub_' + sha1(doi)[:10]).  Writes only under ideas2/recipe1_skepB_out/.

  A  reproduce the headline numbers (OWN1_noC5, 115 cells) from recipe_pairs
  B  strata: n_own = 1 vs >= 2; giant cluster vs rest; per-publication deltas; top-3 share; LOPO range
  C  dose-response the external leg will actually see: SH built from m = 1 or 2 randomly chosen own siblings
     (exact expectation over sibling subsets) at 2 HULL reads, against D7 at 3/4/5 HULL reads
  D  condition-change type of the own siblings (all acid-only vs other)
  E  literature pool split by author overlap with the target's publication (same group vs other group) and by
     condition match; weight matched to the own pool; best-w upper bound
  F  power / false-kill rates of the proposed kill rules, resampling publications as families

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe1_skepB_check.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r4.refs import HIST_D7_BETA  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402

BASE = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023")
RO = BASE / "ideas2" / "recipe_out"
OUT = BASE / "ideas2" / "recipe1_skepB_out"
X = interp.X_R
K = 1.78
GIANT = "g:pub_74428aeab2"
rng = np.random.default_rng(1023)


def r2(sse, syy, sy, N):
    return 1 - sse / (syy - sy ** 2 / N)


def pwl(read_idx, vals, x):
    """PWL interpolation along x through read positions, constant continuation."""
    ri = np.array(sorted(read_idx))
    return np.interp(x, x[ri], vals[ri])


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(dr|prof|mr|ms)\.?\s", " ", s)
    s = re.sub(r"[^a-z\s\-]", " ", s).strip()
    toks = [t for t in s.split() if len(t) > 1]
    return toks[-1] if toks else ""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    cells = pd.read_parquet(RO / "recipe_cells.parquet")
    pairs = pd.read_parquet(RO / "recipe_pairs.parquet")
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    pos = {c: i for i, c in enumerate(f.cell_id.astype(str))}
    pubarr = f.publication_id.astype(str).to_numpy()
    extarr = f.extractant.astype(str).to_numpy()
    cond_cols = [c for c in f.columns if c.startswith("cond__")]
    F = f[cond_cols].to_numpy(float)

    # ---------- authors per publication ----------
    reg = pd.read_csv(BASE / "probes" / "P4_corpus_registry.csv")
    auth = {}
    for k, a in zip(reg.key, reg.authors):
        pid = "pub_" + hashlib.sha1(str(k).encode()).hexdigest()[:10]
        names = re.split(r";|,", str(a))
        auth[pid] = {norm_name(n) for n in names if norm_name(n)}
    P("publications with author lists:", sum(p in auth for p in set(pubarr)), "of", len(set(pubarr)))

    def overlap(p, q):
        if p not in auth or q not in auth:
            return -1
        return len(auth[p] & auth[q])

    def sse_of(arm, n, idx):
        s = pairs[pairs.n_read == n].groupby("ci")[f"hull_{arm}"].sum()
        return s.reindex(range(len(cells))).fillna(0).to_numpy()[idx]

    def pooled(sub, sse):
        return r2(sse.sum(), sub.syy.sum(), sub.sy.sum(), sub.N.sum())

    def boot(sub, a, b=None, B=4000):
        pubs = sub.publication.to_numpy()
        up = np.unique(pubs)
        G = (pubs[:, None] == up[None, :]).astype(float)
        sa, syy, sy, N = a @ G, sub.syy.to_numpy() @ G, sub.sy.to_numpy() @ G, sub.N.to_numpy() @ G
        sb = b @ G if b is not None else None
        W = rng.multinomial(len(up), np.ones(len(up)) / len(up), size=B).astype(float)
        ra = 1 - (W @ sa) / (W @ syy - (W @ sy) ** 2 / (W @ N))
        if sb is None:
            return np.percentile(ra, [2.5, 97.5])
        rb = 1 - (W @ sb) / (W @ syy - (W @ sy) ** 2 / (W @ N))
        # callers pass NEGATED SSE (-SSE_a, -SSE_b); then ra - rb = R2_b - R2_a, so flip sign to get R2_a - R2_b
        return np.percentile(-(ra - rb), [2.5, 97.5])

    own = cells[(cells.n_own >= 1) & ~cells.c5].reset_index(drop=True)
    idx = own.ci.to_numpy()
    A = {}
    A["COPY@0"] = sse_of("SIB_own", 0, idx)
    for n in (2, 3, 4):
        A[f"SH@{n}"] = sse_of("SH_own", n, idx)
    for n in (2, 3, 4, 5, 6):
        A[f"D7@{n}"] = sse_of("D7", n, idx)
    P("\n=== A. reproduction, OWN1_noC5 (n cells, pubs, clusters):", len(own), own.publication.nunique(),
      own.cluster.nunique())
    P("   ", {k: round(pooled(own, v), 3) for k, v in A.items()})

    # ---------- B. strata ----------
    P("\n=== B. strata (pooled R2; SH@2 - D7@m)")
    strata = {"n_own=1": own.n_own == 1, "n_own>=2": own.n_own >= 2, "giant": own.cluster == GIANT,
              "non-giant": own.cluster != GIANT, "non-giant & n_own>=2": (own.cluster != GIANT) & (own.n_own >= 2)}
    rows = []
    for nm, msk in strata.items():
        s = own[msk.to_numpy()]
        ii = msk.to_numpy()
        rec = {"stratum": nm, "cells": len(s), "pubs": s.publication.nunique(), "clusters": s.cluster.nunique(),
               "pairs": int(s.N.sum())}
        for k, v in A.items():
            rec[k] = round(pooled(s, v[ii]), 3)
        if s.publication.nunique() >= 3:
            rec["SH2-D74 CI"] = [round(v, 3) for v in boot(s, -A["SH@2"][ii], -A["D7@4"][ii])]
            rec["SH2-D73 CI"] = [round(v, 3) for v in boot(s, -A["SH@2"][ii], -A["D7@3"][ii])]
        rows.append(rec)
    Bdf = pd.DataFrame(rows)
    P(Bdf.to_string(index=False))

    # per-publication: SSE difference D7@4 - SH@2, share of total gain
    own["dSSE_SH2_D74"] = A["D7@4"] - A["SH@2"]
    own["dSSE_COPY_D73"] = A["D7@3"] - A["COPY@0"]
    own["SST_cell"] = own.syy - own.sy ** 2 / own.N
    pp = own.groupby("publication").agg(cells=("ci", "size"), cluster=("cluster", "first"), n_own=("n_own", "mean"),
                                        dSSE=("dSSE_SH2_D74", "sum"), dSSE0=("dSSE_COPY_D73", "sum"),
                                        syy=("syy", "sum"), sy=("sy", "sum"), N=("N", "sum"))
    for k in ("SH@2", "D7@4", "D7@3", "COPY@0", "D7@2"):
        pp[k] = pd.Series(A[k], index=own.index).groupby(own.publication).sum()
    pp["SST"] = pp.syy - pp.sy ** 2 / pp.N
    pp["dR2_own"] = (pp["D7@4"] - pp["SH@2"]) / pp.SST          # per-family R2 difference SH@2 - D7@4
    pp["dR2_copy_d73"] = (pp["D7@3"] - pp["COPY@0"]) / pp.SST
    pp["rel_SSE"] = pp["SH@2"] / pp["D7@4"] - 1
    tot = pp.dSSE.sum()
    pps = pp.sort_values("dSSE", ascending=False)
    P("\n per-publication SH@2 vs D7@4 (dSSE > 0 = SH better); total dSSE", round(tot, 4))
    P(pps[["cells", "cluster", "n_own", "dSSE", "dR2_own", "rel_SSE"]].round(3).to_string())
    P(" pubs with SH@2 better:", int((pp.dSSE > 0).sum()), "of", len(pp), "; top-3 share of net gain:",
      round(pps.dSSE.head(3).sum() / tot, 2))
    P(" unweighted mean per-family dR2 (SH@2 - D7@4):", round(pp.dR2_own.mean(), 3), " median",
      round(pp.dR2_own.median(), 3), " SD", round(pp.dR2_own.std(), 3))
    P(" unweighted mean per-family dR2 (COPY@0 - D7@3):", round(pp.dR2_copy_d73.mean(), 3), " median",
      round(pp.dR2_copy_d73.median(), 3))
    # LOPO
    lop = []
    for p_ in pp.index:
        m_ = (own.publication != p_).to_numpy()
        s = own[m_]
        lop.append(pooled(s, A["SH@2"][m_]) - pooled(s, A["D7@4"][m_]))
    P(" LOPO range of pooled SH@2 - D7@4:", round(min(lop), 3), round(max(lop), 3), "; min when dropping",
      pp.index[int(np.argmin(lop))])
    mx = (own.publication != pp.index[int(np.argmin(lop))]).to_numpy()
    sx = own[mx]
    P("   without that publication (%d cells): COPY %.3f SH@2 %.3f D7@3 %.3f D7@4 %.3f D7@5 %.3f"
      % (len(sx), pooled(sx, A["COPY@0"][mx]), pooled(sx, A["SH@2"][mx]), pooled(sx, A["D7@3"][mx]),
         pooled(sx, A["D7@4"][mx]), pooled(sx, A["D7@5"][mx])))
    inv = {"pub_" + hashlib.sha1(str(k).encode()).hexdigest()[:10]: (k, str(t)[:90], str(a)[:80])
           for k, t, a in zip(reg.key, reg.title, reg.authors)}
    for p_ in list(pps.index[:3]) + list(pps.index[-3:]):
        P("   ", p_, inv.get(p_))
    pp.to_csv(OUT / "B_per_publication.csv")

    # ---------- C. external-leg dose-response: m random own siblings ----------
    P("\n=== C. SH from m random own siblings (exact expectation), 2 HULL reads, cells with n_own >= 2")
    ccache = []
    for r in own.itertuples():
        i = pos[r.cell_id]
        Mi = [LANTHANIDES.index(s) for s in r.metals.split(",")]
        m = len(Mi)
        x = X[Mi]
        yv = Y[i, Mi]
        pa = np.array([p for p in range(m - 1) if Z_VEC[Mi[p + 1]] - Z_VEC[Mi[p]] == 1])
        pb = pa + 1
        u = np.zeros(m)
        if CO.GD in Mi:
            u[Mi.index(CO.GD)] = HIST_D7_BETA
        ends = [0, m - 1]
        d7 = u + pwl(ends, yv - u, x)
        sib_ids = [pos[j] for j in json.loads(r.own_ids)]
        rec = {"ci": r.ci, "n_own": r.n_own, "cluster": r.cluster, "publication": r.publication}
        for mm in (1, 2):
            if len(sib_ids) < mm:
                rec[f"m{mm}"] = np.nan
                continue
            w = mm / (mm + K)
            ss = []
            combs = list(itertools.combinations(sib_ids, mm))
            for cb in combs:
                T = Y[np.ix_(list(cb), Mi)].mean(0)
                sib = T + pwl(ends, yv - T, x)
                sh = w * sib + (1 - w) * d7
                e = (sh[pa] - sh[pb]) - (yv[pa] - yv[pb])
                ss.append((e ** 2).sum())
            rec[f"m{mm}"] = float(np.mean(ss))
            # raw template (no shrink) m = 1
            if mm == 1:
                ss2 = []
                for cb in combs:
                    T = Y[np.ix_(list(cb), Mi)].mean(0)
                    sib = T + pwl(ends, yv - T, x)
                    e = (sib[pa] - sib[pb]) - (yv[pa] - yv[pb])
                    ss2.append((e ** 2).sum())
                rec["raw_m1"] = float(np.mean(ss2))
        # check own D7@2 against recipe_pairs
        e = (d7[pa] - d7[pb]) - (yv[pa] - yv[pb])
        rec["d7_2_check"] = float((e ** 2).sum())
        ccache.append(rec)
    C = pd.DataFrame(ccache)
    P(" D7@2 recompute max |diff| vs recipe_pairs:", float(np.abs(C.d7_2_check.to_numpy() - A["D7@2"]).max()))
    m2 = (own.n_own >= 2).to_numpy()
    s2 = own[m2]
    resC = {"cells": int(m2.sum()), "pubs": int(s2.publication.nunique()),
            "SH_m1@2": pooled(s2, C.m1.to_numpy()[m2]), "RAW_m1@2": pooled(s2, C.raw_m1.to_numpy()[m2]),
            "SH_m2@2": pooled(s2, C.m2.to_numpy()[m2]), "SH_all@2": pooled(s2, A["SH@2"][m2])}
    for n in (2, 3, 4, 5):
        resC[f"D7@{n}"] = pooled(s2, A[f"D7@{n}"][m2])
    P(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in resC.items()})
    dlo = boot(s2, -C.m1.to_numpy()[m2], -A["D7@4"][m2])
    P("  SH_m1@2 - D7@4 = %.3f [%.3f, %.3f]" % (resC["SH_m1@2"] - resC["D7@4"], *dlo))
    dlo = boot(s2, -C.m1.to_numpy()[m2], -A["D7@3"][m2])
    P("  SH_m1@2 - D7@3 = %.3f [%.3f, %.3f]" % (resC["SH_m1@2"] - resC["D7@3"], *dlo))
    dlo = boot(s2, -C.m2.to_numpy()[m2], -A["D7@4"][m2])
    P("  SH_m2@2 - D7@4 = %.3f [%.3f, %.3f]" % (resC["SH_m2@2"] - resC["D7@4"], *dlo))
    # natural n_own = 1 stratum
    m1 = (own.n_own == 1).to_numpy()
    s1 = own[m1]
    P("  natural n_own=1 stratum (%d cells, %d pubs): SH@2 %.3f, D7@3 %.3f, D7@4 %.3f, D7@5 %.3f; SH@2-D7@4 [%.3f, %.3f]"
      % (m1.sum(), s1.publication.nunique(), pooled(s1, A["SH@2"][m1]), pooled(s1, A["D7@3"][m1]),
         pooled(s1, A["D7@4"][m1]), pooled(s1, A["D7@5"][m1]), *boot(s1, -A["SH@2"][m1], -A["D7@4"][m1])))
    # per-publication m1 deltas (for power)
    own["dSSE_m1"] = A["D7@4"] - C.m1.to_numpy()
    ppm1 = own[m2].groupby("publication").agg(d=("dSSE_m1", "sum"), syy=("syy", "sum"), sy=("sy", "sum"),
                                              N=("N", "sum"))
    ppm1["dR2"] = ppm1.d / (ppm1.syy - ppm1.sy ** 2 / ppm1.N)
    P("  per-family dR2 (SH_m1@2 - D7@4), n_own>=2 pubs:", ppm1.dR2.round(3).tolist())

    # ---------- D. condition-change type ----------
    P("\n=== D. own-sibling condition-change type")
    fams = own.own_fams.apply(json.loads)
    acid_only = fams.apply(lambda L: all(s == "acid_conc" for s in L)).to_numpy()
    for nm, msk in (("all siblings acid-only", acid_only), ("any non-acid sibling", ~acid_only)):
        s = own[msk]
        P("  %-24s %3d cells %2d pubs: COPY %.3f SH@2 %.3f D7@3 %.3f D7@4 %.3f D7@5 %.3f; share of net dSSE %.2f"
          % (nm, len(s), s.publication.nunique(), pooled(s, A["COPY@0"][msk]), pooled(s, A["SH@2"][msk]),
             pooled(s, A["D7@3"][msk]), pooled(s, A["D7@4"][msk]), pooled(s, A["D7@5"][msk]),
             own.dSSE_SH2_D74[msk].sum() / own.dSSE_SH2_D74.sum()))

    # ---------- E. literature pool split ----------
    P("\n=== E. literature (same extractant, other publication) split by author overlap and condition match")
    FAM_TRIV = {"contact_time", "metal_conc"}

    def fam(c):
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

    def diff_fams(i, j):
        s = set()
        for q, c in enumerate(cond_cols):
            a, b = F[i, q], F[j, q]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b) or not np.isclose(a, b):
                s.add(fam(c))
        return s

    from r3 import datav2 as V2
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okp = np.array([c not in v2drop for c in f.cell_id.astype(str)])
    erows = []
    noc5 = cells[~cells.c5 & ~cells.v2drop]
    for r in noc5.itertuples():
        i = pos[r.cell_id]
        Mi = [LANTHANIDES.index(s) for s in r.metals.split(",")]
        m = len(Mi)
        x = X[Mi]
        yv = Y[i, Mi]
        pa = np.array([p for p in range(m - 1) if Z_VEC[Mi[p + 1]] - Z_VEC[Mi[p]] == 1])
        pb = pa + 1
        cover = np.all(np.isfinite(Y[:, Mi]), axis=1)
        cover[i] = False
        xp = np.where((extarr == extarr[i]) & (pubarr != pubarr[i]) & cover & okp)[0]
        if len(xp) == 0:
            continue
        ov = np.array([overlap(pubarr[i], pubarr[j]) for j in xp])
        cm = np.array([len(diff_fams(i, j) - FAM_TRIV - {"acid_conc"}) == 0 for j in xp])
        cm2 = np.array([len(diff_fams(i, j) - FAM_TRIV - {"acid_conc", "extr_conc", "temperature"}) == 0
                        for j in xp])
        u = np.zeros(m)
        if CO.GD in Mi:
            u[Mi.index(CO.GD)] = HIST_D7_BETA
        ends = [0, m - 1]
        d7 = u + pwl(ends, yv - u, x)
        ysf = yv[pa] - yv[pb]
        rec = {"ci": r.ci, "publication": r.publication, "cluster": r.cluster, "n_own": r.n_own,
               "N": r.N, "sy": r.sy, "syy": r.syy, "n_xpub": len(xp), "n_same_grp": int((ov >= 2).sum()),
               "n_other_grp": int((ov == 0).sum()), "n_condmatch": int(cm.sum()),
               "n_other_grp_condmatch": int(((ov == 0) & cm).sum()),
               "n_other_grp_medmatch": int(((ov == 0) & cm2).sum()), "n_ov1": int((ov == 1).sum()),
               "D7@2": float((((d7[pa] - d7[pb]) - ysf) ** 2).sum())}
        w_own = r.n_own / (r.n_own + K) if r.n_own > 0 else 1 / (1 + K)
        for nm, sel in (("ALL", np.ones(len(xp), bool)), ("SAMEGRP", ov >= 2), ("OTHERGRP", ov == 0),
                        ("CONDM", cm), ("OTHERGRP_CONDM", (ov == 0) & cm), ("MEDM", cm2),
                        ("OTHERGRP_MEDM", (ov == 0) & cm2), ("OTHERGRP_NOTMEDM", (ov == 0) & ~cm2)):
            if sel.sum() == 0:
                for suf in ("COPY", "SHK", "SHW"):
                    rec[f"{nm}_{suf}"] = np.nan
                for wv in (0.25, 0.5, 0.75):
                    rec[f"{nm}_w{wv}"] = np.nan
                continue
            T = Y[np.ix_(xp[sel], Mi)].mean(0)
            sib = T + pwl(ends, yv - T, x)
            rec[f"{nm}_COPY"] = float((((T[pa] - T[pb]) - ysf) ** 2).sum())
            n_ = int(sel.sum())
            for suf, w in (("SHK", n_ / (n_ + K)), ("SHW", w_own)):
                sh = w * sib + (1 - w) * d7
                rec[f"{nm}_{suf}"] = float((((sh[pa] - sh[pb]) - ysf) ** 2).sum())
            for wv in (0.25, 0.5, 0.75):
                sh = wv * sib + (1 - wv) * d7
                rec[f"{nm}_w{wv}"] = float((((sh[pa] - sh[pb]) - ysf) ** 2).sum())
        erows.append(rec)
    E = pd.DataFrame(erows)
    E.to_csv(OUT / "E_xpub_split.csv", index=False)
    P("  cells with any literature series (no C5, V2-clean):", len(E), "pubs", E.publication.nunique(),
      "clusters", E.cluster.nunique())
    P("  cells with a same-group (>=2 shared surnames) other-publication series:", int((E.n_same_grp > 0).sum()),
      "; with an other-group (0 shared) series:", int((E.n_other_grp > 0).sum()),
      "; other-group AND condition-matched (only acid conc differs):", int((E.n_other_grp_condmatch > 0).sum()))
    P("  share of literature siblings that come from author-overlapping (>=2) publications:",
      round(E.n_same_grp.sum() / E.n_xpub.sum(), 3), "; >=1 shared:", "n/a")
    P("  cells with other-group series in the same medium (same diluent/acid type/additive):",
      int((E.n_other_grp_medmatch > 0).sum()), "; literature siblings sharing exactly 1 surname:", int(E.n_ov1.sum()))
    for nm in ("ALL", "SAMEGRP", "OTHERGRP", "CONDM", "OTHERGRP_CONDM", "MEDM", "OTHERGRP_MEDM", "OTHERGRP_NOTMEDM"):
        s = E[E[f"{nm}_COPY"].notna()]
        if len(s) == 0:
            continue
        sub = s.rename(columns={})
        res = {"cells": len(s), "pubs": s.publication.nunique(), "clusters": s.cluster.nunique(),
               "D7@2": pooled(s, s["D7@2"].to_numpy())}
        for suf in ("COPY", "SHK", "SHW", "w0.25", "w0.5", "w0.75"):
            res[suf] = pooled(s, s[f"{nm}_{suf}"].to_numpy())
        if s.publication.nunique() >= 3:
            res["SHK-D7@2 CI"] = [round(v, 3) for v in boot(s, -s[f"{nm}_SHK"].to_numpy(), -s["D7@2"].to_numpy())]
        P("  %-15s" % nm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    # head-to-head: cells with both own and an other-group literature series
    hh = E[(E.n_own >= 1) & E.OTHERGRP_SHW.notna()]
    if len(hh):
        hidx = hh.ci.to_numpy()
        own_by_ci = dict(zip(own.ci, A["SH@2"]))
        cpy_by_ci = dict(zip(own.ci, A["COPY@0"]))
        ok = np.array([c in own_by_ci for c in hidx])
        hh = hh[ok]
        hidx = hh.ci.to_numpy()
        P("  head-to-head own vs OTHER-GROUP literature (%d cells, %d pubs, %d clusters): own COPY %.3f own SH@2 %.3f |"
          " lit COPY %.3f lit SH@2(w own) %.3f | D7@2 %.3f"
          % (len(hh), hh.publication.nunique(), hh.cluster.nunique(),
             pooled(hh, np.array([cpy_by_ci[c] for c in hidx])), pooled(hh, np.array([own_by_ci[c] for c in hidx])),
             pooled(hh, hh.OTHERGRP_COPY.to_numpy()), pooled(hh, hh.OTHERGRP_SHW.to_numpy()),
             pooled(hh, hh["D7@2"].to_numpy())))
    hs = E[(E.n_own >= 1) & E.SAMEGRP_SHW.notna()]
    if len(hs):
        hidx = hs.ci.to_numpy()
        own_by_ci = dict(zip(own.ci, A["SH@2"]))
        ok = np.array([c in own_by_ci for c in hidx])
        hs = hs[ok]
        hidx = hs.ci.to_numpy()
        P("  head-to-head own vs SAME-GROUP other-publication (%d cells, %d pubs): own SH@2 %.3f | samegrp COPY %.3f"
          " SH@2(w own) %.3f | D7@2 %.3f"
          % (len(hs), hs.publication.nunique(), pooled(hs, np.array([own_by_ci[c] for c in hidx])),
             pooled(hs, hs.SAMEGRP_COPY.to_numpy()), pooled(hs, hs.SAMEGRP_SHW.to_numpy()),
             pooled(hs, hs["D7@2"].to_numpy())))

    # ---------- F. kill-rule operating characteristics (publications resampled as families) ----------
    P("\n=== F. kill-rule operating characteristics, 11 families drawn from corpus per-publication deltas")

    def sim(d, nfam=11, B=20000, rule="credit"):
        d = np.asarray(d, float)
        dr = d[rng.integers(0, len(d), size=(B, nfam))]
        mu = dr.mean(1)
        se = dr.std(1, ddof=1) / np.sqrt(nfam)
        lb = mu - 1.812 * se  # one-sided 95 % t, df 10
        if rule == "credit":
            return float(((mu < 0) | (lb < -0.05)).mean())
        return float(((mu >= 0.05) & (lb > 0)).mean())

    P("  P(credit claim downgraded) if families look like corpus n_own>=1 publications (SH@2 - D7@4 per family):",
      round(sim(pp.dR2_own.to_numpy()), 3), "| per-family SD", round(pp.dR2_own.std(), 3))
    P("  same, if families look like the m = 1 (2-condition family) scenario:", round(sim(ppm1.dR2.to_numpy()), 3),
      "| mean", round(ppm1.dR2.mean(), 3), "SD", round(ppm1.dR2.std(), 3))
    for nf in (5, 8, 20, 40):
        P("   nfam=%d: P(downgrade) corpus-like %.3f, m1-like %.3f" % (nf, sim(pp.dR2_own.to_numpy(), nf),
                                                                       sim(ppm1.dR2.to_numpy(), nf)))
    # literature clause, per-publication SH_xpub(K) - D7@2 deltas
    e2 = E[E.ALL_SHK.notna()].copy()
    pe = e2.groupby("publication").agg(a=("ALL_SHK", "sum"), b=("D7@2", "sum"), syy=("syy", "sum"),
                                       sy=("sy", "sum"), N=("N", "sum"))
    pe["dR2"] = (pe.b - pe.a) / (pe.syy - pe.sy ** 2 / pe.N)
    P("  literature clause per-family SH_xpub@2 - D7@2: mean %.3f median %.3f SD %.3f (n pubs %d)"
      % (pe.dR2.mean(), pe.dR2.median(), pe.dR2.std(), len(pe)))
    P("  P('own-lab-only' clause killed) with 11 corpus-like families:", round(sim(pe.dR2.to_numpy(), rule="lit"), 3))
    # equivalence version: fraction of draws in which the upper bound of the literature credit is < +0.05
    d = pe.dR2.to_numpy()
    dr = d[rng.integers(0, len(d), size=(20000, 11))]
    ub = dr.mean(1) + 1.812 * dr.std(1, ddof=1) / np.sqrt(11)
    P("  P(equivalence 'no literature credit' shown: one-sided UB < +0.05) with 11 families:",
      round(float((ub < 0.05).mean()), 3))
    (OUT / "recipe1_skepB.log").write_text("\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
