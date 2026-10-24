"""E-KX: the sc009 (TODGA-class) Nd dip, frozen for the Pr/Nd warning (CLAUDE_CODE_PROMPT.md section 6.8; conventions
6.0, 6.10, 8, 9; IR M07 and ideas/critiques.json M07).  Stage-1 item 1.8; outputs under results/EKX/.

No design-folder script implements M07 (the skeptics' reruns were never kept), so nothing here is copied; the B1 base
is round 4's OOF prediction and every level vector goes through r4.refs.predict_levels / potentials unchanged.

**Frozen object (external leg).**  u_Nd(sc009) = -0.11, the raw training-publication mean Nd chord anomaly of sc009
[IR M07; "about -0.11", its computation not specified, so the V1 consistency line is reported, never used], shrunk
toward 0 with w = n / (n + 5), n = the number of donor publications (sealed reading of the source's n / (n + 5)).
For new data the donors are the V1 sc009 publications reporting Pr, Nd and Sm (counted from the ``metals`` metadata,
label-free), so u_ext = w * (-0.11).  It enters B1 as a point potential at Nd (r4.refs' coherent
form; the reads stay exact):  level(m) = PWL_R[y_S - u_S](m) + u(m),  u = beta 1[Gd] + u_Nd 1[Nd].

**Nd chord** of a cell with Pr, Nd and Sm measured: a_Nd = y_Nd - [(1 - w) y_Pr + w y_Sm], w = (x_Nd - x_Pr) /
(x_Sm - x_Pr) along -Shannon CN8 radius (r3.interp 'R'; Pm is absent, so Nd's neighbours are Pr and Sm) [E1 3.4].

**E-KX-LOPO** (in-corpus; never called BP, never placed next to 0.452).  Rows = round-4 REF rows (E-A's 1,802 queries
per seed at k = 3, four reads); B1 base = the round-4 OOF B1 of the design's parquet (BP primary, as section 6.8
says; BPG a sensitivity).  Donors of target t = bench cells (all 521, any k_max) in another publication AND another
copy group than t, never t itself, minus the cells linked to t by the frozen section-5.1 bit-identity rule.  Per donor
set: publication means first, then their mean (u_raw); n = donor publications; u = n / (n + 5) * u_raw (0 without a
donor).  Every donor set is checked by ``Donors.guard``.
  B1           the parquet's B1 (the comparator on identical rows).
  KX_CHEM      B1 + u_Nd(chem(t)) dphi_Nd, donors of t's own chemotype (the "chemotype Nd term", every chemotype).
  KX_SC009     KX_CHEM on sc009 targets, B1 elsewhere (the narrowed deployment arm; equals KX_CHEM on sc009 rows).
  NDU          B1 + u_Nd(all) dphi_Nd, donors of any chemotype ("B1-NdU", a universal Nd potential).
  PRND_OFFSET  Pr/Nd rows: B1 + n / (n + 5) * mean_pub(e), e = the same-chemotype C3 donor's Pr/Nd residual in the same
               seed and design, y - B0 - beta_t dphi_Gd (the donor's B1 recomputed with the TARGET's fold beta, so no
               label of t reaches its prediction through a donor's fold fit).  Other pairs: B1.
  PRND_COPY    Pr/Nd rows: the same-chemotype donors' publication-mean Pr/Nd log SF, unshrunk (copy another
               publication's SF); B1 without a donor and on every other pair.
  KX_FROZEN    DIAGNOSTIC, not fold-independent (-0.11 was computed with every sc009 publication, the target's own
               included): B1 + u_ext dphi_Nd on sc009 targets.
dphi_j = phi_j(A) - phi_j(B), phi_j(m) = 1[m = j] - w_j(m) (r4.refs.potentials; 0 at a read metal).

**Endpoints**: SC009_PRND (primary: sc009 Pr/Nd rows; R^2 and MAE), POOLED (all rows), PRND_ALL, SC009_ALL.
**C5-BTBP**: C5_out (primary, rows and donors without pub_6a0ef8528c, as M07's seen numbers), C5_in, C5_cell_out.
**Donor-pool sensitivities** (C5_out): NB (no bit rule), OFS (offset / near-identity links dropped), QNT (quantised
cells dropped), V2 (V2-1/2/3 cells dropped), V25 (a V2-5 floor reading at Pr, Nd or Sm voids the chord), C3DON (C3
donors only), CELLW (cell-weighted instead of publication-weighted mean).
**Controls**: random-chemotype permutation (chemotype labels shuffled over publication x chemotype units, unit counts
per chemotype kept; B_PERM seeded draws); publication sign-flip over sc009 publications (r5.common, exact at G <= 13);
leave-top-3-out (the report rule); stratification by |dlog10 acid| > 0.5 between t and its chemotype donors.
**Report rule** (CONSISTENCY CHECK; the expected outcome was seen): narrowed to "universal Nd term" if
R^2(KX_CHEM) - R^2(NDU) <= 0 on the POOLED rows without the 3 publications that carry most of KX_CHEM's pooled SSE gain
over B1 (design BP, C5_out, rows removed; sensitivities: C5_in, SC009_PRND, donors removed too, BPG).
**External rider** (section 6.8; not a package claim, excluded by CP-13): ``external_rider``.

Information contract (section 8): the target's labels never enter its arms (the reads enter only through the round-4
B1 base); ``intervention`` scrambles every label of the target's publication and copy group and requires
bit-identical arms, with two positive controls (the target in its own donor pool; a leaky own-publication pool).
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import paths
from . import common as CM
from . import dupes as DU
from . import population as POP
from r3 import interp  # noqa: E402
from r4 import refs as RF  # noqa: E402
from gen20k.data import Z_VEC  # noqa: E402

LN = POP.LN
N_LN = len(LN)
PR, ND, SM, GD = (LN.index(m) for m in ("Pr", "Nd", "Sm", "Gd"))
X_R = np.asarray(interp._coords("R"), float)
W_SM = float((X_R[ND] - X_R[PR]) / (X_R[SM] - X_R[PR]))     # weight of Sm in the Pr-Sm chord at Nd
CHEM = "sc009"
U_RAW = -0.11                    # frozen u_Nd(sc009) [IR M07; brief 6.8]
U_RAW_SOURCE = "about -0.11 [IR M07], computation not specified"   # the source's provenance; -0.11 stays frozen
K_SHRINK = 5.0                   # w = n / (n + 5), n = donor publications
TOP = 3
DLOG_ACID = 0.5
ACID_FLOOR_M = 1e-4              # as r5.population's condition distance
B_PERM = 2000                    # >= 1,000 random-chemotype draws (brief 6.8)
N_SPLITHALF = 200
U_PROBE = -0.1                   # construction-only value for the additive-form identity check
B1_TOL = 8.9e-16                 # brief 6.0 reproduction target [P1 2]
ADD_TOL = 1e-12
ROWS_PER_SEED = 1802
PERM_C5 = ("C5_out", "C5_in")
ARMS = ("B1", "KX_CHEM", "KX_SC009", "NDU", "PRND_OFFSET", "PRND_COPY", "KX_FROZEN")
VARIANT_ARMS = ("KX_CHEM", "KX_SC009", "NDU")
C5_VARIANTS = ("C5_out", "C5_in", "C5_cell_out")
PRIMARY_C5 = "C5_out"
DONOR_VARIANTS = ("PRIMARY", "NB", "OFS", "QNT", "V2", "V25", "C3DON", "CELLW")
CONFIGS = tuple((c5, "PRIMARY") for c5 in C5_VARIANTS) + tuple((PRIMARY_C5, v) for v in DONOR_VARIANTS[1:])
ENDPOINTS = ("SC009_PRND", "POOLED", "PRND_ALL", "SC009_ALL")
PRIMARY_ENDPOINT = "SC009_PRND"
PRIMARY_DESIGN = "BP"
CONTRASTS = (("KX_SC009", "B1"), ("KX_CHEM", "B1"), ("NDU", "B1"), ("KX_CHEM", "NDU"), ("KX_SC009", "NDU"),
             ("PRND_OFFSET", "B1"), ("PRND_COPY", "B1"), ("KX_SC009", "PRND_OFFSET"), ("KX_SC009", "PRND_COPY"),
             ("KX_FROZEN", "B1"))
SEEN = "EXPLORATORY (seen)"
SENS = "SENSITIVITY"
DIAG = "DIAGNOSTIC"
CC = "CONSISTENCY CHECK"
SMOKE = "TECHNICAL SMOKE"
NOT_NEXT_TO = ("E-KX-LOPO (known-chemotype deployment; B1 base = round-4 OOF, donors = other publications and other "
               "copy groups): never called BP, never placed next to E-A's 0.452; B1 on the same rows is the comparator")
SEEN_VALUES = {  # IR M07 / brief 6.8, quoted as written (units as in the source); none is a reproduction target
    "sc009_PrNd_R2": {"B1": 0.265, "NDU": 0.419, "KX_SC009": 0.522}, "sc009_PrNd_rows_per_seed": 46,
    "sc009_chord_negative_publications": "11 of 12", "u_raw_sc009": -0.11, "top3_share_of_gain": 0.75,
    "pooled_dR2_without_top3": {"KX_CHEM": 0.0010, "NDU": 0.0027},
    "pooled_dR2": {"KX_CHEM": "+0.011 to +0.023", "NDU": "+0.0053 to +0.0063", "PRND_OFFSET": "+0.0082",
                   "PRND_COPY": "-0.45"},
    "random_chemotype_max": "+0.0035 to +0.0048 (40 draws) vs observed +0.011", "NdPr_SF": {"B1": 1.7, "observed": 1.34},
    "acid_rho": "-0.28 to -0.36"}
RIDER_RULE = ("killed if the Nd chord is >= 0 in >= 2 of the first 4 new sc009 publications, or if the Pr/Nd squared "
              "error against B1 does not fall in most (>= 3) of them [IR M07; brief 6.8]")


class LeakageError(AssertionError):
    """A donor pool that breaks the E-KX information contract."""


# ------------------------------------------------------------------------------------------------------------
# pure helpers
# ------------------------------------------------------------------------------------------------------------
def shrink(n, K: float = K_SHRINK):
    n = np.asarray(n, float)
    out = n / (n + K)
    return float(out) if out.ndim == 0 else out


def nd_chord(Y) -> np.ndarray:
    """Nd chord anomaly per row of a (cells x 14) matrix; NaN unless Pr, Nd and Sm are finite."""
    Y = np.atleast_2d(np.asarray(Y, float))
    return Y[:, ND] - ((1.0 - W_SM) * Y[:, PR] + W_SM * Y[:, SM])


def prnd_sf(Y) -> np.ndarray:
    Y = np.atleast_2d(np.asarray(Y, float))
    return Y[:, PR] - Y[:, ND]


def reads_of(s) -> frozenset:
    return frozenset(LN.index(m) for m in str(s).split(",") if m)


def masked(y, read) -> np.ndarray:
    ym = np.full(N_LN, np.nan)
    r = sorted(int(m) for m in read)
    ym[r] = np.asarray(y, float)[r]
    if not np.isfinite(ym[r]).all():
        raise ValueError("a read metal has no finite value")
    return ym


def phi(read, metals, j: int) -> dict:
    """{unread metal b: phi_j(b)} (r4.refs.potentials; values at the reads are irrelevant, only positions)."""
    ym = np.full(N_LN, np.nan)
    ym[sorted(int(m) for m in read)] = 0.0
    return {int(b): float(v[0]) for b, v in RF.potentials(ym, metals, (j,)).items()}


def dphi_pairs(read, metals, pairs, j: int) -> np.ndarray:
    p = phi(read, metals, j)
    return np.array([p.get(int(a), 0.0) - p.get(int(b), 0.0) for a, b in pairs], float)


def b1_fit(beta: float) -> dict:
    return {"name": "B1", "spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([float(beta)])}


def kx_fit(beta: float, u_nd: float) -> dict:
    """B1 with a second point potential at Nd (the frozen E-KX form)."""
    return {"name": "B1_ND", "spec": {"pot": (GD, ND), "geo": False}, "kind": "R",
            "coef": np.array([float(beta), float(u_nd)])}


def levels(fit: dict, ym, metals) -> np.ndarray:
    return RF.predict_levels(fit, SimpleNamespace(y_masked=np.asarray(ym, float)), tuple(metals))


def pair_sf(lv, pairs) -> np.ndarray:
    return np.array([lv[a] - lv[b] for a, b in pairs], float)


def pub_means(vals, idx, pcode, n_pub: int, weighting: str = "pub", *, full: bool = False):
    """(u_raw, n_pub, n_cell[, sd of publication means]) of vals[idx] over its finite entries: publication means
    first, then their plain mean ('pub'), or the cell mean ('cell').  np.bincount sums in index order, so the result
    depends only on the donor set and its values (the intervention needs bit-identity)."""
    idx = np.asarray(idx, dtype=np.int64)
    v = np.asarray(vals, float)[idx]
    ok = np.isfinite(v)
    idx, v = idx[ok], v[ok]
    if len(v) == 0:
        return (float("nan"), 0, 0, float("nan")) if full else (float("nan"), 0, 0)
    pc = pcode[idx]
    c = np.bincount(pc, minlength=n_pub)
    s = np.bincount(pc, weights=v, minlength=n_pub)
    has = c > 0
    pm = s[has] / c[has]
    raw = float(v.mean()) if weighting == "cell" else float(pm.mean())
    if not full:
        return raw, int(has.sum()), int(len(v))
    return raw, int(has.sum()), int(len(v)), float(pm.std(ddof=1)) if len(pm) > 1 else float("nan")


def fold_betas(design: str) -> dict:
    """{(split_seed, fold): beta} from round4_20261022/results/REF/fitted_<design>.json (the stored B1 fits)."""
    d = json.loads((paths.R4RES / "REF" / f"fitted_{design}.json").read_text(encoding="utf-8"))
    out = {tuple(int(x) for x in k.split("_")): float(v["ref:B1"]["coef"][0]) for k, v in d.items()
           if re.fullmatch(r"\d+_\d+", k)}
    if len(out) != 25:
        raise AssertionError(f"{design}: expected 25 fold betas, found {len(out)}")
    return out


# ------------------------------------------------------------------------------------------------------------
# donors (bench, metadata, frozen duplicate links)
# ------------------------------------------------------------------------------------------------------------
class Donors:
    """Bench cells as Nd-chord donors: metadata, copy groups, the frozen section-5.1 links and the sensitivity flags."""

    def __init__(self, bench=None, *, log=print):
        from gen20k import data as D
        from gen20k.splits import copy_groups_of
        t0 = time.time()
        self.bench = bench if bench is not None else D.load_bench()
        f = self.bench.frame.reset_index(drop=True)
        self.Y = np.asarray(self.bench.Y, float).copy()                 # V1
        self.N = len(f)
        self.ids = f["cell_id"].astype(str).tolist()
        self.pos = {c: i for i, c in enumerate(self.ids)}
        self.meta = POP.metadata_frame(f)
        self.finite = POP.finite_from_metals(self.meta)                  # label-free coverage
        if not np.array_equal(self.finite, np.isfinite(self.Y)):
            raise AssertionError("metals metadata disagrees with the finite pattern of V1")
        self.pub = self.meta["publication_id"].astype(str).to_numpy()
        self.chem = self.meta["chemotype"].astype(str).to_numpy()
        self.cg = np.asarray(copy_groups_of(f)).astype(str)
        self.pubs = sorted(set(self.pub))
        self.pcode = np.array([self.pubs.index(p) for p in self.pub], dtype=np.int64)
        self.n_pub = len(self.pubs)
        acid = self.meta["cond__acid_concentration_M"].astype(float).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            self.lacid = np.where(np.isfinite(acid), np.log10(np.maximum(acid, ACID_FLOOR_M)), np.nan)
        ukey = pd.Series(self.pub).str.cat(pd.Series(self.chem), sep="|").to_numpy()
        self.unit_code, self.units = pd.factorize(ukey, sort=True)
        uc = pd.Series(self.chem).groupby(self.unit_code).first()
        self.chem_names = sorted(set(self.chem))
        self.unit_chem = np.array([self.chem_names.index(c) for c in uc.sort_index()], dtype=np.int64)
        self.chem_code = np.array([self.chem_names.index(c) for c in self.chem], dtype=np.int64)
        self.c3_ids = POP.c3_cells_from_ref()
        self.c3 = np.isin(np.array(self.ids), self.c3_ids)
        self.v2 = np.isin(np.array(self.ids), sorted(POP.v2drop_set()))
        bt = DU.bit_identity(self.Y, self.ids)
        self.bit_links = DU.flagged_links(bt)
        ofs = DU.offset_scan(self.Y, self.ids, self.bit_links)
        q = DU.quant_scan(self.Y, self.ids)
        self.quant = np.isin(np.array(self.ids), q.cell_id[q.Q_SMALLINT | q.Q_COARSE].astype(str).tolist())
        self.floor = {self.pos[c]: frozenset(LN.index(m) for m in mets) for c, mets in POP.v2_lists()["V2-5"].items()
                      if c in self.pos}
        self.bit = [set() for _ in self.ids]
        for a, b in self.bit_links:
            self.bit[self.pos[a]].add(self.pos[b])
            self.bit[self.pos[b]].add(self.pos[a])
        self.ofs = [set() for _ in self.ids]
        for a, b in zip(ofs.cell_a, ofs.cell_b):
            self.ofs[self.pos[a]].add(self.pos[b])
            self.ofs[self.pos[b]].add(self.pos[a])
        self.c5pub = self.pub == paths.C5
        self.c5cell = np.array(self.ids) == paths.C5_CELL
        self.sc = self.chem == CHEM
        self.chord_meta = self.finite[:, PR] & self.finite[:, ND] & self.finite[:, SM]
        self.seconds = round(time.time() - t0, 1)
        log(f"  donors: {self.N} bench cells, {self.n_pub} publications, {len(set(self.cg))} copy groups, "
            f"{int(self.chord_meta.sum())} cells with Pr/Nd/Sm ({len(set(self.pub[self.chord_meta]))} publications; "
            f"sc009 {int((self.sc & self.chord_meta).sum())} cells / {len(set(self.pub[self.sc & self.chord_meta]))} "
            f"publications), {len(self.c3_ids)} C3 targets, {len(self.bit_links)} bit-flagged links, "
            f"{len(ofs)} offset / near-id links, {int(self.quant.sum())} quantised cells ({self.seconds}s)")

    # ---- values ---------------------------------------------------------------------------------------------
    def values(self, Y=None, variant: str = "PRIMARY"):
        """(Nd chord, Pr/Nd log SF) per bench cell from Y (V1 by default); V25 voids floor readings."""
        Y = self.Y if Y is None else np.asarray(Y, float)
        ch, sf = nd_chord(Y), prnd_sf(Y)
        if variant == "V25":
            for i, mets in self.floor.items():
                if mets & {PR, ND, SM}:
                    ch[i] = np.nan
                if mets & {PR, ND}:
                    sf[i] = np.nan
        return ch, sf

    # ---- donor sets -----------------------------------------------------------------------------------------
    def donor_idx(self, i: int, *, variant: str = "PRIMARY", c5: str = PRIMARY_C5, lab=None, extra_excl=None,
                  leaky: bool = False) -> np.ndarray:
        """Donor cell indices of target i: another publication and another copy group, never i, minus bit-flagged
        links (all variants but NB) and the variant's / C5 stratum's exclusions; ``lab`` (cells,) restricts to
        lab == lab[i].  ``leaky`` drops the publication / copy-group / self exclusion (positive control only)."""
        ok = np.ones(self.N, bool)
        if not leaky:
            ok &= (self.pub != self.pub[i]) & (self.cg != self.cg[i])
            ok[i] = False
        if variant != "NB" and self.bit[i]:
            ok[sorted(self.bit[i])] = False
        if variant == "OFS" and self.ofs[i]:
            ok[sorted(self.ofs[i])] = False
        elif variant == "QNT":
            ok &= ~self.quant
        elif variant == "V2":
            ok &= ~self.v2
        elif variant == "C3DON":
            ok &= self.c3
        elif variant not in DONOR_VARIANTS:
            raise KeyError(variant)
        if c5 == "C5_out":
            ok &= ~self.c5pub
        elif c5 == "C5_cell_out":
            ok &= ~self.c5cell
        elif c5 != "C5_in":
            raise KeyError(c5)
        if extra_excl is not None:
            ok &= ~np.asarray(extra_excl, bool)
        if lab is not None:
            ok &= np.asarray(lab) == np.asarray(lab)[i]
        idx = np.flatnonzero(ok)
        if not leaky:
            self.guard(i, idx, variant)
        return idx

    def guard(self, i: int, idx, variant: str = "PRIMARY") -> None:
        """Refuses a donor set holding the target, a cell of its publication or copy group, or (except NB) a cell
        bit-identical to it under the frozen section-5.1 rule."""
        idx = np.asarray(idx, dtype=np.int64)
        if (idx == i).any():
            raise LeakageError(f"{self.ids[i]}: the target is in its own donor pool")
        same = (self.pub[idx] == self.pub[i]) | (self.cg[idx] == self.cg[i])
        if same.any():
            raise LeakageError(f"{self.ids[i]}: {int(same.sum())} donors share its publication or copy group")
        if variant != "NB" and set(idx.tolist()) & self.bit[i]:
            raise LeakageError(f"{self.ids[i]}: a bit-identical copy is in its donor pool")

    def target_values(self, targets, *, Y=None, variant: str = "PRIMARY", c5: str = PRIMARY_C5, extra_excl=None,
                      leaky: bool = False) -> pd.DataFrame:
        """Per target: the chemotype and universal Nd terms (raw, n, shrunk), the copied Pr/Nd SF, the donor-set
        |mean| / SE gate (DIAGNOSTIC) and the acid distance to the chemotype donors (metadata)."""
        ch, sf = self.values(Y, variant)
        wt = "cell" if variant == "CELLW" else "pub"
        rows = []
        for t in targets:
            i = self.pos[t]
            dc = self.donor_idx(i, variant=variant, c5=c5, lab=self.chem, extra_excl=extra_excl, leaky=leaky)
            da = self.donor_idx(i, variant=variant, c5=c5, lab=None, extra_excl=extra_excl, leaky=leaky)
            uc, nc, mc, sdc = pub_means(ch, dc, self.pcode, self.n_pub, wt, full=True)
            ua, na, ma = pub_means(ch, da, self.pcode, self.n_pub, wt)
            cp, ncp, mcp = pub_means(sf, dc, self.pcode, self.n_pub, wt)
            dch = dc[np.isfinite(ch[dc])]
            ac = pub_means(self.lacid, dch, self.pcode, self.n_pub)[0]
            dla = abs(self.lacid[i] - ac) if np.isfinite(self.lacid[i]) and np.isfinite(ac) else float("nan")
            se = sdc / math.sqrt(nc) if nc > 1 else float("nan")
            rows.append({"cell_id": t, "is_sc009": bool(self.sc[i]), "u_chem_raw": uc, "n_chem_pub": nc,
                         "n_chem_cell": mc, "u_chem": shrink(nc) * uc if nc else 0.0, "se_chem": se,
                         "gate_2se": bool(nc > 1 and np.isfinite(se) and abs(uc) > 2 * se),
                         "u_ndu_raw": ua, "n_ndu_pub": na, "n_ndu_cell": ma, "u_ndu": shrink(na) * ua if na else 0.0,
                         "copy_sf": cp, "n_copy_pub": ncp, "n_copy_cell": mcp, "acid_t": self.lacid[i], "acid_donors": ac,
                         "dla": dla, "acid_stratum": ("NA" if not np.isfinite(dla) else
                                                      ("FAR" if dla > DLOG_ACID else "NEAR"))})
        return pd.DataFrame(rows).set_index("cell_id")


def all_target_values(D: Donors, targets, *, Y=None, configs=CONFIGS) -> dict:
    return {cfg: D.target_values(targets, Y=Y, c5=cfg[0], variant=cfg[1]) for cfg in configs}


# ------------------------------------------------------------------------------------------------------------
# rows (round-4 REF parquet), identity checks, the frozen external object
# ------------------------------------------------------------------------------------------------------------
def build_rows(D: Donors, design: str, *, log=print) -> tuple[pd.DataFrame, dict]:
    """E-A rows of REF_<design> (r5.common.load_rows) with dphi_Gd / dphi_Nd, flags and the identity checks: y from V1,
    B1 recomputed from the fold betas through r4.refs.predict_levels (the round-4 path), B0 through r3.interp, the
    additive form B0 + beta dphi_Gd and the Nd-potential identity B1 + u dphi_Nd = predict_levels(kx_fit) at
    u = U_PROBE.  Real V1 is used for the checks only (identities, no score)."""
    t0 = time.time()
    R = CM.load_rows(design)
    targets = D.c3_ids
    if set(R.cell_id.astype(str)) != set(targets):
        raise AssertionError(f"{design}: REF cells differ from the C3 targets")
    betas = fold_betas(design)
    n = len(R)
    dG, dN, b1r, b0r, kxr = (np.full(n, np.nan) for _ in range(5))
    beta = np.full(n, np.nan)
    rs_bad = 0
    for (s, t), g in R.groupby(["split_seed", "cell_id"], sort=False):
        i = D.pos[t]
        idx = g.index.to_numpy()
        if g.read_set.nunique() != 1 or g.fold.nunique() != 1:
            rs_bad += 1
        read = reads_of(g.read_set.iloc[0])
        metals = tuple(int(m) for m in np.flatnonzero(D.finite[i]))
        pairs = list(zip(g.ia.astype(int), g.ib.astype(int)))
        b = betas[(int(s), int(g.fold.iloc[0]))]
        ym = masked(D.Y[i], read)
        dG[idx] = dphi_pairs(read, metals, pairs, GD)
        dN[idx] = dphi_pairs(read, metals, pairs, ND)
        b1r[idx] = pair_sf(levels(b1_fit(b), ym, metals), pairs)
        b0r[idx] = pair_sf(interp.pwl_levels(ym, metals, "R"), pairs)
        kxr[idx] = pair_sf(levels(kx_fit(b, U_PROBE), ym, metals), pairs)
        beta[idx] = b
    yl = D.Y[R.cell_id.map(D.pos).to_numpy(int), R.ia.to_numpy(int)] - D.Y[R.cell_id.map(D.pos).to_numpy(int),
                                                                            R.ib.to_numpy(int)]
    R["dphi_Gd"], R["dphi_Nd"], R["beta"] = dG, dN, beta
    R["is_sc009"] = R.chemotype.astype(str).to_numpy() == CHEM
    R["is_prnd"] = (R.A.astype(str).to_numpy() == "Pr") & (R.B.astype(str).to_numpy() == "Nd")
    R["tix"] = R.cell_id.map({t: r for r, t in enumerate(targets)}).to_numpy(int)
    zd = np.abs(Z_VEC[R.ia.to_numpy(int)].astype(int) - Z_VEC[R.ib.to_numpy(int)].astype(int))
    per_seed = R.groupby("split_seed").size()
    e1 = R[R.is_sc009 & R.is_prnd].groupby("split_seed").size()
    b1p, b0p = R.B1.to_numpy(float), R.B0.to_numpy(float)
    checks = {"design": design, "rows": int(n), "rows_per_seed": {int(k): int(v) for k, v in per_seed.items()},
              "cells": int(R.cell_id.nunique()), "readset_or_fold_inconsistent_groups": int(rs_bad),
              "y_maxabs": float(np.max(np.abs(yl - R.y.to_numpy(float)))),
              "b1_predict_levels_maxabs": float(np.max(np.abs(b1r - b1p))),
              "b1_additive_maxabs": float(np.max(np.abs(b0p + beta * dG - b1p))),
              "b0_maxabs": float(np.max(np.abs(b0r - b0p))),
              "kx_additive_identity_maxabs": float(np.max(np.abs(b1r + U_PROBE * dN - kxr))),
              "both_ends_read_rows": int((R.a_read & R.b_read).sum()),
              "adjacent_lighter_first": bool((zd == 1).all() and (R.ia < R.ib).all()),
              "dphi_nd_nonzero_rows": int((dN != 0).sum()),
              "dphi_nd_on_cells_without_nd": int(((dN != 0) & ~D.finite[R.cell_id.map(D.pos).to_numpy(int), ND]).sum()),
              "sc009_prnd_rows_per_seed": {int(k): int(v) for k, v in e1.items()},
              "seconds": round(time.time() - t0, 1)}
    log(f"  {design}: {n} rows, {checks['cells']} cells; identity B1 {checks['b1_predict_levels_maxabs']:.1e} "
        f"(additive {checks['b1_additive_maxabs']:.1e}), B0 {checks['b0_maxabs']:.1e}, y {checks['y_maxabs']:.1e}, "
        f"Nd-potential form {checks['kx_additive_identity_maxabs']:.1e}; sc009 Pr/Nd rows per seed "
        f"{sorted(set(checks['sc009_prnd_rows_per_seed'].values()))} ({checks['seconds']}s)")
    return R, checks


def frozen_external(D: Donors, sc: dict | None = None) -> dict:
    """The frozen external object: u_ext = n / (n + 5) * (-0.11), n = V1 sc009 publications reporting Pr, Nd and Sm
    (metadata count; no label).  The source gives "about -0.11" without its computation; ``sc`` (chord_tables'
    sc009 consistency line on V1) adds the computed V1 means to the provenance only, never to the frozen value."""
    pubs = sorted(set(D.pub[D.sc & D.chord_meta]))
    n = len(pubs)
    w = shrink(n)
    prov = U_RAW_SOURCE
    if sc:
        prov += (f"; V1 publication-first mean {sc['u_raw']:.3f}, cell-weighted {sc['u_raw_cell_weighted']:.3f} "
                 f"(computed; the 2-decimal consistency line {'holds' if sc['consistent_2dp'] else 'FAILED'}, "
                 "results/EKX/chord_sc009.json); -0.11 and u_ext stay as frozen (brief 6.8)")
    return {"label": "FROZEN (item 1.8 code seal; external leg value)", "chemotype": CHEM, "u_raw": U_RAW,
            "K": K_SHRINK, "n_donor_publications": n, "donor_publications": pubs, "w": w, "u_ext": w * U_RAW,
            "definition": "Nd chord a_Nd = y_Nd - [(1 - w) y_Pr + w y_Sm] along -Shannon CN8 radius, w = "
                          f"{W_SM:.6f}; u_raw = {prov}",
            "apply": "level(m) = PWL_R[y_S - u_S](m) + u(m), u = beta_frozen 1[Gd] + u_ext 1[Nd] (r4.refs coherent "
                     "potentials; reads exact); r5.ekx.external_pairs",
            "rider": RIDER_RULE, "source": "brief 6.8; IR M07; ideas/critiques.json M07 test design"}


def external_pairs(values_at_reads: dict, metals, pairs, beta: float, u_nd: float) -> dict:
    """B1 and B1 + Nd potential pair predictions for a new cell from its legal reads only ({metal: log D})."""
    ym = np.full(N_LN, np.nan)
    for m, v in values_at_reads.items():
        ym[LN.index(m) if isinstance(m, str) else int(m)] = float(v)
    mets = tuple(sorted(LN.index(m) if isinstance(m, str) else int(m) for m in metals))
    return {"B1": pair_sf(levels(b1_fit(beta), ym, mets), pairs), "KX": pair_sf(levels(kx_fit(beta, u_nd), ym, mets), pairs)}


def external_rider(pubs, *, beta: float, u_nd: float, first: int = 4) -> dict:
    """The section-6.8 rider on new sc009 publications in arrival order.  ``pubs``: [{"publication": str, "cells":
    [{"y": (14,) full series, NaN = not measured, "read_set": the frozen external 4-read set (metal indices)}]}].
    Eligible: >= 1 cell with Pr, Nd and Sm.  Per publication: mean Nd chord; mean Pr/Nd squared error of B1 and of
    the frozen arm over cells whose Pr/Nd pair is scored (both measured, not both read); 'falls' needs >= 1 scored
    row.  Killed if, over the first 4 eligible publications, >= 2 have a chord >= 0 or fewer than 3 'fall'."""
    rows = []
    for p in pubs:
        ch, e_b1, e_kx = [], [], []
        for c in p["cells"]:
            y = np.asarray(c["y"], float)
            if np.isfinite(y[[PR, ND, SM]]).all():
                ch.append(float(nd_chord(y)[0]))
            read = frozenset(int(m) for m in c["read_set"])
            if np.isfinite(y[[PR, ND]]).all() and not {PR, ND} <= read:
                mets = tuple(int(m) for m in np.flatnonzero(np.isfinite(y)))
                pr = external_pairs({m: y[m] for m in read}, mets, [(PR, ND)], beta, u_nd)
                yt = y[PR] - y[ND]
                e_b1.append(float((yt - pr["B1"][0]) ** 2))
                e_kx.append(float((yt - pr["KX"][0]) ** 2))
        sb, sk = (float(np.mean(e_b1)) if e_b1 else float("nan")), (float(np.mean(e_kx)) if e_kx else float("nan"))
        rows.append({"publication": p["publication"], "eligible": bool(ch), "n_chord_cells": len(ch),
                     "chord_mean": float(np.mean(ch)) if ch else float("nan"),
                     "chord_nonneg": bool(ch and np.mean(ch) >= 0), "n_prnd_rows": len(e_b1), "sqerr_B1": sb,
                     "sqerr_KX": sk, "falls": bool(e_b1 and sk < sb)})
    el = [r for r in rows if r["eligible"]][:first]
    if len(el) < first:
        verdict, nn, nf = "PENDING", sum(r["chord_nonneg"] for r in el), sum(r["falls"] for r in el)
    else:
        nn, nf = sum(r["chord_nonneg"] for r in el), sum(r["falls"] for r in el)
        verdict = "KILLED" if (nn >= 2 or nf < 3) else "NOT KILLED"
    return {"rule": RIDER_RULE, "verdict": verdict, "n_eligible_counted": len(el), "n_chord_nonneg": int(nn),
            "n_falls": int(nf), "publications": rows, "label": "external rider (not a package claim; CP-13)"}


# ------------------------------------------------------------------------------------------------------------
# arms
# ------------------------------------------------------------------------------------------------------------
def offsets(D: Donors, R: pd.DataFrame, *, betas: dict, Y=None, c5: str = PRIMARY_C5, variant: str = "PRIMARY",
            extra_excl=None, only=None) -> pd.DataFrame:
    """PRND_OFFSET per (seed, target with a Pr/Nd row): same-chemotype C3 donors with a scored Pr/Nd row in the same
    seed; e = (Y_Pr - Y_Nd) - B0 - beta_t dphi_Gd (beta_t = the target's fold beta); n / (n + 5) * mean_pub(e)."""
    Y = D.Y if Y is None else np.asarray(Y, float)
    out = []
    for s, g in R[R.is_prnd].groupby("split_seed", sort=True):
        a, gg = np.full(D.N, np.nan), np.full(D.N, np.nan)
        ci = g.cell_id.map(D.pos).to_numpy(int)
        a[ci] = (Y[ci, PR] - Y[ci, ND]) - g.B0.to_numpy(float)
        gg[ci] = g.dphi_Gd.to_numpy(float)
        for t, fo in zip(g.cell_id.astype(str), g.fold.astype(int)):
            if only is not None and t not in only:
                continue
            i = D.pos[t]
            b = betas[(int(s), int(fo))]
            idx = D.donor_idx(i, variant=variant, c5=c5, lab=D.chem, extra_excl=extra_excl)
            raw, n, m = pub_means(a - b * gg, idx, D.pcode, D.n_pub)
            out.append({"split_seed": int(s), "cell_id": t, "off_raw": raw, "n_off_pub": n, "n_off_cell": m,
                        "off": shrink(n) * raw if n else 0.0})
    return pd.DataFrame(out, columns=["split_seed", "cell_id", "off_raw", "n_off_pub", "n_off_cell", "off"])


def sfx(c5: str, var: str = "PRIMARY") -> str:
    return f"@{c5}" + ("" if var == "PRIMARY" else f"~{var}")


def add_arms(R: pd.DataFrame, TV: dict, OFF: dict, u_ext: float) -> pd.DataFrame:
    """Arm columns <arm>@<c5>[~<variant>] on the row frame (in place; also returned)."""
    b1, dn = R.B1.to_numpy(float), R.dphi_Nd.to_numpy(float)
    sc, pn = R.is_sc009.to_numpy(bool), R.is_prnd.to_numpy(bool)
    cells = R.cell_id.astype(str)
    new = {}
    for (c5, var), tv in TV.items():
        x = sfx(c5, var)
        uc = cells.map(tv.u_chem).to_numpy(float)
        ua = cells.map(tv.u_ndu).to_numpy(float)
        new["KX_CHEM" + x] = b1 + uc * dn
        new["KX_SC009" + x] = b1 + np.where(sc, uc, 0.0) * dn
        new["NDU" + x] = b1 + ua * dn
        new["u_chem" + x], new["n_chem_pub" + x], new["u_ndu" + x] = uc, cells.map(tv.n_chem_pub).to_numpy(float), ua
        if var != "PRIMARY":
            continue
        o = OFF[c5].set_index(["split_seed", "cell_id"])
        key = pd.MultiIndex.from_arrays([R.split_seed.astype(int), cells])
        off = o.off.reindex(key).to_numpy(float)
        if np.isnan(off[pn]).any():
            raise AssertionError(f"{c5}: a Pr/Nd row has no offset record")
        new["PRND_OFFSET" + x] = b1 + np.where(pn, np.nan_to_num(off), 0.0)
        new["n_off_pub" + x] = np.where(pn, o.n_off_pub.reindex(key).to_numpy(float), np.nan)
        cp = cells.map(tv["copy_sf"]).to_numpy(float)
        new["PRND_COPY" + x] = np.where(pn & np.isfinite(cp), cp, b1)
        new["KX_FROZEN" + x] = b1 + np.where(sc, u_ext, 0.0) * dn
        new["acid_stratum" + x] = cells.map(tv.acid_stratum).to_numpy(object)
        new["dla" + x] = cells.map(tv.dla).to_numpy(float)
    for k, v in new.items():
        R[k] = v
    return R


BASE_COLS = ("split_seed", "fold", "cell_id", "A", "B", "ia", "ib", "y", "B1", "B0", "dphi_Gd", "dphi_Nd", "tix",
             "publication", "extractant", "chemotype", "copy_group", "cluster_cpg", "pubchem", "is_sc009", "is_prnd",
             "read_set", "a_read", "b_read")


def endpoint_mask(R: pd.DataFrame, ep: str) -> np.ndarray:
    if ep == "SC009_PRND":
        return (R.is_sc009 & R.is_prnd).to_numpy(bool)
    if ep == "POOLED":
        return np.ones(len(R), bool)
    if ep == "PRND_ALL":
        return R.is_prnd.to_numpy(bool)
    if ep == "SC009_ALL":
        return R.is_sc009.to_numpy(bool)
    raise KeyError(ep)


def c5_rows(R: pd.DataFrame, c5: str) -> np.ndarray:
    if c5 == "C5_in":
        return np.ones(len(R), bool)
    if c5 == "C5_out":
        return (R.publication.astype(str) != paths.C5).to_numpy(bool)
    if c5 == "C5_cell_out":
        return (R.cell_id.astype(str) != paths.C5_CELL).to_numpy(bool)
    raise KeyError(c5)


def view(R: pd.DataFrame, *, c5: str = PRIMARY_C5, var: str = "PRIMARY", endpoint: str = "POOLED",
         extra_rows=None) -> pd.DataFrame:
    """Rows of one (C5 stratum, donor variant, endpoint) with bare arm columns (identical rows for every arm)."""
    m = endpoint_mask(R, endpoint) & c5_rows(R, c5)
    if extra_rows is not None:
        m &= np.asarray(extra_rows, bool)
    V = R.loc[m, [c for c in BASE_COLS if c in R.columns]].copy()
    x = sfx(c5, var)
    for a in ARMS[1:]:
        if a + x in R.columns:
            V[a] = R.loc[m, a + x].to_numpy(float)
    if "acid_stratum" + sfx(c5) in R.columns:
        V["acid_stratum"] = R.loc[m, "acid_stratum" + sfx(c5)].to_numpy(object)
    return V.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------------
# random-chemotype permutation
# ------------------------------------------------------------------------------------------------------------
def perm_labels(D: Donors, B: int, key: str = "EKX|PERM_CHEM") -> tuple[np.ndarray, int]:
    """(B, cells) chemotype codes after shuffling the chemotype labels over publication x chemotype units (the
    number of units per chemotype is kept); RNG default_rng(paths.seed_for(key)).  Design-independent."""
    seed = paths.seed_for(key)
    rng = np.random.default_rng(seed)
    nu = len(D.unit_chem)
    out = np.empty((B, D.N), dtype=np.int64)
    for b in range(B):
        out[b] = D.unit_chem[rng.permutation(nu)][D.unit_code]
    return out, seed


def perm_u(D: Donors, targets, labs: np.ndarray, *, Y=None, c5: str = PRIMARY_C5) -> np.ndarray:
    """(B, targets) shrunk chemotype Nd term under each relabelling (primary donor rule, same C5 stratum)."""
    ch, _ = D.values(Y)
    cand = []
    for t in targets:
        i = D.pos[t]
        idx = D.donor_idx(i, c5=c5)
        cand.append((i, idx[np.isfinite(ch[idx])]))
    U = np.zeros((len(labs), len(cand)))
    for b, lab in enumerate(labs):
        for r, (i, idx) in enumerate(cand):
            sel = idx[lab[idx] == lab[i]]
            if len(sel):
                raw, n, _ = pub_means(ch, sel, D.pcode, D.n_pub)
                U[b, r] = shrink(n) * raw
    return U


def perm_eval(V: pd.DataFrame, U: np.ndarray, *, sc_only: bool, chunk: int = 200) -> np.ndarray:
    """Seed-mean dR^2 of B1 + u dphi_Nd over B1 on V's rows for every row of U (B, targets)."""
    y, b1 = V.y.to_numpy(float), V.B1.to_numpy(float)
    dn, tix = V.dphi_Nd.to_numpy(float), V.tix.to_numpy(int)
    if sc_only:
        dn = dn * V.is_sc009.to_numpy(float)
    sc, inv = np.unique(V.split_seed.to_numpy(), return_inverse=True)
    H = np.zeros((len(y), len(sc)))
    H[np.arange(len(y)), inv] = 1.0
    r0 = y - b1
    sst = np.array([((y[inv == k] - y[inv == k].mean()) ** 2).sum() for k in range(len(sc))])
    sse0 = (r0 ** 2) @ H
    out = np.empty(len(U))
    for a in range(0, len(U), chunk):
        E = r0[None, :] - U[a:a + chunk][:, tix] * dn[None, :]
        out[a:a + chunk] = (((sse0[None, :] - (E * E) @ H) / sst[None, :])).mean(1)
    return out


def permutation_tables(R: pd.DataFrame, U: dict, labs_seed: int, design: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed vs null seed-mean dR^2 for KX_SC009 on SC009_PRND and KX_CHEM on POOLED; p = (1 + #{null >= obs}) /
    (1 + B).  The observed value is recomputed by perm_eval from the arm's own u (must equal the r5.common value)."""
    rows, draws = [], []
    for c5, Uc in U.items():
        for ep, arm, sc_only in (("SC009_PRND", "KX_SC009", True), ("POOLED", "KX_CHEM", False)):
            V = view(R, c5=c5, endpoint=ep)
            u_obs = np.zeros(int(R.tix.max()) + 1)
            uc = R.drop_duplicates("tix").set_index("tix")["u_chem" + sfx(c5)]
            u_obs[uc.index.to_numpy(int)] = uc.to_numpy(float)
            obs = float(perm_eval(V, u_obs[None, :], sc_only=sc_only)[0])
            ref = CM.r2_seedmean(V, arm) - CM.r2_seedmean(V, "B1")
            if abs(obs - ref) > 1e-10:
                raise AssertionError(f"permutation observed {obs} != r5.common {ref}")
            null = perm_eval(V, Uc, sc_only=sc_only)
            ge = int((null >= obs - 1e-12).sum())
            rows.append({"design": design, "c5": c5, "endpoint": ep, "arm": arm, "stat": "dR2 vs B1 (seed mean)",
                         "observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                         "null_q95": float(np.quantile(null, 0.95)), "null_max": float(null.max()), "n_ge": ge,
                         "B": int(len(null)), "p": (1.0 + ge) / (1.0 + len(null)), "rng_seed": labs_seed,
                         "label": f"{DIAG}; {SEEN}"})
            draws.append(pd.DataFrame({"design": design, "c5": c5, "endpoint": ep, "arm": arm,
                                       "draw": np.arange(len(null)), "dR2_null": null, "label": f"{DIAG}; {SEEN}"}))
    return pd.DataFrame(rows), pd.concat(draws, ignore_index=True)


# ------------------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------------------
def label_for(design: str, c5: str, var: str = "PRIMARY", arm: str | None = None) -> str:
    if arm == "KX_FROZEN":
        return f"{DIAG} (not fold-independent); {SEEN}"
    if design == PRIMARY_DESIGN and c5 == PRIMARY_C5 and var == "PRIMARY":
        return SEEN
    return f"{SENS}; {SEEN}"


def design_label(design: str) -> str:
    """Top-level label of a design's outputs (summary, per-item table, rows): BP is this item's registered primary
    base, the BPG base a SENSITIVITY (REGISTRY 1.8-S03)."""
    return SEEN if design == PRIMARY_DESIGN else f"{SENS}; {SEEN}"


def _bare(df: pd.DataFrame) -> pd.DataFrame:
    """Row labels without the E-KX-LOPO note (the per-item table states it once, in its own label)."""
    return df.assign(label=df.label.astype(str).str.replace(f"; {NOT_NEXT_TO}", "", regex=False))


def mae_seeds(V: pd.DataFrame, col: str) -> pd.Series:
    return pd.Series({s: float(np.abs(g.y - g[col]).mean()) for s, g in V.groupby("split_seed")}, dtype=float)


def _ps(s: pd.Series) -> str:
    return ";".join(f"{v:.4f}" for v in s)


def arm_rows(V: pd.DataFrame, arms, meta: dict, design: str, *, balanced: bool) -> list:
    out = []
    ns = V.split_seed.nunique()
    for a in arms:
        if a not in V.columns:
            continue
        r2 = CM.r2_seeds(V, a)
        ms = mae_seeds(V, a)
        lab = label_for(design, meta.get("c5", PRIMARY_C5), meta.get("variant", "PRIMARY"), a)
        out.append({**meta, "arm": a, "R2": float(r2.mean()), "MSE": CM.mse_seedmean(V, a),
                    "relSSE_vs_B1": CM.rel_sse(V, a, "B1"), "MAE": float(ms.mean()),
                    "MAE_balanced": CM.bal_mae(V, a) if balanced else float("nan"), "R2_per_seed": _ps(r2),
                    "MAE_per_seed": _ps(ms), "rows_per_seed": len(V) / ns if ns else 0.0,
                    "cells": int(V.cell_id.nunique()), "publications": int(V.publication.nunique()),
                    "label": f"{lab}; {NOT_NEXT_TO}"})                 # absolute R2: the E-KX-LOPO note on every row
    return out


def contrast_rows(V: pd.DataFrame, pairs, meta: dict, design: str, *, balanced: bool) -> list:
    out = []
    for cand, ref in pairs:
        if cand not in V.columns or ref not in V.columns:
            continue
        d = CM.r2_seeds(V, cand) - CM.r2_seeds(V, ref)
        dm = mae_seeds(V, ref) - mae_seeds(V, cand)
        lab = label_for(design, meta.get("c5", PRIMARY_C5), meta.get("variant", "PRIMARY"),
                        "KX_FROZEN" if "KX_FROZEN" in (cand, ref) else cand)
        out.append({**meta, "cand": cand, "ref": ref, "dR2": float(d.mean()), "dR2_per_seed": _ps(d),
                    "seeds_positive": int((d > 0).sum()), "relSSE": CM.rel_sse(V, cand, ref), "dMAE": float(dm.mean()),
                    "dMAE_balanced": CM.dbal_mae(V, cand, ref) if balanced else float("nan"),
                    "label": f"{lab}; {NOT_NEXT_TO}"})
    return out


UNIT_SPECS = (("SC009_PRND", "publication", "KX_SC009", "B1"), ("SC009_PRND", "publication", "KX_SC009", "NDU"),
              ("POOLED", "cluster_cpg", "KX_CHEM", "B1"), ("POOLED", "publication", "KX_CHEM", "B1"),
              ("POOLED", "publication", "KX_CHEM", "NDU"))
LOPO_SPECS = (("SC009_PRND", "KX_SC009", "B1"), ("SC009_PRND", "KX_SC009", "NDU"), ("POOLED", "KX_CHEM", "B1"),
              ("POOLED", "KX_CHEM", "NDU"), ("POOLED", "NDU", "B1"))


def unit_tables(R: pd.DataFrame, design: str, c5s=("C5_out", "C5_in")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section-9.2 unit table and the section-4.2 tests (DIAGNOSTIC; publication sign-flip over sc009 publications on
    the primary endpoint; exact enumeration at G <= 13)."""
    U, Ts = [], []
    for c5 in c5s:
        for ep, unit, cand, ref in UNIT_SPECS:
            V = view(R, c5=c5, endpoint=ep)
            us = CM.unit_stats(V, cand, ref, unit)
            key = f"EKX|{design}|{c5}|{ep}|{unit}|{cand}-{ref}"
            te = CM.unit_tests(us, key)
            u = us.reset_index().rename(columns={unit: "unit_id"})
            u.insert(0, "design", design)
            u.insert(1, "c5", c5)
            u.insert(2, "endpoint", ep)
            u.insert(3, "unit", unit)
            u.insert(4, "cand", cand)
            u.insert(5, "ref", ref)
            u["label"] = f"{DIAG}; {SEEN}"
            U.append(u)
            sl = te["SF_LOG"]
            Ts.append({"design": design, "c5": c5, "endpoint": ep, "unit": unit, "cand": cand, "ref": ref, "G": te["G"],
                       "W/L/T": te["W/L/T"], "SF_POOL_T": te["SF_POOL"]["T"], "SF_POOL_p": te["SF_POOL"]["p"],
                       "SF_POOL_exact": te["SF_POOL"]["exact"], "SF_POOL_n_flips": te["SF_POOL"]["n_flips"],
                       "SF_POOL_seed": te["SF_POOL"]["seed"], "SIGN_p": te["SIGN"]["p"],
                       "SF_LOG_p": sl["p"] if sl else float("nan"), "IM_REL_t": te["IM_REL"]["t"],
                       "IM_REL_p": te["IM_REL"]["p"], "top3_share": CM.top3_share(us.dSSE),
                       "label": f"{DIAG}; {SEEN}; no in-corpus p-value is called significant"})
    return pd.concat(U, ignore_index=True), pd.DataFrame(Ts)


def lopo_table(R: pd.DataFrame, design: str, c5s=("C5_out", "C5_in")) -> pd.DataFrame:
    rows = []
    for c5 in c5s:
        for ep, cand, ref in LOPO_SPECS:
            V = view(R, c5=c5, endpoint=ep)
            L = CM.lopo(V, lambda d, c=cand, r=ref: CM.r2_seedmean(d, c) - CM.r2_seedmean(d, r), by="publication")
            us = CM.unit_stats(V, cand, ref, "publication")
            rows.append({"design": design, "c5": c5, "endpoint": ep, "cand": cand, "ref": ref, "stat": "dR2 (seed mean)",
                         "point": L["point"], "se": L["se"], "lo": L["lo"], "hi": L["hi"], "P": L["P"],
                         "n_undefined": L["n_undefined"], "min": L["min"], "min_by": L["min_by"], "max": L["max"],
                         "max_by": L["max_by"], "top3_share": CM.top3_share(us.dSSE),
                         "top3_publications": ",".join(us.dSSE.sort_values(ascending=False).index[:TOP].astype(str)),
                         "label": label_for(design, c5)})
    return pd.DataFrame(rows)


def _top3(V: pd.DataFrame, cand: str) -> tuple[list, float, pd.DataFrame]:
    us = CM.unit_stats(V, cand, "B1", "publication")
    o = us.assign(_u=us.index.astype(str)).sort_values(["dSSE", "_u"], ascending=[False, True])
    return o.index[:TOP].astype(str).tolist(), CM.top3_share(us.dSSE), us


def rule_row(V: pd.DataFrame, *, cand: str, design: str, c5: str, endpoint: str, removal: str, top: list,
             share: float, primary: bool) -> dict:
    W = V[~V.publication.astype(str).isin(top)]
    dc = CM.r2_seedmean(W, cand) - CM.r2_seedmean(W, "B1")
    du = CM.r2_seedmean(W, "NDU") - CM.r2_seedmean(W, "B1")
    stat = CM.r2_seedmean(W, cand) - CM.r2_seedmean(W, "NDU")
    return {"design": design, "c5": c5, "endpoint": endpoint, "cand": cand, "removal": removal,
            "top3_publications": ",".join(top), "top3_share_of_gain": share,
            "dR2_cand_full": CM.r2_seedmean(V, cand) - CM.r2_seedmean(V, "B1"),
            "dR2_NDU_full": CM.r2_seedmean(V, "NDU") - CM.r2_seedmean(V, "B1"),
            "dR2_cand_without_top3": dc, "dR2_NDU_without_top3": du, "rule_stat": stat,
            "rows_per_seed_without": len(W) / max(W.split_seed.nunique(), 1), "narrowed": bool(stat <= 0),
            "outcome": ("narrowed to 'universal Nd term'" if stat <= 0 else "chemotype Nd term not narrowed"),
            "role": "REPORT RULE (primary)" if primary else "rule sensitivity",
            "label": (f"{CC}; {SEEN}" if primary else f"{CC}; {SENS}; {SEEN}")}


def top3_rule(R: pd.DataFrame, design: str, D: Donors, targets, *, Y=None) -> pd.DataFrame:
    """The section-6.8 report rule and its sensitivities (rows removed; rows and donors removed)."""
    rows = []
    prim = design == PRIMARY_DESIGN
    for c5, ep, cand in (("C5_out", "POOLED", "KX_CHEM"), ("C5_in", "POOLED", "KX_CHEM"),
                         ("C5_out", "SC009_PRND", "KX_SC009")):
        V = view(R, c5=c5, endpoint=ep)
        top, share, _ = _top3(V, cand)
        rows.append(rule_row(V, cand=cand, design=design, c5=c5, endpoint=ep, removal="rows", top=top, share=share,
                             primary=prim and c5 == "C5_out" and ep == "POOLED"))
        if c5 == "C5_out":                     # donors removed as well: recompute u without the top-3 publications
            ex = np.isin(D.pub, top)
            tv = D.target_values(targets, Y=Y, c5=c5, extra_excl=ex)
            cells = V.cell_id.astype(str)
            uc, ua = cells.map(tv.u_chem).to_numpy(float), cells.map(tv.u_ndu).to_numpy(float)
            W = V.copy()
            dn = W.dphi_Nd.to_numpy(float)
            W["KX_CHEM"] = W.B1 + uc * dn
            W["KX_SC009"] = W.B1 + np.where(W.is_sc009, uc, 0.0) * dn
            W["NDU"] = W.B1 + ua * dn
            rows.append(rule_row(W, cand=cand, design=design, c5=c5, endpoint=ep, removal="rows and donors", top=top,
                                 share=share, primary=False))
    return pd.DataFrame(rows)


def acid_table(R: pd.DataFrame, design: str) -> pd.DataFrame:
    """Stratification by |dlog10 acid| > 0.5 between the target and its chemotype donors (metadata; C5_out)."""
    out = []
    for ep, cand in (("SC009_PRND", "KX_SC009"), ("POOLED", "KX_CHEM")):
        V = view(R, c5=PRIMARY_C5, endpoint=ep)
        for st in ("NEAR", "FAR", "NA"):
            W = V[V.acid_stratum == st]
            meta = {"design": design, "c5": PRIMARY_C5, "endpoint": ep, "acid_stratum": st,
                    "rows_per_seed": len(W) / max(W.split_seed.nunique(), 1), "cells": int(W.cell_id.nunique())}
            if len(W) == 0 or W.split_seed.nunique() == 0:
                out.append({**meta, "cand": cand, "ref": "B1", "dR2": float("nan"), "label": f"{SENS}; {SEEN}"})
                continue
            for c, r in ((cand, "B1"), ("NDU", "B1"), (cand, "NDU")):
                d = CM.r2_seeds(W, c) - CM.r2_seeds(W, r)
                out.append({**meta, "cand": c, "ref": r, "R2_cand": CM.r2_seedmean(W, c), "R2_ref": CM.r2_seedmean(W, r),
                            "dR2": float(d.mean()), "dR2_per_seed": _ps(d), "relSSE": CM.rel_sse(W, c, r),
                            "dMAE": float((mae_seeds(W, r) - mae_seeds(W, c)).mean()), "label": f"{SENS}; {SEEN}"})
    return pd.DataFrame(out)


def dup_table(R: pd.DataFrame, design: str) -> pd.DataFrame:
    out = []
    for var in DONOR_VARIANTS:
        for ep in ("SC009_PRND", "POOLED"):
            V = view(R, c5=PRIMARY_C5, var=var, endpoint=ep)
            for c, r in (("KX_SC009", "B1"), ("KX_CHEM", "B1"), ("NDU", "B1"), ("KX_CHEM", "NDU")):
                d = CM.r2_seeds(V, c) - CM.r2_seeds(V, r)
                out.append({"design": design, "c5": PRIMARY_C5, "variant": var, "endpoint": ep, "cand": c, "ref": r,
                            "R2_cand": CM.r2_seedmean(V, c), "R2_ref": CM.r2_seedmean(V, r), "dR2": float(d.mean()),
                            "dR2_per_seed": _ps(d), "relSSE": CM.rel_sse(V, c, r),
                            "label": label_for(design, PRIMARY_C5, var)})
    return pd.DataFrame(out)


def implied_sf_table(R: pd.DataFrame, design: str) -> pd.DataFrame:
    """Mean predicted vs observed sc009 Pr/Nd log SF and the implied Nd/Pr SF = 10^(-mean) (DIAGNOSTIC)."""
    V = view(R, c5=PRIMARY_C5, endpoint="SC009_PRND")
    out = [{"design": design, "arm": "observed", "mean_logSF_PrNd": float(V.groupby("split_seed").y.mean().mean())}]
    for a in ARMS:
        out.append({"design": design, "arm": a, "mean_logSF_PrNd": float(V.groupby("split_seed")[a].mean().mean())})
    T = pd.DataFrame(out)
    T["implied_NdPr_SF"] = 10.0 ** (-T.mean_logSF_PrNd)
    T["label"] = f"{DIAG}; {SEEN}"
    return T


def population_table(R: pd.DataFrame, design: str) -> pd.DataFrame:
    """Cells, publications, clusters, sc009 publications and rows per seed of every (C5 stratum, endpoint)."""
    out = []
    for c5 in C5_VARIANTS:
        for ep in ENDPOINTS:
            V = view(R, c5=c5, endpoint=ep)
            p = CM.population(V, "publication")
            out.append({"design": design, "c5": c5, "endpoint": ep, **p, "clusters": int(V.cluster_cpg.nunique()),
                        "sc009_publications": int(V[V.is_sc009].publication.nunique()),
                        "label": "METADATA (population construction; no score)"})
    return pd.DataFrame(out)


def all_tables(R: pd.DataFrame, design: str, D: Donors, targets, U: dict, labs_seed: int, *, Y=None) -> dict:
    T = {}
    rows, cons = [], []
    for c5 in C5_VARIANTS:
        for ep in ENDPOINTS:
            V = view(R, c5=c5, endpoint=ep)
            meta = {"design": design, "c5": c5, "variant": "PRIMARY", "endpoint": ep}
            bal = ep in ("POOLED", "SC009_ALL", "PRND_ALL")
            rows += arm_rows(V, ARMS, meta, design, balanced=bal)
            cons += contrast_rows(V, CONTRASTS, meta, design, balanced=bal)
    T["arms"], T["contrasts"] = pd.DataFrame(rows), pd.DataFrame(cons)
    T["population"] = population_table(R, design)
    T["units"], T["unit_tests"] = unit_tables(R, design)
    T["lopo"] = lopo_table(R, design)
    T["top3_rule"] = top3_rule(R, design, D, targets, Y=Y)
    T["permutation"], T["permutation_draws"] = permutation_tables(R, U, labs_seed, design)
    T["acid_strata"] = acid_table(R, design)
    T["dup_sens"] = dup_table(R, design)
    T["implied_sf"] = implied_sf_table(R, design)
    return T


# ------------------------------------------------------------------------------------------------------------
# design-independent descriptive tables (DIAGNOSTIC)
# ------------------------------------------------------------------------------------------------------------
def chord_tables(D: Donors, *, Y=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Nd chord per publication x chemotype unit and per chemotype (publication means first); the sc009 line is the
    consistency check of the frozen -0.11 (seen: negative in 11 of 12 publications)."""
    ch, _ = D.values(Y)
    f = pd.DataFrame({"publication": D.pub, "chemotype": D.chem, "chord": ch})
    f = f[np.isfinite(f.chord)]
    u = f.groupby(["chemotype", "publication"]).chord.agg(["size", "mean"]).reset_index()
    u.columns = ["chemotype", "publication", "n_cells", "chord_mean"]
    u["negative"] = u.chord_mean < 0
    u["label"] = f"{DIAG}; {SEEN}"
    c = u.groupby("chemotype").agg(n_publications=("publication", "size"), n_cells=("n_cells", "sum"),
                                   u_raw=("chord_mean", "mean"), sd=("chord_mean", "std"),
                                   n_negative=("negative", "sum")).reset_index()
    c["se"] = c.sd / np.sqrt(c.n_publications)
    c["gate_2se"] = (c.n_publications > 1) & (c.u_raw.abs() > 2 * c.se)
    c["u_shrunk"] = shrink(c.n_publications.to_numpy(float)) * c.u_raw
    c["label"] = f"{DIAG}; {SEEN}"
    s = c[c.chemotype == CHEM]
    fs = f[f.chemotype == CHEM]
    sc = {"label": f"{CC}; {SEEN}", "chemotype": CHEM, "n_publications": int(s.n_publications.iloc[0]) if len(s) else 0,
          "n_negative": int(s.n_negative.iloc[0]) if len(s) else 0,
          "u_raw": float(s.u_raw.iloc[0]) if len(s) else float("nan"), "frozen_u_raw": U_RAW,
          "consistent_2dp": bool(len(s) and abs(float(s.u_raw.iloc[0]) - U_RAW) <= 0.005),
          "u_raw_cell_weighted": float(fs.chord.mean()) if len(fs) else float("nan"), "n_cells": int(len(fs)),
          "seen": "negative in 11 of 12 publications, anomaly about -0.11 [IR M07]",
          "note": (f"u_raw (publication means first) is the registered line; the frozen value's source reads "
                   f"'{U_RAW_SOURCE}'; the cell-weighted mean is descriptive ({DIAG})")}
    return u, c, sc


def consistency_table(cons: dict, designs=None) -> pd.DataFrame:
    """The registered consistency lines (ITEM_SPEC 'Reproduction targets'; they never stop the item), one row each:
    sc009 Pr/Nd rows per seed per design; the V1 sc009 publication-first mean Nd chord vs the frozen -0.11 at 2
    decimals and its negative count; the cell-weighted mean (descriptive).  ``cons`` = reproduction()['consistency']."""
    out = []
    for d, v in cons.items():
        if d == "frozen_value_source" or not v or (designs is not None and d not in designs):
            continue
        got = sorted(set(v.get("sc009_prnd_rows_per_seed", {}).values()))
        out.append({"line": f"sc009 Pr/Nd rows per seed ({d})", "observed": ", ".join(str(x) for x in got),
                    "reference": f"{v.get('seen')} (seen)", "result": "match" if v.get("match") else "MISMATCH",
                    "label": f"{CC}; {SEEN}"})
    sc = cons.get("frozen_value_source") or {}
    if sc:
        out.append({"line": "V1 sc009 mean Nd chord, publication means first, vs the frozen -0.11 (2 decimals)",
                    "observed": f"{sc['u_raw']:.4f}", "reference": f"{U_RAW} (frozen; {U_RAW_SOURCE})",
                    "result": ("consistent (2 decimals)" if sc["consistent_2dp"] else
                               "FAILED (consistent_2dp = false; -0.11 and u_ext stay as frozen)"),
                    "label": f"{CC}; {SEEN}"})
        neg = f"{sc['n_negative']} of {sc['n_publications']}"
        out.append({"line": "sc009 publications with a negative mean Nd chord", "observed": neg,
                    "reference": f"{SEEN_VALUES['sc009_chord_negative_publications']} (seen)",
                    "result": "match" if neg == SEEN_VALUES["sc009_chord_negative_publications"] else "MISMATCH",
                    "label": f"{CC}; {SEEN}"})
        out.append({"line": "V1 sc009 mean Nd chord, cell-weighted (descriptive; not a registered line)",
                    "observed": f"{sc['u_raw_cell_weighted']:.4f}", "reference": f"{U_RAW} (frozen)",
                    "result": "descriptive", "label": f"{DIAG}; {SEEN}"})
    return pd.DataFrame(out, columns=["line", "observed", "reference", "result", "label"])


def split_half(D: Donors, *, Y=None, n_split: int = N_SPLITHALF, key: str = "EKX|SPLITHALF") -> dict:
    """Split-half reliability of chemotype-mean Nd chords (publications of each chemotype with >= 2 chord publications
    split at random into halves; Pearson r of the half means across chemotypes; Spearman-Brown 2r / (1 + r)).
    DIAGNOSTIC (memory rule: report a derived descriptor's reliability before reading its correlation)."""
    u, _, _ = chord_tables(D, Y=Y)
    g = {c: x.chord_mean.to_numpy(float) for c, x in u.groupby("chemotype") if len(x) >= 2}
    seed = paths.seed_for(key)
    if len(g) < 3:
        return {"label": f"{DIAG}; {SEEN}", "n_chemotypes": len(g), "r_mean": float("nan"), "sb_mean": float("nan"),
                "note": "fewer than 3 chemotypes with >= 2 chord publications", "rng_seed": seed}
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_split):
        a, b = [], []
        for v in g.values():
            p = rng.permutation(len(v))
            h = len(v) // 2
            a.append(v[p[:h]].mean())
            b.append(v[p[h:]].mean())
        r = float(np.corrcoef(a, b)[0, 1])
        rs.append(r)
    rs = np.array(rs)
    sb = 2 * rs / (1 + rs)
    return {"label": f"{DIAG}; {SEEN}", "n_chemotypes": len(g), "chemotypes": sorted(g),
            "n_publications": {c: int(len(v)) for c, v in g.items()}, "n_split": n_split, "rng_seed": seed,
            "r_mean": float(np.nanmean(rs)), "r_q05_q95": [float(np.nanquantile(rs, 0.05)), float(np.nanquantile(rs, 0.95))],
            "sb_mean": float(np.nanmean(sb)), "reliable_ge_0_5": bool(np.nanmean(sb) >= 0.5)}


def targets_table(TV: dict) -> pd.DataFrame:
    out = []
    for (c5, var), tv in TV.items():
        t = tv.reset_index()
        t.insert(1, "c5", c5)
        t.insert(2, "variant", var)
        out.append(t)
    T = pd.concat(out, ignore_index=True)
    T["label"] = "DIAGNOSTIC (arm inputs; donor labels only, seen)"
    return T


# ------------------------------------------------------------------------------------------------------------
# leakage: prediction intervention with positive controls (section 8 item 4)
# ------------------------------------------------------------------------------------------------------------
def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            if not np.array_equal(x.astype(float), y.astype(float), equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def intervention(D: Donors, R: pd.DataFrame, design: str, *, betas: dict, Y=None, targets=None) -> dict:
    """For every target: scramble every finite label of every cell of its publication or copy group (the target's
    unread and read labels included) and rebuild its arm inputs for every (C5 stratum, donor variant) and its Pr/Nd
    offsets in every seed: they must be bit-identical.  Positive controls that must be caught: (a) the target (or a
    cell of its publication) injected into its own donor pool -> Donors.guard raises; (b) a leaky pool that keeps the
    target's own publication / copy group changes under the same scramble (applicable where that pool holds a
    same-chemotype cell with a chord)."""
    Y0 = D.Y if Y is None else np.asarray(Y, float)
    targets = list(dict.fromkeys(D.c3_ids if targets is None else targets))
    rng = np.random.default_rng(paths.seed_for(f"EKX|intervention|{design}"))
    t0 = time.time()
    base = {cfg: D.target_values(targets, Y=Y0, c5=cfg[0], variant=cfg[1]) for cfg in CONFIGS}
    boff = {c5: offsets(D, R, betas=betas, Y=Y0, c5=c5).set_index(["split_seed", "cell_id"]) for c5 in C5_VARIANTS}
    ch0, _ = D.values(Y0)
    bad, pc_guard, pc_pub_guard, pc_leak_app, pc_leak = [], 0, 0, 0, 0
    for t in targets:
        i = D.pos[t]
        aff = (D.pub == D.pub[i]) | (D.cg == D.cg[i])
        Yp = Y0.copy()
        m = aff[:, None] & np.isfinite(Yp)
        Yp[m] = rng.normal(0.0, 5.0, int(m.sum()))
        for cfg in CONFIGS:
            tv = D.target_values([t], Y=Yp, c5=cfg[0], variant=cfg[1])
            if not _same(tv, base[cfg].loc[[t]]):
                bad.append(f"{t} {cfg}")
        for c5 in C5_VARIANTS:
            o = offsets(D, R, betas=betas, Y=Yp, c5=c5, only={t}).set_index(["split_seed", "cell_id"])
            if len(o) and not _same(o, boff[c5].loc[o.index]):
                bad.append(f"{t} offset {c5}")
        idx = D.donor_idx(i)
        try:
            D.guard(i, np.append(idx, i))
        except LeakageError:
            pc_guard += 1
        same_pub = np.flatnonzero((D.pub == D.pub[i]) & (np.arange(D.N) != i))
        if len(same_pub):
            try:
                D.guard(i, np.append(idx, same_pub[0]))
            except LeakageError:
                pc_pub_guard += 1
        lk = D.donor_idx(i, lab=D.chem, leaky=True)
        if (aff[lk] & np.isfinite(ch0[lk])).any():
            pc_leak_app += 1
            a = D.target_values([t], Y=Y0, leaky=True)
            b = D.target_values([t], Y=Yp, leaky=True)
            pc_leak += int(not _same(a, b))
    n_same_pub = int(sum(1 for t in targets if ((D.pub == D.pub[D.pos[t]]).sum() > 1)))
    ok = (not bad and pc_guard == len(targets) and pc_pub_guard == n_same_pub and pc_leak == pc_leak_app)
    return {"label": "LEAKAGE CHECK (section 8 item 4)", "design": design, "targets": len(targets),
            "configs": [list(c) for c in CONFIGS], "mismatches": bad[:20], "n_mismatches": len(bad),
            "pc_self_guard_caught": pc_guard, "pc_samepub_guard_caught": pc_pub_guard,
            "pc_samepub_guard_applicable": n_same_pub, "pc_leaky_caught": pc_leak, "pc_leaky_applicable": pc_leak_app,
            "rng_seed": paths.seed_for(f"EKX|intervention|{design}"), "passed": bool(ok),
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------------------
# reproduction (TECHNICAL SMOKE) and the per-item table
# ------------------------------------------------------------------------------------------------------------
def reproduction(D: Donors, built: dict) -> dict:
    """Brief 6.0 target for this item: B1 recomputed from the fold betas equals the parquet within 8.9e-16 on every
    row (18,020 over both designs); plus construction identities (y, B0, the additive and Nd-potential forms, row
    counts).  The seen 6.8 counts and the -0.11 source value are CONSISTENCY CHECK lines that do not stop the item."""
    out = {"label": SMOKE, "targets": {}, "consistency": {}}
    ok = True
    tot = 0
    for design, (R, ck) in built.items():
        tot += ck["rows"]
        t = {"b1_predict_levels_maxabs": ck["b1_predict_levels_maxabs"], "b1_tol": B1_TOL,
             "b1_ok": ck["b1_predict_levels_maxabs"] <= B1_TOL, "y_ok": ck["y_maxabs"] == 0.0,
             "b0_ok": ck["b0_maxabs"] <= B1_TOL, "b1_additive_ok": ck["b1_additive_maxabs"] <= ADD_TOL,
             "kx_form_ok": ck["kx_additive_identity_maxabs"] <= ADD_TOL, "support_ok": ck["both_ends_read_rows"] == 0
             and ck["adjacent_lighter_first"] and ck["readset_or_fold_inconsistent_groups"] == 0,
             "rows_ok": set(ck["rows_per_seed"].values()) == {ROWS_PER_SEED} and ck["cells"] == len(D.c3_ids),
             "checks": ck}
        t["passed"] = all(t[k] for k in ("b1_ok", "y_ok", "b0_ok", "b1_additive_ok", "kx_form_ok", "support_ok",
                                         "rows_ok"))
        ok &= t["passed"]
        out["targets"][design] = t
        out["consistency"][design] = {"label": CC, "sc009_prnd_rows_per_seed": ck["sc009_prnd_rows_per_seed"],
                                      "seen": SEEN_VALUES["sc009_PrNd_rows_per_seed"],
                                      "match": set(ck["sc009_prnd_rows_per_seed"].values()) == {46}}
    _, _, sc = chord_tables(D)
    out["consistency"]["frozen_value_source"] = sc
    out["rows_total"] = tot
    out["passed"] = bool(ok)
    return out


RULE_COLS = ("role", "c5", "endpoint", "cand", "removal", "top3_publications", "top3_share_of_gain", "dR2_cand_full",
             "dR2_NDU_full", "dR2_cand_without_top3", "dR2_NDU_without_top3", "rule_stat", "rows_per_seed_without",
             "narrowed", "outcome", "label")
EXTRA_SECTIONS = (("rule", "Report rule: every rule row"), ("consistency", "Consistency lines (do not stop the item)"))


def rule_note(rule: pd.DataFrame, design: str) -> str:
    """Which design decides the rule (BP: a sealed exception to brief 9.2) and which rule rows narrow."""
    nn = rule[~rule.narrowed.astype(bool)]
    own = ("this table holds the primary row (role 'REPORT RULE (primary)': C5_out, POOLED, rows removed; REGISTRY "
           "1.8-R01) and its sensitivities (1.8-R02)" if design == PRIMARY_DESIGN else
           "under the BPG base every row is a rule sensitivity (REGISTRY 1.8-R02, 1.8-S03)")
    miss = ("; not narrowed: " + "; ".join(f"{r.c5} {r.endpoint} {r.removal} ({r.rule_stat:+.4f})"
                                           for r in nn.itertuples())) if len(nn) else ""
    return (f"BP decides this item's report rule (brief 6.8 'B1 base = the BP OOF prediction'; ITEM_SPEC; REGISTRY "
            f"1.8-S03), a sealed exception to brief 9.2 'BPG decides labels'; {own}.  Narrowed in "
            f"{len(rule) - len(nn)} of {len(rule)} rows{miss}.")


def per_item(T: dict, design: str, *, seal_tag: str, leakage: str, seconds: float, ext: dict,
             consistency: dict | None = None) -> dict:
    """The section-9.3 fields plus this item's extra sections (EXTRA_SECTIONS; rendered by ``per_item_table``).
    ``consistency`` = reproduction()['consistency'] (None in --smoke, where no consistency line is computed)."""
    a, c = _bare(T["arms"]), _bare(T["contrasts"])
    prim = (a.c5 == PRIMARY_C5) & a.endpoint.isin(["SC009_PRND", "POOLED"])
    met = a[prim][["endpoint", "arm", "R2", "MSE", "relSSE_vs_B1", "MAE", "MAE_balanced", "rows_per_seed", "label"]]
    cp = c[(c.c5 == PRIMARY_C5) & c.endpoint.isin(["SC009_PRND", "POOLED"])][
        ["endpoint", "cand", "ref", "dR2", "dR2_per_seed", "seeds_positive", "relSSE", "dMAE", "label"]]
    ut = T["unit_tests"][T["unit_tests"].c5 == PRIMARY_C5][["endpoint", "unit", "cand", "ref", "G", "W/L/T", "SF_POOL_p",
                                                            "SF_POOL_exact", "SIGN_p", "IM_REL_p", "top3_share"]]
    lo = T["lopo"][T["lopo"].c5 == PRIMARY_C5][["endpoint", "cand", "ref", "point", "lo", "hi", "min", "min_by", "max",
                                                "max_by", "top3_share"]]
    c5t = c[c.endpoint.isin(["SC009_PRND", "POOLED"]) & c.cand.isin(["KX_SC009", "KX_CHEM", "NDU"])
            & c.ref.isin(["B1", "NDU"])][["c5", "endpoint", "cand", "ref", "dR2", "relSSE", "label"]]
    du = T["dup_sens"][["variant", "endpoint", "cand", "ref", "dR2", "relSSE", "label"]]
    rule = T["top3_rule"]
    pr = rule[rule.role == "REPORT RULE (primary)"]
    pr = pr.iloc[0] if len(pr) else rule.iloc[0]
    e1 = c[(c.c5 == PRIMARY_C5) & (c.endpoint == "SC009_PRND") & (c.cand == "KX_SC009") & (c.ref == "B1")]
    d1 = float(e1.dR2.iloc[0]) if len(e1) else float("nan")
    pm = T["permutation"]
    pp = pm[(pm.c5 == PRIMARY_C5) & (pm.endpoint == "POOLED")]
    po = T["population"]
    V = po[(po.c5 == PRIMARY_C5) & po.endpoint.isin(["SC009_PRND", "POOLED"])]
    ct = consistency_table(consistency or {}, designs=(design,))
    sc = (consistency or {}).get("frozen_value_source") or {}
    if not sc:
        fz = "consistency lines not computed in this run"
    elif sc["consistent_2dp"]:
        fz = "the frozen -0.11 is consistent with V1 at 2 decimals"
    else:
        fz = (f"the 2-decimal consistency line of the frozen -0.11 FAILED on V1: publication-first mean "
              f"{sc['u_raw']:.4f}, cell-weighted {sc['u_raw_cell_weighted']:.4f}")
    which = ("the report rule reads" if design == PRIMARY_DESIGN else
             "the BPG sensitivity of the report rule (BP decides for this item) reads")
    return {
        "item": "1.8 M07 sc009 (TODGA-class) Nd dip; estimand E-KX-LOPO (known-chemotype deployment)",
        "population": V,
        "design": (f"E-KX-LOPO: B1 base = round-4 {design} OOF (REF_{design}.parquet), donors = other publications and "
                   f"other copy groups, never called BP" + ("" if design == PRIMARY_DESIGN else " (SENSITIVITY: BPG base)")),
        "read_budget": "4 reads (code k = 3; E-A's nested DOPT_Q read sets, unchanged)",
        "arms": (", ".join(ARMS) + "; frozen external u = " + f"{ext['u_ext']:.4f} (n = {ext['n_donor_publications']}; "
                 f"{fz}; see 'Consistency lines (do not stop the item)' below)"),
        "metrics": met, "per_seed": cp, "units": ut, "lopo_top3": lo, "c5": c5t, "duplicates": du,
        "leakage": leakage, "compute": f"{seconds:.0f}s", "label": f"{design_label(design)}; {NOT_NEXT_TO}",
        "changed": (f"On the seen rows the sc009 Pr/Nd dR2 of the chemotype Nd term over B1 is {d1:+.4f}; without the "
                    f"top-3 publications R2(KX_CHEM) - R2(NDU) = {pr.rule_stat:+.4f} on the pooled rows, so {which} "
                    f"\"{pr.outcome}\" ({CC}; random-chemotype p = "
                    f"{float(pp.p.iloc[0]) if len(pp) else float('nan'):.4f}, descriptive; every rule row in 'Report "
                    f"rule: every rule row' below).  Nothing here is a test; the Pr/Nd warning stands as practical "
                    f"guidance labelled EXPLORATORY."),
        "next_step": (f"Stage 2 rider: the external sign test on the first 4 new sc009 publications with the frozen "
                      f"u_ext = {ext['u_ext']:.4f} ({RIDER_RULE}); data-gated, not a package claim (CP-13).  "
                      f"Seal {seal_tag}."),
        "rule": rule[[k for k in RULE_COLS if k in rule.columns]], "rule_note": rule_note(rule, design),
        "consistency": ct, "consistency_note": (f"Registered lines of ITEM_SPEC 'Reproduction targets' ({CC}); a "
                                                f"failed line is reported and never stops the item.  The frozen "
                                                f"-0.11: {U_RAW_SOURCE}." if len(ct) else
                                                "No consistency line computed in this run (--smoke).")}


def per_item_table(item: dict, path=None, *, title: str | None = None) -> str:
    """r5.common.per_item_table (the section-9.3 fields) followed by this item's EXTRA_SECTIONS (a note, then the
    sub-table); writes LF markdown to ``path`` when given and returns the text."""
    text = CM.per_item_table(item, None, title=title)
    for k, name in EXTRA_SECTIONS:
        if k in item:
            text += f"\n## {name}\n\n{item.get(k + '_note', '')}\n\n{CM.md_table(item[k])}\n"
    if path is not None:
        paths.write_text_lf(Path(path), text)
    return text
