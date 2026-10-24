"""Stage-1 item 1.6 (CLAUDE_CODE_PROMPT.md 6.6; 6.0, 6.10, 8, 9): E-D, the read-budget ladder (M18), the 5th-read
policy envelope (M17) and the recipe-2 row.  Every in-corpus number is EXPLORATORY (seen); nothing here is a test.

Read counts (brief 3.3): r reads = code budget k = r - 1 = the union of the cell's DOPT_Q sequence steps seq[:k].
Row families (identical queries at every rung and under every policy; fixed denominators):
* LAD  DOPT_Q-nested ladder of every C3 cell instance at r = 2 .. min(k_max, 6) + 1 reads.  Frames: the C3 population
       row "C3 min(k, k_max), k = reads - 1" at 2-5 reads (a cell that cannot take r reads keeps min(r, k_max + 1));
       fixed cohorts Cj = {k_max >= j} at 2 .. j + 1 reads (C4 2-5, C5 2-6, C6 2-7).  Frozen query rule: every
       strict-adjacent query of the cohort's cells at every budget; a pair with both ends read at the cohort's largest
       budget would be excluded at every budget -- legal reads never close one, so that set is asserted empty.
* ENV  M17 5th-read policies on the 4 DOPT_Q reads (P3's rules and tie-breaks): DOPTQ (= LAD at 5 reads), MAXMIN_R,
       MAXMIN_Z, BISECT_R, BISECT_Z, BISECT_RMID, HULL_BISECT, RANDOM (exact expectation; Monte Carlo in the analysis),
       ORACLE_B1 [O]; LOOSE-legality sensitivity MAXMIN_R_LOOSE, RANDOM_LOOSE.  k_max = 3 cells keep their 4 reads.
* REC  recipe-2 at 2-5 reads (C3 min rule): LADDER (static La -> Ho -> Nd -> Lu -> Gd, a metal skipped when it is
       not legal, farthest-point HULL fallback), LADDER_GUARD (HULL for cation exchangers of the frozen mechanism map),
       LOCO_STATIC (static set chosen leave-one-cluster-out by B0 SSE on the other clusters' cells; HULL fallback), HULL
       (lightest legal metal, then farthest point in -CN8 radius, ties -> lower Z), DOPTQ, RANDOM_SET (exact mean over
       every legal read set of that size).
Arms: B1 (beta refit per fold and budget, r4.refs.fit_reference(k = reads - 1); k = 3 is the stored round-4 fit, the
refit is an identity check), B0 (the beta = 0 row), HIST_D7 (fixed -0.072; DIAGNOSTIC), B1_k3beta (the k = 3 beta at
every budget; SENSITIVITY).  Legality: gen20k.design.legal_metals for every added read (LOOSE: measured, unread, no
strict neighbour read).  Leakage (brief 8): isolation on every fold; support checks (no query with both ends read,
nested ladders, legal additions, identical row keys); prediction intervention on every read set with a block-Y peek
as the positive control.  The 4-read rung is E-A itself; 5-7 reads are a different read budget (brief 3.3).
"""
# Adapted from the round-5 design folder (never run in place; output paths removed, logic kept):
#   probes/scripts/P3_k4_compute.py (LF SHA-256 ae9762fb8fd3b41a1bf3df55397a1003e5630d15db2a9eef9d9118514f133cc9)
#     5th-read policies maxmin / segments / bisect / outside_hull, the LOOSE rule, the candidate enumeration;
#   probes/scripts/P3_k4_summarise.py (LF SHA-256 7e99fd8271c85adda18845113d0208815c2487cea5a60a12e4bfec5505b70385)
#     pooled R2 per seed, shared cluster bootstrap and LOCO, the RANDOM Monte Carlo, density / saturation / strata;
#   probes/scripts/P3_k4_blocks.py (LF SHA-256 c8a0bb5e5b6f3f34ce80ef9abbc5900dc4502647e92d2590566fc9dfac80e01e)
#     the light / mid / heavy block decomposition in pooled-R2 units;
#   probes/scripts/P3SK_recompute.py (LF SHA-256 c7d0faab5f7e542664abb05e0f13e7e7e816113fd7d3309c74e7a08c5106bf41)
#     the radius-interpolation decomposition levels = PWL_R[y_S - u_S] + u used for RANDOM_SET;
#   probes/scripts/P3SK_analyse.py (LF SHA-256 cba50cd7665299db935e81fc612256c996c7923d59ffbde393733ed563965404)
#     leave-one-publication-out of the 5-read rows;
#   probes/scripts/P3SK_extra.py (LF SHA-256 ec199cd6c9eebe7edc479144823529911a97a75b283f0ff0d051b5e52db2ec1c)
#     contrast-specific implied MDE = 2.8 x cluster-bootstrap sd;
#   ideas2/scripts/recipe2_build_skA.py (LF SHA-256 829dd5aef654eb6c60ec8d59849ac4fb9ccfa59cf519d6a05bf9641e77e42636)
#     the HULL farthest-point order and the static ladder (made legal-only here: identical queries at every rung).
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import paths
from . import common as CM
from r4 import refs as RF  # noqa: E402
from r3 import interp  # noqa: E402
from g21 import cache as GC  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k import inference as INF  # noqa: E402
from gen20k.data import LANTHANIDES, N_LN, Z_VEC, cell_view  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/probes/scripts/P3_k4_compute.py":
        "ae9762fb8fd3b41a1bf3df55397a1003e5630d15db2a9eef9d9118514f133cc9",
    "generations/gen21/round5_design_20261023/probes/scripts/P3_k4_summarise.py":
        "7e99fd8271c85adda18845113d0208815c2487cea5a60a12e4bfec5505b70385",
    "generations/gen21/round5_design_20261023/probes/scripts/P3_k4_blocks.py":
        "c8a0bb5e5b6f3f34ce80ef9abbc5900dc4502647e92d2590566fc9dfac80e01e",
    "generations/gen21/round5_design_20261023/probes/scripts/P3SK_recompute.py":
        "c7d0faab5f7e542664abb05e0f13e7e7e816113fd7d3309c74e7a08c5106bf41",
    "generations/gen21/round5_design_20261023/probes/scripts/P3SK_analyse.py":
        "cba50cd7665299db935e81fc612256c996c7923d59ffbde393733ed563965404",
    "generations/gen21/round5_design_20261023/probes/scripts/P3SK_extra.py":
        "ec199cd6c9eebe7edc479144823529911a97a75b283f0ff0d051b5e52db2ec1c",
    "generations/gen21/round5_design_20261023/ideas2/scripts/recipe2_build_skA.py":
        "829dd5aef654eb6c60ec8d59849ac4fb9ccfa59cf519d6a05bf9641e77e42636",
}

ITEM_ID, ESTIMAND = "1.6", "E-D"
OUT = paths.RES / "ED"
LEAK = paths.RES / "leakage"
XR = np.asarray(interp._coords("R"), float)            # -CN8 radius coordinate of PWL_R
XZ = np.asarray(interp._coords("Z"), float)
ZV = np.asarray(Z_VEC, int)
IDX = {n: i for i, n in enumerate(LANTHANIDES)}
GD = IDX["Gd"]
KEY = list(CM.KEY)                                      # split_seed, fold, cell_id, A, B
BETA_KS = (1, 2, 3, 4, 5, 6)                            # code budgets refitted per fold (reads = k + 1)
K_EA = 3                                                # E-A's budget: four reads
MAX_READS = GC.KMAX + 1                                 # 7
POP_READS = (2, 3, 4, 5)                                # the C3 population row
COHORT_READS = {"C3": (2, 3, 4), "C4": (2, 3, 4, 5), "C5": (2, 3, 4, 5, 6), "C6": (2, 3, 4, 5, 6, 7)}
ARMS = ("B1", "B0", "HIST_D7", "B1_k3beta")
ENV_DET = ("MAXMIN_R", "MAXMIN_Z", "BISECT_R", "BISECT_Z", "BISECT_RMID", "HULL_BISECT")
ENV_LABEL_FREE = ("DOPTQ",) + ENV_DET + ("RANDOM",)       # the envelope family (strict legality)
ENV_POLICIES = ENV_LABEL_FREE + ("ORACLE_B1", "MAXMIN_R_LOOSE", "RANDOM_LOOSE")
REC_READS = (2, 3, 4, 5)
REC_POLICIES = ("LADDER", "LADDER_GUARD", "LOCO_STATIC", "HULL", "DOPTQ", "RANDOM_SET")
STATIC_ORDER = ("La", "Ho", "Nd", "Lu", "Gd")             # I2 recipe-2 salvage: the nested static order
CX_CLASSES = ("CX_ACIDIC_OP", "CX_CARBOXYLIC", "CX_CHELATING")   # guard: cation exchangers (r5.supply MECH_RULES)
BLOCK = {"La/Ce": "light", "Ce/Pr": "light", "Pr/Nd": "light", "Sm/Eu": "mid", "Eu/Gd": "mid", "Gd/Tb": "mid",
         "Tb/Dy": "mid", "Dy/Ho": "mid", "Ho/Er": "heavy", "Er/Tm": "heavy", "Tm/Yb": "heavy", "Yb/Lu": "heavy"}
PAIR_ORDER = list(BLOCK)
BLOCKS = ("light", "mid", "heavy")
REPS = 2000                                             # cluster bootstrap draws (gen20k.inference._draw_counts)
BOOT_SEED = INF.SEED                                    # 8675309, P3's shared draw matrix
MC_REPS = 2000                                          # RANDOM Monte Carlo joint draws
PARETO_FLOOR = -0.002                                   # M17: every block above -0.002 ...
MDE_MULT = 2.8                                          # implied 80 % MDE = 2.8 x bootstrap sd (PC P3)
SAT_FLOOR, SAT_SPREAD, SAT_MIN_CLUSTERS = 0.48, 0.05, 3  # M18 density-invariance rule (consistency check)
DBINS = [0, 0.29, 0.32, 0.37, 0.45, 0.55, 0.70, 1.01]
DLAB = ["0.286 (4/14)", "0.29-0.32 (4/13)", "0.32-0.37 (5/14,4/12,4/11)", "0.37-0.45 (5/13,4/10,4/9,5/12,6/14)",
        "0.45-0.55 (5/11,4/8,5/10,6/13,7/14)", "0.55-0.70 (5/9,4/7,5/8,4/6,6/10)", ">0.70 (5/7,4/5,5/6)"]
HBINS = [0, 1, 2, 3, 4, 6, 99]
HLAB = ["1", "2", "3", "4", "5-6", ">=7"]
TARGETS = {"B1_vs_REF_max_abs": 8.9e-16,                # brief 6.0 reproduction targets (PC P3; P1 section 2)
           "C3pop_r5_B1": {"BPG": 0.4968, "BP": 0.4988}, "C4_r5_B1": {"BPG": 0.4798, "BP": 0.4821}, "tol": 5e-5}
VARIANTS = ("all", "noC5", "noC5cell")
SEEN = "EXPLORATORY (seen)"
C3_ROW = "C3 min(k, k_max), k = reads - 1"                # the C3 population row's name in every headline (brief 6.6)
RB = "different read budget"                            # every rung but 4 reads (brief 3.3; ITEM_SPEC labels)
EA_NOTE = "(E-A rows)"                                  # the C3 4-read cell: no level printed (s0-fidelity-9)
EA_STRATUM_NOTE = "(E-A rows) stratum: a breakdown of E-A's 4-read rows, the reference of the other budgets; never an E-D level"
SEEN_ENVELOPE = {"BPG": 0.0091, "BP": 0.0096}           # brief 6.6 / ITEM_SPEC: the seen envelope widths [P3 5, 9]
PARETO_ROLE = {**{p: "family member" for p in ENV_LABEL_FREE}, "ORACLE_B1": "ORACLE [O] bound (never a member)",
               "MAXMIN_R_LOOSE": "SENSITIVITY (LOOSE legality; never a member)",
               "RANDOM_LOOSE": "SENSITIVITY (LOOSE legality; never a member)"}
pd.set_option("future.no_silent_downcasting", True)


def mc_seed(design: str, variant: str) -> int:
    """REGISTRY scheme default_rng(20261023 + crc32(key)) for the RANDOM Monte Carlo."""
    return paths.seed_for(f"ED|RANDOM_MC|{design}|{variant}")


def join_labels(*parts) -> str:
    """'; '-joined non-empty label parts (no empty separators)."""
    return "; ".join(p for p in parts if p)


def budget_tag(*reads, extra: str = "") -> str:
    """'different read budget' when any of ``reads`` is not 4 (the E-A budget), followed by ``extra``."""
    return join_labels(RB if any(int(r) != 4 for r in reads) else "", extra)


def reads_of(name: str) -> int:
    """Read count of a frame name ('C3pop|r5', 'LADDER C4|r4', 'RECIPE C3pop|r2|HULL', ...)."""
    return int(name.split("|r")[1].split("|")[0].split()[0])


def row_name(name: str) -> str:
    """Headline name of a frame: 'C3 min(k, k_max), k = reads - 1 (5 reads)'; 'C4 fixed cohort (k_max >= 4), 5 reads'."""
    coh = name.split()[-1].split("|")[0].split(":")[0]
    r = reads_of(name)
    return f"{C3_ROW} ({r} reads)" if coh == "C3pop" else f"{coh} fixed cohort (k_max >= {coh[1]}), {r} reads"


def is_ea_cell(name: str) -> bool:
    """True for the C3 4-read cell (E-A's rows and read sets) under any frame spelling."""
    return name.split()[-1].split("|")[0] == "C3pop" and reads_of(name) == 4 and (
        name.endswith("|r4") or name.endswith("|r4|DOPTQ"))


def label_for(arm: str = "B1", policy: str = "", variant: str = "all", design: str = "BPG", extra: str = "") -> str:
    """The section-9.1 label of one number (arm / policy / data variant / design)."""
    parts = [SEEN]
    if policy == "ORACLE_B1":
        parts = ["ORACLE [O]"]
    if arm == "HIST_D7":
        parts.append("DIAGNOSTIC (HIST_D7 not fold-independent)")
    if arm == "B1_k3beta":
        parts.append("SENSITIVITY (k = 3 beta)")
    if policy.endswith("_LOOSE"):
        parts.append("SENSITIVITY (LOOSE legality)")
    if variant == "noC5":
        parts.append("SENSITIVITY (C5-BTBP out)")
    if variant == "noC5cell":
        parts.append("SENSITIVITY (cell c950dc2982e96b83 out)")
    if variant in ("noV2", "noBITDUP"):
        parts.append(f"SENSITIVITY (duplicates: {variant})")
    if design == "BP":
        parts.append("SENSITIVITY (design BP)")
    if extra:
        parts.append(extra)
    return "; ".join(parts)


# ------------------------------------------------------------------------------------------------------------
# levels (exact r4.refs path) and the decomposition used for set enumeration
# ------------------------------------------------------------------------------------------------------------
def levels(yrow, metals, read, beta) -> np.ndarray:
    """B1-form levels from the read metals only (P3_k4_compute ``lv``): r4.refs.predict_levels with u = beta at Gd."""
    ym = np.full(N_LN, np.nan)
    rl = sorted(read)
    ym[rl] = np.asarray(yrow, float)[rl]
    f = {"spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([float(beta)])}
    return RF.predict_levels(f, SimpleNamespace(y_masked=ym), metals)


def sf_parts(yrow, metals, read, qa, qb):
    """(P0, DD): the B0 strict-adjacent prediction per query and its Gd-feature difference, so that the B1-form
    prediction with amplitude beta is P0 + beta * DD (phi_Gd(b) = 1[b = Gd] - w_Gd(b) for unread b, 0 when read)."""
    yrow = np.asarray(yrow, float)
    rd = sorted(read, key=lambda m: XR[m])
    xs = XR[rd]
    ms = np.asarray(sorted(metals), int)
    base = np.full(N_LN, np.nan)
    base[ms] = np.interp(XR[ms], xs, yrow[rd])
    d = np.zeros(N_LN)
    d[ms] = (ms == GD).astype(float) - np.interp(XR[ms], xs, np.array([1.0 if m == GD else 0.0 for m in rd]))
    rl = list(read)
    base[rl] = yrow[rl]
    d[rl] = 0.0
    return base[qa] - base[qb], d[qa] - d[qb]


def scramble_unread(yrow, read, rng) -> np.ndarray:
    """g21.evaluate.scramble_hidden: every measured metal outside ``read`` replaced by N(0, 5) noise."""
    return E.scramble_hidden(np.asarray(yrow, float), tuple(read), rng)


# ------------------------------------------------------------------------------------------------------------
# label-free read policies
# ------------------------------------------------------------------------------------------------------------
def _z(m) -> int:
    return int(ZV[m])


def maxmin(read, legal, x):
    """P3: farthest point -- the legal metal whose distance to the nearest read (coordinate x) is largest; ties ->
    lower Z."""
    return max(legal, key=lambda m: (min(abs(x[m] - x[s]) for s in read), -ZV[m]))


def segments(metals, read):
    ms = sorted(metals, key=_z)
    rd = sorted(read, key=_z)
    segs = [("end_lo", [m for m in ms if ZV[m] < ZV[rd[0]]], None)]
    for a, b in zip(rd, rd[1:]):
        segs.append(("inner", [m for m in ms if ZV[a] < ZV[m] < ZV[b]], (a, b)))
    segs.append(("end_hi", [m for m in ms if ZV[m] > ZV[rd[-1]]], None))
    return ms, segs


def bisect(metals, read, legal, x, *, end_target="extreme", hull_first=False):
    """P3 BISECT: the segment with the most unread measured metals that holds a legal metal (ties -> the lighter
    segment); inside it the legal metal nearest the target (inner gap: coordinate midpoint of the bracketing reads;
    end segment: the cell's extreme metal, or with end_target='mid' the midpoint toward it).  hull_first: read the
    legal metal nearest an unread extreme first (heavy end first)."""
    ms, segs = segments(metals, read)
    lg = set(legal)
    rd = sorted(read, key=_z)
    lo_ext, hi_ext = ms[0], ms[-1]

    def target(kind, br):
        if kind == "inner":
            return 0.5 * (x[br[0]] + x[br[1]])
        ext = lo_ext if kind == "end_lo" else hi_ext
        edge = rd[0] if kind == "end_lo" else rd[-1]
        return x[ext] if end_target == "extreme" else 0.5 * (x[ext] + x[edge])

    if hull_first:
        for kind in ("end_hi", "end_lo"):
            seg = next(s for s in segs if s[0] == kind)
            cand = [m for m in seg[1] if m in lg]
            if cand:
                ext = hi_ext if kind == "end_hi" else lo_ext
                return min(cand, key=lambda m: (abs(x[m] - x[ext]), ZV[m]))
    order = sorted(range(len(segs)), key=lambda i: (-len(segs[i][1]), i))
    for i in order:
        kind, seg, br = segs[i]
        cand = [m for m in seg if m in lg]
        if cand:
            t = target(kind, br)
            return min(cand, key=lambda m: (abs(x[m] - t), ZV[m]))
    raise AssertionError("no legal metal in any segment")


def outside_hull(a, b, read) -> bool:
    """Strict-adjacent pair with an unread end outside the read hull (constant continuation)."""
    zr = [ZV[m] for m in read]
    lo, hi = min(zr), max(zr)
    return any((m not in read) and (ZV[m] < lo or ZV[m] > hi) for m in (a, b))


def loose_metals(metals, read, query) -> list:
    """E3's LOOSE rule: measured, unread, no strict-adjacent neighbour read (no reachability look-ahead)."""
    qs = set(query)
    return sorted((m for m in metals if m not in read and not any(((m, s) in qs or (s, m) in qs) for s in read)), key=_z)


def env_picks(metals, read4, legal, loose, dopt5) -> dict:
    """The M17 5th-read picks (P3_k4_compute, unchanged)."""
    return {"DOPTQ": dopt5, "MAXMIN_R": maxmin(read4, legal, XR), "MAXMIN_Z": maxmin(read4, legal, XZ),
            "BISECT_R": bisect(metals, read4, legal, XR), "BISECT_Z": bisect(metals, read4, legal, XZ),
            "BISECT_RMID": bisect(metals, read4, legal, XR, end_target="mid"),
            "HULL_BISECT": bisect(metals, read4, legal, XR, hull_first=True),
            "MAXMIN_R_LOOSE": maxmin(read4, loose, XR)}


def _farthest(order, legal):
    if not order:
        return min(legal, key=_z)
    return max(legal, key=lambda m: (min(abs(XR[m] - XR[s]) for s in order), -ZV[m]))


def hull_sequence(dz, n: int) -> list:
    """HULL (recipe-2's HULL_MAXMIN made legal-only): the lightest legal metal, then the legal metal farthest (in
    -CN8 radius) from the read set, ties -> lower Z; at most n metals, never past k_max + 1."""
    order = []
    while len(order) < n:
        legal = DS.legal_metals(dz, frozenset(order))
        if not legal:
            break
        order.append(_farthest(order, legal))
    return order


def ladder_sequence(dz, n: int, static=STATIC_ORDER) -> list:
    """recipe-2's nested static order: at each step the first static metal that is legal; when none is, the HULL
    farthest-point pick."""
    st = [IDX[s] for s in static]
    order = []
    while len(order) < n:
        legal = DS.legal_metals(dz, frozenset(order))
        if not legal:
            break
        nxt = next((m for m in st if m in legal), None)
        order.append(nxt if nxt is not None else _farthest(order, legal))
    return order


def is_legal_set(dz, S) -> bool:
    """A read set reachable by legal sequential reads: no query inside and still extendable to a maximum
    independent set (equivalent to every addition passing gen20k.design.legal_metals, in any order)."""
    S = frozenset(S)
    if not S <= set(dz.cell.metals):
        return False
    return DS.max_independent_size(dz.cell.metals, dz.query, S) == DS.max_independent_size(dz.cell.metals, dz.query,
                                                                                            frozenset())


def legal_sets(dz, r: int) -> list:
    """Every legal read set of size r of the cell, sorted lexicographically by Z."""
    qs = set(dz.query)
    out = []
    for S in itertools.combinations(sorted(dz.cell.metals, key=_z), r):
        if any((a, b) in qs or (b, a) in qs for a, b in itertools.combinations(S, 2)):
            continue
        if is_legal_set(dz, S):
            out.append(tuple(S))
    return out


def static_candidates(r: int) -> list:
    """Every r-subset of the 14 lanthanides with no strict-adjacent pair (|dZ| = 1), sorted by Z."""
    out = []
    for S in itertools.combinations(range(N_LN), r):
        if not any(abs(_z(a) - _z(b)) == 1 for a, b in itertools.combinations(S, 2)):
            out.append(S)
    return out


def assert_legal_chain(dz, reads) -> None:
    """Every rung legal, nested, one metal more per rung (two at the first)."""
    prev = frozenset()
    for i, R in enumerate(reads):
        R = frozenset(R)
        if not prev <= R or len(R - prev) != (len(R) if i == 0 else 1):
            raise AssertionError("ladder not nested")
        new = R - prev
        if i > 0 and next(iter(new)) not in DS.legal_metals(dz, prev):
            raise AssertionError("an added read is not legal")
        if not is_legal_set(dz, R):
            raise AssertionError("an illegal read set")
        prev = R


def longest_hidden_run(metals, read) -> int:
    best = cur = 0
    for m in sorted(metals, key=_z):
        cur = 0 if m in read else cur + 1
        best = max(best, cur)
    return best


def out_share(metals, read) -> float:
    """Share of the cell's unread measured metals outside the read hull."""
    zr = [ZV[m] for m in read]
    lo, hi = min(zr), max(zr)
    un = [m for m in metals if m not in read]
    return float(np.mean([(ZV[m] < lo or ZV[m] > hi) for m in un])) if un else 0.0


def rs_str(read) -> str:
    return ",".join(LANTHANIDES[m] for m in sorted(read))


def cation_exchanger(chem_family: str, extractant_name: str) -> bool:
    """The recipe-2 chemistry-class guard, blind to values: the frozen r5.supply mechanism map classes a cation
    exchanger (acidic organophosphorus, carboxylic acids, chelating cation exchangers)."""
    from . import supply as SU
    return SU.classify_extractant(str(chem_family).replace("_", " "), str(extractant_name))["mechanism_class"] in CX_CLASSES


# ------------------------------------------------------------------------------------------------------------
# compute: fits and row-level outputs
# ------------------------------------------------------------------------------------------------------------
def stored_fits(design: str) -> dict:
    d = json.loads((paths.R4RES / "REF" / f"fitted_{design}.json").read_bytes())
    return {tuple(map(int, k.split("_"))): (float(v["ref:B1"]["coef"][0]), str(v["ref:B1"]["lambda"]))
            for k, v in d.items() if "_" in k and k.replace("_", "").isdigit()}


def fit_betas(bench, rec, ks, stored: tuple) -> tuple[dict, list]:
    """beta per code budget k (k = 3: the stored round-4 fit; its refit, when asked, is an identity check)."""
    betas, log = {K_EA: stored[0]}, []
    for k in ks:
        t0 = time.time()
        with threadpool_limits(limits=1):
            fit = RF.fit_reference(bench, rec, bench.Y, "B1", k=k)
        b = float(fit["coef"][0])
        row = {"design": rec["design"], "split_seed": int(rec["split_seed"]), "fold": int(rec["fold"]), "k": int(k),
               "reads": int(k) + 1, "beta": b, "lambda": str(fit["lambda"]), "n_situations": int(fit["n_situations"]),
               "n_rows": int(fit["n_rows"]), "train_cells_used": int(fit["coverage"]["train_cells_k3_used"]),
               "train_cells_eligible": int(fit["coverage"]["train_cells_k3_eligible"]),
               "inner_folds_scored": int(fit["coverage"]["inner_folds_scored"]), "seconds": round(time.time() - t0, 2)}
        if k == K_EA:
            row.update({"beta_stored": stored[0], "lambda_stored": stored[1], "abs_diff_stored": abs(b - stored[0]),
                        "lambda_equal": str(fit["lambda"]) == stored[1]})
        else:
            betas[k] = b
        log.append(row)
    return betas, log


class _Buf:
    """Column buffer for the row-level output (scalars broadcast)."""

    def __init__(self):
        self.cols: dict[str, list] = {}

    def add(self, n: int, cols: dict) -> None:
        if self.cols and set(cols) != set(self.cols):
            raise KeyError(f"row schema changed: {sorted(set(cols) ^ set(self.cols))}")
        for k, v in cols.items():
            a = np.asarray(v)
            if a.ndim == 0:
                a = np.full(n, v, dtype=object if isinstance(v, str) else None)
            if len(a) != n:
                raise ValueError(f"column {k}: {len(a)} != {n}")
            self.cols.setdefault(k, []).append(a)

    def frame(self) -> pd.DataFrame:
        if not self.cols:
            return pd.DataFrame()
        return pd.DataFrame({k: np.concatenate(v) for k, v in self.cols.items()})


def _read_info(metals, read, qa, qb) -> dict:
    rd = set(read)
    return {"read_set": rs_str(read), "n_read": len(rd),
            "hull_out": np.array([float(outside_hull(a, b, rd)) for a, b in zip(qa, qb)]),
            "a_read": np.array([float(a in rd) for a in qa]), "b_read": np.array([float(b in rd) for b in qb]),
            "lhr": float(longest_hidden_run(metals, rd)), "out_share": out_share(metals, rd)}


class Checks:
    """Leakage / support counters (brief 8 items 1, 4, 5)."""

    def __init__(self):
        self.c = {"folds": 0, "isolation_ok": 0, "cell_instances": 0, "read_sets_checked": 0,
                  "intervention_checked": 0, "intervention_failed": 0, "peek_control_runs": 0, "peek_control_caught": 0,
                  "both_ends_read": 0, "illegal_or_unnested": 0, "dopt5_not_legal": 0, "dopt5_mismatch_cache": 0,
                  "loose_not_superset": 0, "dopt_candidate_mismatch": 0, "static_sets_intervention_checked": 0,
                  "static_sets_intervention_failed": 0, "cohort_largest_budget_closed_queries": 0}

    def add(self, k, v=1):
        self.c[k] += v


def _arm_levels(yrow, metals, read, betas_r: dict) -> dict:
    return {a: levels(yrow, metals, read, b) for a, b in betas_r.items()}


def _emit(buf, base, qa, qb, y, info, preds=None, se=None, ae=None, **extra):
    n = len(qa)
    cols = dict(base)
    cols.update({"A": np.array([LANTHANIDES[a] for a in qa], dtype=object),
                 "B": np.array([LANTHANIDES[b] for b in qb], dtype=object), "y": y})
    cols.update(info)
    for a in ARMS:
        p = preds[a] if preds is not None else np.full(n, np.nan)
        cols["p_" + a] = p
        cols["se_" + a] = (p - y) ** 2 if preds is not None else se[a]
        cols["ae_" + a] = np.abs(p - y) if preds is not None else ae[a]
    cols.update({"n_alt": 1, "fallback": False, "family": "", "policy": "", "reads_req": 0})
    cols.update(extra)
    buf.add(n, cols)


def _mean_info(infos) -> dict:
    return {"read_set": "", "n_read": int(infos[0]["n_read"]),
            "hull_out": np.mean([i["hull_out"] for i in infos], 0), "a_read": np.mean([i["a_read"] for i in infos], 0),
            "b_read": np.mean([i["b_read"] for i in infos], 0), "lhr": float(np.mean([i["lhr"] for i in infos])),
            "out_share": float(np.mean([i["out_share"] for i in infos]))}


def static_tables(bench, c3_cis, *, intervention=True, checks=None, rng=None) -> dict:
    """Per C3 cell (label-free designs; labels only in the error terms): HULL and LADDER sequences, the guard flag and,
    for r = 2 .. min(5, k_max + 1), every legal read set with its (P0, DD) per query."""
    rng = rng if rng is not None else np.random.default_rng(paths.seed_for("ED|static_intervention"))
    fr = bench.frame
    out = {}
    for ci in c3_cis:
        cv = cell_view(bench, ci)
        dz = DS.make_design(cv, "adjacent", "metal")
        qa = np.array([a for a, b in dz.query])
        qb = np.array([b for a, b in dz.query])
        yrow = np.asarray(bench.Y[ci], float)
        nmax = min(dz.k_max, GC.KMAX) + 1
        rec = {"cell_id": cv.cell_id, "metals": tuple(cv.metals), "query": tuple(dz.query), "k_max": int(dz.k_max),
               "hull": hull_sequence(dz, min(nmax, max(REC_READS))), "ladder": ladder_sequence(dz, min(nmax, max(REC_READS))),
               "cx": bool(cation_exchanger(fr.chem_family.iat[ci], fr.extractant_name.iat[ci])), "sets": {}}
        for seqn in ("hull", "ladder"):
            assert_legal_chain(dz, [rec[seqn][:r] for r in range(2, len(rec[seqn]) + 1)])
        for r in range(2, min(max(REC_READS), nmax) + 1):
            S = legal_sets(dz, r)
            P0 = np.empty((len(S), len(qa)))
            DD = np.empty((len(S), len(qa)))
            for i, s in enumerate(S):
                P0[i], DD[i] = sf_parts(yrow, cv.metals, s, qa, qb)
                if intervention:
                    p, d = sf_parts(scramble_unread(yrow, s, rng), cv.metals, s, qa, qb)
                    ok = np.array_equal(p, P0[i]) and np.array_equal(d, DD[i])
                    if checks is not None:
                        checks.add("static_sets_intervention_checked")
                        checks.add("static_sets_intervention_failed", int(not ok))
            rec["sets"][r] = {"S": S, "P0": P0, "DD": DD, "y": yrow[qa] - yrow[qb]}
        out[ci] = rec
    return out


def loco_static_selection(static: dict, cluster_of: dict) -> pd.DataFrame:
    """LOCO-selected static sets (recipe-2 salvage): for every requested size r and held-out cluster c, the static
    set S (no strict-adjacent pair) minimising the summed B0 SSE over the C3 cells of the OTHER clusters, where a cell
    on which S is not a legal read set uses its HULL set of size min(r, k_max + 1); ties -> lexicographically smallest
    by Z.  The held-out cluster's labels never enter its own choice."""
    rows = []
    cis = sorted(static)
    cl = np.array([cluster_of[static[ci]["cell_id"]] for ci in cis])
    for r in REC_READS:
        cands = static_candidates(r)
        cidx = {s: j for j, s in enumerate(cands)}
        M = np.empty((len(cis), len(cands)))
        for i, ci in enumerate(cis):
            st = static[ci]
            reff = min(r, st["k_max"] + 1)
            hs = tuple(sorted(st["hull"][:reff], key=_z))
            tab = st["sets"][reff]
            hrow = tab["S"].index(hs)
            hull_sse = float(((tab["P0"][hrow] - tab["y"]) ** 2).sum())
            M[i] = hull_sse
            if reff == r:
                for j, s in enumerate(tab["S"]):
                    M[i, cidx[s]] = float(((tab["P0"][j] - tab["y"]) ** 2).sum())
        for c in sorted(set(cl)):
            tot = M[cl != c].sum(0)
            j = int(np.argmin(tot))
            held = [ci for ci, cc in zip(cis, cl) if cc == c]
            app = [cands[j] in static[ci]["sets"].get(r, {"S": []})["S"] for ci in held]
            rows.append({"reads_req": r, "cluster_cpg": c, "set": rs_str(cands[j]), "set_idx": ",".join(map(str, cands[j])),
                         "n_candidates": len(cands), "held_out_cells": len(held),
                         "applicable_share_held_out": float(np.mean(app)) if app else float("nan")})
    return pd.DataFrame(rows)


def poison_loco(bench, c3_cis, cluster_of: dict, static: dict, selection: pd.DataFrame) -> dict:
    """Fit poisoning of the recipe-2 LOCO static read-set selection (brief 8 item 2; s0-mechanics-14).  For each
    cluster c, the finite labels of c's C3 cells are replaced by seeded noise (seed_for('ED|LOCO_POISON|<c>')), the
    static tables are rebuilt on that bench and the selection is rerun: the rows of c (the sets chosen for c) must be
    identical to the production selection.  Positive control: a leaky selection that includes c's own rows
    (argmin over every cell) must change for at least one c."""
    from types import SimpleNamespace
    prod = selection.set_index(["reads_req", "cluster_cpg"]).set_idx.astype(str)
    cis = sorted(static)
    cl = {ci: cluster_of[static[ci]["cell_id"]] for ci in cis}
    rep = {"clusters": 0, "changed": [], "pc_runs": 0, "pc_caught": 0}

    def leaky(st):
        out = {}
        for r in REC_READS:
            cands = static_candidates(r)
            cidx = {s: j for j, s in enumerate(cands)}
            M = np.empty((len(cis), len(cands)))
            for i, ci in enumerate(cis):
                x = st[ci]
                reff = min(r, x["k_max"] + 1)
                tab = x["sets"][reff]
                hrow = tab["S"].index(tuple(sorted(x["hull"][:reff], key=_z)))
                M[i] = float(((tab["P0"][hrow] - tab["y"]) ** 2).sum())
                if reff == r:
                    for j, s_ in enumerate(tab["S"]):
                        M[i, cidx[s_]] = float(((tab["P0"][j] - tab["y"]) ** 2).sum())
            out[r] = ",".join(map(str, cands[int(np.argmin(M.sum(0)))]))
        return out
    leak0 = leaky(static)
    for c in sorted(set(cl.values())):
        rng = np.random.default_rng(paths.seed_for(f"ED|LOCO_POISON|{c}"))
        Y2 = np.array(bench.Y, float, copy=True)
        for ci in cis:
            if cl[ci] == c:
                fin = np.isfinite(Y2[ci])
                Y2[ci, fin] = rng.normal(0.0, 3.0, int(fin.sum()))
        b2 = SimpleNamespace(**{k: getattr(bench, k) for k in ("frame", "basis") if hasattr(bench, k)}, Y=Y2)
        st2 = static_tables(b2, cis, intervention=False)
        sel2 = loco_static_selection(st2, cluster_of).set_index(["reads_req", "cluster_cpg"]).set_idx.astype(str)
        same = all(sel2.loc[(r, c)] == prod.loc[(r, c)] for r in REC_READS if (r, c) in prod.index)
        rep["clusters"] += 1
        if not same:
            rep["changed"].append(str(c))
        lk = leaky(st2)
        rep["pc_runs"] += 1
        rep["pc_caught"] += int(any(lk[r] != leak0[r] for r in REC_READS))
    rep["passed"] = bool(not rep["changed"] and rep["pc_caught"] > 0 and rep["clusters"] > 0)
    return rep


def compute_design(bench, design: str, *, ks=BETA_KS, folds=None, families=("LAD", "ENV", "REC"), max_reads=MAX_READS,
                   min_reads=2, static=None, selection=None, cluster_of=None, intervention=True, log=print) -> dict:
    """Row-level outputs of one design: LAD / ENV / REC rows, the ENV candidate SSE table (Monte Carlo, oracle), the
    beta fits per fold and budget, and the leakage / support counters.  min_reads / max_reads bound the emitted LAD
    rungs (the reproduction run needs 4 and 5 reads only); legality is checked on every rung regardless."""
    t0 = time.time()
    Y = np.asarray(bench.Y, float)
    stored = stored_fits(design)
    chk = Checks()
    rng = np.random.default_rng(paths.seed_for(f"ED|intervention|{design}"))
    buf, cand_rows, fit_log = _Buf(), [], []
    sel = {} if selection is None else {(int(r.reads_req), r.cluster_cpg): tuple(map(int, r.set_idx.split(",")))
                                        for r in selection.itertuples(index=False)}
    fl = folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS))
    if folds is not None:
        fl = fl[:folds]
    fr = bench.frame
    for f in fl:
        rec = GC.load(f.design, f.seed, f.fold)
        E.assert_test_isolation(bench, rec)
        chk.add("folds")
        chk.add("isolation_ok")
        betas, flog = fit_betas(bench, rec, [k for k in ks], stored[(int(f.seed), int(f.fold))])
        fit_log += flog
        for cell in rec["outer"]["cells"]:
            if cell["k_max"] < 3:
                continue
            chk.add("cell_instances")
            ci, metals, query = int(cell["ci"]), tuple(cell["metals"]), tuple(cell["query"])
            dz = DS.make_design(E._view(cell), "adjacent", "metal")
            assert tuple(dz.query) == query and dz.k_max == cell["k_max"]
            kmax = int(cell["k_max"])
            seq = cell["seqs"]["DOPT_Q"]
            nmax = min(kmax, GC.KMAX) + 1
            assert len(seq) == nmax - 1
            qa = np.array([a for a, b in query])
            qb = np.array([b for a, b in query])
            yrow = Y[ci]
            y = yrow[qa] - yrow[qb]
            cid = cell["cell_id"]
            base = {"design": design, "split_seed": int(f.seed), "fold": int(f.fold), "cell_id": cid,
                    "publication": cell["publication"], "chemotype": cell["chemotype"], "extractant": cell["extractant"],
                    "cluster_cpg": cluster_of.get(cid, "") if cluster_of else "", "n_metals": len(metals), "k_max": kmax,
                    "cx": bool(static[ci]["cx"]) if static is not None and ci in static else False,
                    "pair": np.array([f"{LANTHANIDES[a]}/{LANTHANIDES[b]}" for a, b in query], dtype=object)}
            base["block"] = np.array([BLOCK[p] for p in base["pair"]], dtype=object)

            def predict(read, k_r):
                b_r = {"B1": betas[k_r], "B0": 0.0, "HIST_D7": RF.HIST_D7_BETA, "B1_k3beta": betas[K_EA]}
                L = _arm_levels(yrow, metals, read, b_r)
                chk.add("read_sets_checked")
                if any(a in read and b in read for a, b in query):
                    chk.add("both_ends_read")
                if intervention:
                    Ls = _arm_levels(scramble_unread(yrow, read, rng), metals, read, b_r)
                    ok = all(np.array_equal(L[a][list(metals)], Ls[a][list(metals)], equal_nan=True) for a in L)
                    chk.add("intervention_checked")
                    chk.add("intervention_failed", int(not ok))
                return {a: L[a][qa] - L[a][qb] for a in ARMS}

            rd = frozenset(m for p in seq[:K_EA] for m in p)
            if intervention and chk.c["peek_control_runs"] < chk.c["folds"] and any(m not in rd for m in metals):
                # positive control (s0-mechanics-6): a block-Y peek through the real predictor and comparator -- the
                # level function reads every measured metal, so scrambling the unread metals must change its levels
                b_r = {"B1": betas[K_EA], "B0": 0.0, "HIST_D7": RF.HIST_D7_BETA, "B1_k3beta": betas[K_EA]}
                allm = frozenset(m for m in metals if np.isfinite(yrow[m]))
                L1 = _arm_levels(yrow, metals, allm, b_r)
                L2 = _arm_levels(scramble_unread(yrow, rd, rng), metals, allm, b_r)
                caught = not all(np.array_equal(L1[a][list(metals)], L2[a][list(metals)], equal_nan=True) for a in L1)
                chk.add("peek_control_runs")
                chk.add("peek_control_caught", int(caught))

            lad = {}
            for r in range(2, min(nmax, max_reads) + 1):
                lad[r] = frozenset(m for p in seq[:r - 1] for m in p)
            try:
                assert_legal_chain(dz, [lad[r] for r in sorted(lad)])
            except AssertionError:
                chk.add("illegal_or_unnested")
                raise
            top = frozenset(m for p in seq for m in p)                            # the cell's largest budget
            chk.add("cohort_largest_budget_closed_queries", sum(a in top and b in top for a, b in query))
            lad_pred = {}
            if "LAD" in families:
                for r, read in lad.items():
                    if r < min_reads:
                        continue
                    lad_pred[r] = predict(read, r - 1)
                    _emit(buf, base, qa, qb, y, _read_info(metals, read, qa, qb), preds=lad_pred[r], family="LAD",
                          policy="DOPTQ", reads_req=r)
            if "ENV" in families and kmax >= 4:
                read4 = lad[4]
                legal = sorted(DS.legal_metals(dz, read4), key=_z)
                d5 = sorted(lad[5] - read4)
                chk.add("dopt5_mismatch_cache", int(len(d5) != 1))
                dopt5 = d5[0]
                chk.add("dopt5_not_legal", int(dopt5 not in legal))
                loose = loose_metals(metals, read4, query)
                chk.add("loose_not_superset", int(not set(legal) <= set(loose)))
                picks = env_picks(metals, read4, legal, loose, dopt5)
                C, I = {}, {}
                for m in loose:
                    read5 = read4 | {m}
                    C[m] = predict(read5, 4)
                    I[m] = _read_info(metals, read5, qa, qb)
                    cand_rows.append({"design": design, "split_seed": int(f.seed), "fold": int(f.fold), "cell_id": cid,
                                      "publication": cell["publication"], "cluster_cpg": base["cluster_cpg"],
                                      "k_max": kmax, "cand": LANTHANIDES[m], "strict_legal": m in legal,
                                      "n_q": len(qa), **{f"sse_{a}": float(((C[m][a] - y) ** 2).sum()) for a in ARMS}})
                if "LAD" in families:
                    chk.add("dopt_candidate_mismatch", int(not all(np.array_equal(C[dopt5][a], lad_pred[5][a]) for a in ARMS)))
                for pol in ENV_DET + ("MAXMIN_R_LOOSE",):
                    m = picks[pol]
                    _emit(buf, base, qa, qb, y, I[m], preds=C[m], family="ENV", policy=pol, reads_req=5)
                for pol, pool in (("RANDOM", legal), ("RANDOM_LOOSE", loose)):
                    se = {a: np.mean([(C[m][a] - y) ** 2 for m in pool], 0) for a in ARMS}
                    ae = {a: np.mean([np.abs(C[m][a] - y) for m in pool], 0) for a in ARMS}
                    _emit(buf, base, qa, qb, y, _mean_info([I[m] for m in pool]), se=se, ae=ae, family="ENV",
                          policy=pol, reads_req=5, n_alt=len(pool))
                orc = min(legal, key=lambda m: (float(((C[m]["B1"] - y) ** 2).sum()), ZV[m]))
                _emit(buf, base, qa, qb, y, I[orc], preds=C[orc], family="ENV", policy="ORACLE_B1", reads_req=5)
            if "REC" in families and static is not None:
                st = static[ci]
                for r in REC_READS:
                    reff = min(r, kmax + 1)
                    k_r = reff - 1
                    for pol in ("HULL", "LADDER"):
                        read = frozenset(st["hull" if pol == "HULL" else "ladder"][:reff])
                        assert len(read) == reff
                        _emit(buf, base, qa, qb, y, _read_info(metals, read, qa, qb), preds=predict(read, k_r),
                              family="REC", policy=pol, reads_req=r)
                    S = sel.get((r, base["cluster_cpg"]))
                    fb = not (S is not None and len(S) == reff and S in st["sets"][reff]["S"])
                    read = frozenset(st["hull"][:reff]) if fb else frozenset(S)
                    _emit(buf, base, qa, qb, y, _read_info(metals, read, qa, qb), preds=predict(read, k_r), family="REC",
                          policy="LOCO_STATIC", reads_req=r, fallback=fb)
                    tab = st["sets"][reff]
                    assert np.array_equal(tab["y"], y)
                    b_r = {"B1": betas[k_r], "B0": 0.0, "HIST_D7": RF.HIST_D7_BETA, "B1_k3beta": betas[K_EA]}
                    P = {a: tab["P0"] + b * tab["DD"] for a, b in b_r.items()}
                    se = {a: ((P[a] - y[None, :]) ** 2).mean(0) for a in ARMS}
                    ae = {a: np.abs(P[a] - y[None, :]).mean(0) for a in ARMS}
                    infos = [_read_info(metals, s, qa, qb) for s in tab["S"]]
                    _emit(buf, base, qa, qb, y, _mean_info(infos), se=se, ae=ae, family="REC", policy="RANDOM_SET",
                          reads_req=r, n_alt=len(tab["S"]))
        log(f"  {design} {f.seed} fold {f.fold}: beta fits k {sorted(ks)} ({sum(r['seconds'] for r in flog):.0f}s); "
            f"rows so far {sum(len(x) for x in buf.cols.get('y', []))}; {time.time() - t0:.0f}s")
    rows = buf.frame()
    for c in ("design", "cell_id", "publication", "chemotype", "extractant", "cluster_cpg", "A", "B", "pair", "block",
              "family", "policy", "read_set"):
        if c in rows:
            rows[c] = rows[c].astype(str).astype("category")
    return {"rows": rows, "cand": pd.DataFrame(cand_rows), "fits": pd.DataFrame(fit_log), "checks": chk.c,
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------------------
# frames on identical rows
# ------------------------------------------------------------------------------------------------------------
def row_labels(R: pd.DataFrame) -> pd.Series:
    """The section-9.1 label of every row-level output (per policy and design; arms carry theirs in the tables)."""
    lab = {(d, p): label_for("B1", p, "all", d, "row-level output; HIST_D7 columns DIAGNOSTIC, B1_k3beta SENSITIVITY")
           for d in R.design.astype(str).unique() for p in R.policy.astype(str).unique()}
    return pd.Series([lab[(d, p)] for d, p in zip(R.design.astype(str), R.policy.astype(str))], index=R.index,
                     dtype="category")


def split_rows(R: pd.DataFrame) -> dict:
    return {k: g for k, g in R.groupby(["family", "policy"], observed=True)}


def _sorted(F: pd.DataFrame) -> pd.DataFrame:
    return F.sort_values(KEY).reset_index(drop=True)


def lad_frame(G: dict, cohort: str, r: int) -> pd.DataFrame:
    """DOPT_Q-nested rows: cohort 'C3pop' = C3 min(k, k_max) at r reads; 'Cj' = cells with k_max >= j at r reads."""
    L = G[("LAD", "DOPTQ")]
    km = L.k_max.to_numpy()
    if cohort == "C3pop":
        m = L.n_read.to_numpy() == np.minimum(r, np.minimum(km, GC.KMAX) + 1)
    else:
        j = int(cohort[1:])
        if r > j + 1:
            raise ValueError(f"{cohort} has no {r}-read rung")
        m = (km >= j) & (L.n_read.to_numpy() == r)
    return _sorted(L[m])


def env_frame(G: dict, cohort: str, policy: str) -> pd.DataFrame:
    """5-read policy rows; cohort 'C3pop' (k_max = 3 cells keep 4 reads) or 'C4'."""
    if policy == "DOPTQ":
        return lad_frame(G, cohort, 5)
    X = G[("ENV", policy)]
    if cohort == "C4":
        return _sorted(X)
    k3 = lad_frame(G, "C3pop", 4)
    return _sorted(pd.concat([k3[k3.k_max == 3], X], ignore_index=True))


def rec_frame(G: dict, policy: str, r: int) -> pd.DataFrame:
    if policy == "DOPTQ":
        return lad_frame(G, "C3pop", r)
    if policy == "LADDER_GUARD":
        L, H = rec_frame(G, "LADDER", r), rec_frame(G, "HULL", r)
        return _sorted(pd.concat([L[~L.cx.astype(bool)], H[H.cx.astype(bool)]], ignore_index=True))
    X = G[("REC", policy)]
    return _sorted(X[X.reads_req == r])


def variant(F: pd.DataFrame, v: str, drop_cells=None) -> pd.DataFrame:
    if v == "all":
        return F
    if v == "noC5":
        return F[F.publication.astype(str) != paths.C5]
    if v == "noC5cell":
        return F[F.cell_id.astype(str) != paths.C5_CELL]
    if drop_cells is not None:
        return F[~F.cell_id.astype(str).isin(drop_cells)]
    raise KeyError(v)


def same_rows(F1: pd.DataFrame, F2: pd.DataFrame) -> bool:
    return len(F1) == len(F2) and all(np.array_equal(F1[c].astype(str).to_numpy(), F2[c].astype(str).to_numpy())
                                      for c in KEY)


# ------------------------------------------------------------------------------------------------------------
# statistics on squared-error columns (pooled per seed, then the seed mean)
# ------------------------------------------------------------------------------------------------------------
class Stats:
    """Cluster-indexed sums with one shared bootstrap draw matrix over the C3 clusters (P3's convention)."""

    def __init__(self, clusters, reps: int = REPS, seed: int = BOOT_SEED):
        self.CL = sorted(map(str, clusters))
        self.pos = {c: i for i, c in enumerate(self.CL)}
        self.W = INF._draw_counts(len(self.CL), reps, seed)

    def sums(self, F: pd.DataFrame, cols) -> dict:
        pos = F.cluster_cpg.astype(str).map(self.pos).to_numpy()
        if np.isnan(pos.astype(float)).any():
            raise KeyError("row with an unknown cluster")
        pos = pos.astype(int)
        seeds = F.split_seed.to_numpy()
        y = F.y.to_numpy(float)
        vals = [np.ones_like(y), y, y * y] + [F[c].to_numpy(float) for c in cols]
        out = {}
        for s in np.unique(seeds):
            m = seeds == s
            out[int(s)] = np.stack([np.bincount(pos[m], weights=v[m], minlength=len(self.CL)) for v in vals], 1)
        return out

    @staticmethod
    def _sst(a):
        n, sy, syy = a[..., 0], a[..., 1], a[..., 2]
        return syy - sy ** 2 / np.maximum(n, 1e-12)

    def level(self, F: pd.DataFrame, col: str) -> dict:
        """Per-seed pooled R2 of 1 - sum(col) / SST, bootstrap draws of the seed mean, LOCO seed means."""
        S = self.sums(F, [col])
        per, draws, loco = {}, [], {}
        for s, a in S.items():
            sst = float(a[:, 2].sum() - a[:, 1].sum() ** 2 / a[:, 0].sum())
            per[s] = 1.0 - float(a[:, 3].sum()) / sst if a[:, 0].sum() >= 2 and sst > 0 else np.nan
            wa = self.W @ a
            draws.append(1.0 - wa[:, 3] / self._sst(wa))
        present = [self.CL[g] for g in range(len(self.CL)) if any(a[g, 0] > 0 for a in S.values())]
        for c in present:
            g = self.pos[c]
            v = []
            for a in S.values():
                b = np.delete(a, g, 0).sum(0)
                sst = b[2] - b[1] ** 2 / b[0] if b[0] > 1 else np.nan
                v.append(1.0 - b[3] / sst if b[0] > 1 and sst > 0 else np.nan)
            loco[c] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        return {"per_seed": pd.Series(per, dtype=float), "draws": np.mean(np.vstack(draws), 0) if draws else np.array([]),
                "loco": loco}

    def delta(self, F: pd.DataFrame, col_ref: str, col_cand: str, mask_col: str | None = None) -> dict:
        """(SSE_ref - SSE_cand) / SST per seed (pooled-R2 units; positive favours the candidate); bootstrap; LOCO.
        mask_col: restrict both SSEs to rows with mask_col true (block decomposition; SST stays the frame's)."""
        if mask_col is not None:
            m = F[mask_col].to_numpy(bool)
            F = F.assign(_r=np.where(m, F[col_ref].to_numpy(float), 0.0), _c=np.where(m, F[col_cand].to_numpy(float), 0.0))
            col_ref, col_cand = "_r", "_c"
        S = self.sums(F, [col_ref, col_cand])
        per, draws, loco = {}, [], {}
        for s, a in S.items():
            sst = float(a[:, 2].sum() - a[:, 1].sum() ** 2 / a[:, 0].sum())
            per[s] = float(a[:, 3].sum() - a[:, 4].sum()) / sst if sst > 0 else np.nan
            wa = self.W @ a
            draws.append((wa[:, 3] - wa[:, 4]) / self._sst(wa))
        present = [self.CL[g] for g in range(len(self.CL)) if any(a[g, 0] > 0 for a in S.values())]
        for c in present:
            g = self.pos[c]
            v = []
            for a in S.values():
                b = np.delete(a, g, 0).sum(0)
                sst = b[2] - b[1] ** 2 / b[0] if b[0] > 1 else np.nan
                v.append((b[3] - b[4]) / sst if b[0] > 1 and sst > 0 else np.nan)
            loco[c] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        return {"per_seed": pd.Series(per, dtype=float), "draws": np.mean(np.vstack(draws), 0) if draws else np.array([]),
                "loco": loco}


def r2_seeds(F: pd.DataFrame, arm: str) -> pd.Series:
    """Pooled R2 per seed from the squared-error column (NaN when undefined; never clipped)."""
    out = {}
    for s, g in F.groupby("split_seed"):
        y = g.y.to_numpy(float)
        sst = float(((y - y.mean()) ** 2).sum())
        out[int(s)] = 1.0 - float(g["se_" + arm].sum()) / sst if len(g) >= 2 and sst > 0 else np.nan
    return pd.Series(out, dtype=float)


def seed_mean(F: pd.DataFrame, col: str, how: str = "mean") -> float:
    return float(F.groupby("split_seed")[col].agg(how).mean()) if len(F) else float("nan")


def pair_frame(F1: pd.DataFrame, a1: str, F2: pd.DataFrame, a2: str) -> pd.DataFrame:
    """One frame holding the reference (F1, a1) and candidate (F2, a2) errors on identical rows."""
    if not same_rows(F1, F2):
        raise AssertionError("contrast on non-identical rows")
    P = F2[KEY + ["y", "publication", "cluster_cpg", "chemotype", "k_max", "n_metals", "pair", "block"]].copy()
    P["se_r"], P["se_c"] = F1["se_" + a1].to_numpy(float), F2["se_" + a2].to_numpy(float)
    P["ae_r"], P["ae_c"] = F1["ae_" + a1].to_numpy(float), F2["ae_" + a2].to_numpy(float)
    return P


def _d_seeds(P: pd.DataFrame) -> pd.Series:
    out = {}
    for s, g in P.groupby("split_seed"):
        y = g.y.to_numpy(float)
        sst = float(((y - y.mean()) ** 2).sum())
        out[int(s)] = (float(g.se_r.sum()) - float(g.se_c.sum())) / sst if sst > 0 else np.nan
    return pd.Series(out, dtype=float)


def unit_stats_se(P: pd.DataFrame, unit: str = "cluster_cpg") -> pd.DataFrame:
    """r5.common.unit_stats on squared-error columns (ref = se_r, cand = se_c)."""
    g = pd.DataFrame({"u": P[unit].astype(str).to_numpy(), "s": P.split_seed.to_numpy(), "one": 1.0,
                      "er": P.se_r.to_numpy(float), "ec": P.se_c.to_numpy(float)})
    ps = g.groupby(["u", "s"])[["one", "er", "ec"]].sum()
    m = ps.groupby(level="u").mean()
    out = pd.DataFrame({"n": m.one, "SSE_ref": m.er, "SSE_cand": m.ec, "n_seeds": ps.groupby(level="u").size()})
    out.index.name = unit
    out["dSSE"] = out.SSE_ref - out.SSE_cand
    with np.errstate(divide="ignore", invalid="ignore"):
        out["L"] = np.log(out.SSE_ref / out.SSE_cand)
        out["rel"] = 1.0 - out.SSE_cand / out.SSE_ref
    out["win"], out["loss"], out["tie"] = out.dSSE > 0, out.dSSE < 0, out.dSSE == 0
    return out


def _span(x) -> str:
    return ";".join(f"{v:.4f}" for v in np.asarray(x, float))


def level_row(ST: Stats, F: pd.DataFrame, arm: str, *, lopo: bool = False, thr: float = 0.50) -> dict:
    """R2 (seed mean, per seed, seeds >= thr), MSE, MAE, bootstrap P(R2 >= thr) and 95 % percentile, LOCO and (when
    asked) leave-one-publication-out range; population counts; top-3 cluster share of SSE."""
    if F.empty:
        return {"rows_per_seed": 0}
    L = ST.level(F, "se_" + arm)
    r = L["per_seed"]
    d = L["draws"]
    sse_cl = F.groupby(["cluster_cpg", "split_seed"], observed=True)["se_" + arm].sum().groupby(level=0, observed=True).mean()
    out = {"rows_per_seed": len(F) / F.split_seed.nunique(), "cells": F.cell_id.nunique(),
           "publications": F.publication.nunique(), "clusters": F.cluster_cpg.nunique(),
           "R2": float(r.mean()), "R2_min": float(r.min()), "R2_max": float(r.max()), "per_seed": _span(r),
           "seeds_ge_050": int((r >= thr).sum()), "MSE": seed_mean(F, "se_" + arm), "MAE": seed_mean(F, "ae_" + arm),
           "P_boot_ge_050": float((d >= thr).mean()), "boot_lo": float(np.quantile(d, 0.025)),
           "boot_hi": float(np.quantile(d, 0.975)), "loco_min": float(np.nanmin(list(L["loco"].values()))),
           "loco_max": float(np.nanmax(list(L["loco"].values()))),
           "top3_cluster_SSE_share": CM.top3_share(sse_cl.to_numpy(float))}
    if lopo:
        lp = CM.lopo(F, lambda t: float(r2_seeds(t, arm).mean()), by="publication")
        vals = np.array([v for v in lp["values"].values() if np.isfinite(v)])
        out.update({"lopo_min": lp["min"], "lopo_min_by": lp["min_by"], "lopo_max": lp["max"], "lopo_max_by": lp["max_by"],
                    "lopo_n_ge_050": int((vals >= thr).sum()), "lopo_P": lp["P"]})
    return out


def contrast_row(ST: Stats, F1, a1, F2, a2, *, lopo: bool = True) -> dict:
    """Paired contrast on identical rows (candidate F2/a2 minus reference F1/a1) in pooled-R2 units; relative SSE,
    dMAE, per seed, bootstrap interval / sd / implied MDE, LOCO, LOPO, unit W/L/T and the top-3 unit share."""
    P = pair_frame(F1, a1, F2, a2)
    D = ST.delta(P, "se_r", "se_c")
    dl = D["per_seed"]
    rel = {int(s): 1.0 - float(g.se_c.sum()) / float(g.se_r.sum()) for s, g in P.groupby("split_seed")}
    dmae = {int(s): float(g.ae_r.mean() - g.ae_c.mean()) for s, g in P.groupby("split_seed")}
    us = unit_stats_se(P)
    out = {"rows_per_seed": len(P) / P.split_seed.nunique(), "delta_R2": float(dl.mean()), "seeds_pos": int((dl > 0).sum()),
           "per_seed": _span(dl), "relSSE": float(np.mean(list(rel.values()))), "dMAE": float(np.mean(list(dmae.values()))),
           "boot_lo": float(np.quantile(D["draws"], 0.025)), "boot_hi": float(np.quantile(D["draws"], 0.975)),
           "boot_sd": float(D["draws"].std()), "MDE80": MDE_MULT * float(D["draws"].std()),
           "loco_min": float(np.nanmin(list(D["loco"].values()))), "loco_max": float(np.nanmax(list(D["loco"].values()))),
           "units_WLT": f"{int(us.win.sum())}/{int(us.loss.sum())}/{int(us.tie.sum())}",
           "top3_unit_share": CM.top3_share(us.dSSE.to_numpy(float)),
           "top3_unit_share_abs": CM.top3_share(np.abs(us.dSSE.to_numpy(float)))}
    if lopo:
        lp = CM.lopo(P, lambda t: float(_d_seeds(t).mean()), by="publication")
        out.update({"lopo_min": lp["min"], "lopo_min_by": lp["min_by"], "lopo_max": lp["max"], "lopo_max_by": lp["max_by"]})
    return out


def unit_table(P: pd.DataFrame, key: str) -> tuple[pd.DataFrame, dict]:
    """The 13-row cluster_cpg unit table of one contrast and its section-4.2 descriptive tests (DIAGNOSTIC)."""
    us = unit_stats_se(P)
    has_c5 = P.groupby("cluster_cpg", observed=True).publication.agg(lambda s: bool((s.astype(str) == paths.C5).any()))
    us["has_C5"] = us.index.map(lambda c: bool(has_c5.get(c, False)))
    us["pubs"] = us.index.map(P.groupby("cluster_cpg", observed=True).publication.nunique().to_dict())
    tests = CM.unit_tests(us, key) if len(us) else {}
    return us.reset_index(), tests


# ------------------------------------------------------------------------------------------------------------
# analysis tables
# ------------------------------------------------------------------------------------------------------------
def _rec(design, family, frame, arm, var, **kw) -> dict:
    return {"design": design, "estimand": ESTIMAND, "family": family, "frame": frame, "arm": arm, "variant": var, **kw}


def cohort_frames(G) -> dict:
    out = {f"C3pop|r{r}": lad_frame(G, "C3pop", r) for r in POP_READS}
    for c, rr in COHORT_READS.items():
        if c == "C3":
            continue
        for r in rr:
            out[f"{c}|r{r}"] = lad_frame(G, c, r)
    return out


E_A_RUNG = "C3pop|r4"               # the C3 4-read rung is E-A itself (brief header, 3.3, 10 Q1)
LEVEL_FIELDS = ("R2", "R2_min", "R2_max", "per_seed", "seeds_ge_050", "MSE", "MAE", "P_boot_ge_050", "boot_lo", "boot_hi",
                "loco_min", "loco_max", "top3_cluster_SSE_share", "lopo_min", "lopo_max", "lopo_n_ge_050", "R2_B0",
                "relSSE_vs_B1_4reads")


def ea_blank(rec: dict) -> dict:
    """s0-fidelity-9 writer rule: the C3 4-read cell (E-A rows) reads "(E-A rows)" and no level is printed (every level
    field blanked; identities, counts and paired contrasts stay)."""
    rec = {k: (np.nan if k in LEVEL_FIELDS else v) for k, v in rec.items()}
    rec["level_note"] = EA_NOTE
    return rec


def rel_sse_same_rows(F: pd.DataFrame, arm: str, ref: pd.DataFrame, ref_arm: str = "B1") -> float:
    """Relative SSE vs B1 at 4 reads on the same rows: seed mean of 1 - SSE_arm(F) / SSE_B1(ref) (NaN unless the rows
    are identical)."""
    if ref is None or not same_rows(F, ref):
        return float("nan")
    return float(np.mean([1 - float(g["se_" + arm].sum()) / float(h["se_" + ref_arm].sum())
                          for (_, g), (_, h) in zip(F.groupby("split_seed"), ref.groupby("split_seed"))]))


def rung_estimand(name: str) -> str:
    r = int(name.split("|r")[1])
    if name.startswith("C3pop") and r == 4:
        return "E-A rung (4 reads; B1 = the E-A reference)"
    return "E-D 4-read rung of a fixed cohort" if r == 4 else "E-D (different read budget)"


def ladder_tables(G, ST, design, variants=VARIANTS, lopo=True, log=print) -> dict:
    """M18: level rows of every rung and arm; paired rung contrasts and arm contrasts on identical rows."""
    frames = cohort_frames(G)
    lv, ct = [], []
    for var in variants:
        Fv = {k: variant(v, var) for k, v in frames.items()}
        for name, F in Fv.items():
            coh, r = name.split("|")[0], reads_of(name)
            fewer = F[F.n_read.to_numpy() < r]            # C3 min rule: rows that keep min(r, k_max + 1) < r reads
            for arm in ARMS:
                rec = _rec(design, "LADDER", name, arm, var, row=row_name(name), rung_estimand=rung_estimand(name),
                           label=label_for(arm, "DOPTQ", var, design, "different read budget" if "different" in rung_estimand(name) else ""),
                           **level_row(ST, F, arm, lopo=lopo and arm == "B1"))
                rec["relSSE_vs_B1_4reads"] = rel_sse_same_rows(F, arm, Fv[f"{coh}|r4"])
                if coh == "C3pop":            # the mixture of the C3 row (brief 6.6: 286 of 1,802 rows keep 4 reads)
                    rec["rows_fewer_reads_per_seed"] = len(fewer) / max(F.split_seed.nunique(), 1)
                    rec["R2_rows_fewer_reads"] = float(r2_seeds(fewer, arm).mean()) if len(fewer) else np.nan
                if name == E_A_RUNG:          # s0-fidelity-9: the C3 4-read cell reads "(E-A rows)"; no level printed
                    rec = ea_blank(rec)
                lv.append(rec)
        for coh in ["C3pop"] + [c for c in COHORT_READS if c != "C3"]:
            rr = POP_READS if coh == "C3pop" else COHORT_READS[coh]
            pairs = {(r - 1, r) for r in rr if r - 1 in rr} | {(4, r) for r in rr if r >= 6} | {(2, 4)}
            for r0, r1 in sorted(pairs):
                F0, F1 = Fv[f"{coh}|r{r0}"], Fv[f"{coh}|r{r1}"]
                ct.append(_rec(design, "LADDER", f"{coh}: r{r1} - r{r0}", "B1", var,
                               row=f"{row_name(f'{coh}|r{r1}')} - {r0} reads (identical queries)",
                               label=label_for("B1", "DOPTQ", var, design, "different read budget (paired, identical queries)"),
                               **contrast_row(ST, F0, "B1", F1, "B1", lopo=lopo)))
            for r in rr:
                F = Fv[f"{coh}|r{r}"]
                for ref, lab in (("B0", ""), ("HIST_D7", "DIAGNOSTIC (HIST_D7 reference)"),
                                 ("B1_k3beta", "SENSITIVITY (beta refit vs k = 3 beta)")):
                    if ref == "B1_k3beta" and r == 4:
                        continue
                    ct.append(_rec(design, "LADDER", f"{coh}|r{r}: B1 - {ref}", "B1", var, row=row_name(f"{coh}|r{r}"),
                                   label=label_for("B1", "DOPTQ", var, design, budget_tag(r, extra=lab)),
                                   **contrast_row(ST, F, ref, F, "B1", lopo=False)))
        log(f"    ladder {design} {var}: {len(lv)} level rows, {len(ct)} contrasts")
    return {"ladder_levels": pd.DataFrame(lv), "ladder_contrasts": pd.DataFrame(ct)}


def pareto_answers(rows: list, design: str, coh: str, var: str) -> list:
    """The answer to the pre-declared Pareto question (REGISTRY 1.6-R01: 'does any member ...') over the written
    label-free family only (ENV_LABEL_FREE; DOPTQ is the reference), with ORACLE_B1 [O] on its own row as a bound
    (never a member) and the LOOSE rule on its own SENSITIVITY row.  ``rows``: the per-policy Pareto rows of one frame."""
    groups = (("ANSWER: written label-free family", "answer (1.6-R01): family members", "",
               [p for p in ENV_LABEL_FREE if p != "DOPTQ"],
               "pre-declared Pareto question 1.6-R01 over the written label-free family (descriptive)"),
              ("ANSWER: ORACLE_B1 [O] bound", "answer: ORACLE [O] bound (never a member)", "ORACLE_B1", ["ORACLE_B1"],
               "bound only; dominates by construction (the legal 5th metal minimising the cell's B1 SSE)"),
              ("ANSWER: LOOSE legality", "answer: SENSITIVITY (LOOSE legality; never a member)", "MAXMIN_R_LOOSE",
               ["MAXMIN_R_LOOSE", "RANDOM_LOOSE"], "LOOSE-legality sensitivity of the Pareto question"))
    out = []
    for name, role, lab_policy, pols, extra in groups:
        sel = [r for r in rows if r["policy"] in pols]
        dom = [r["policy"] for r in sel if r["pareto_dominates"]]
        if coh != "C3pop":
            extra += "; the pre-declared frame is C3 min(4, k_max), this frame is descriptive"
        out.append(_rec(design, "ENVELOPE", coh, "B1", var, policy=name, role=role, row=row_name(f"{coh}|r5"),
                        label=label_for("B1", lab_policy, var, design, join_labels(RB, extra)),
                        policies=",".join(pols), n_policies=len(sel), pareto_dominates=bool(dom),
                        dominating_policies=",".join(dom),
                        keeps_every_block_above_floor_policies=",".join(r["policy"] for r in sel
                                                                       if r["keeps_every_block_above_floor"])))
    return out


def envelope_tables(G, ST, cand: pd.DataFrame, design, variants=VARIANTS, lopo=True, log=print) -> dict:
    """M17: 5th-read policy levels, contrasts, blocks, Pareto answer, per pair against FLAT, k_max and cluster splits,
    envelope width, RANDOM Monte Carlo."""
    lv, ct, bl, pa, pp, km, pc, ew, mc = [], [], [], [], [], [], [], [], []
    for var in variants:
        for coh in ("C3pop", "C4"):
            F = {p: variant(env_frame(G, coh, p), var) for p in ENV_POLICIES}
            K3 = variant(lad_frame(G, coh, 4), var)
            for p, X in F.items():
                if not same_rows(X, K3):
                    raise AssertionError(f"ENV {p} {coh}: rows differ from the 4-read frame")
                for arm in ARMS:
                    lv.append(_rec(design, "ENVELOPE", f"{coh}|r5|{p}", arm, var, policy=p, row=row_name(f"{coh}|r5"),
                                   label=label_for(arm, p, var, design, "different read budget"),
                                   **level_row(ST, X, arm, lopo=lopo and arm == "B1" and p in ("DOPTQ", "BISECT_R", "RANDOM"))))
            specs = [(f"{p} - DOPTQ", F["DOPTQ"], "B1", F[p], "B1", p) for p in ENV_POLICIES if p != "DOPTQ"]
            specs += [(f"{p} (5 reads) - 4 reads", K3, "B1", F[p], "B1", p) for p in ENV_POLICIES]
            specs += [("BISECT_R - RANDOM", F["RANDOM"], "B1", F["BISECT_R"], "B1", "BISECT_R"),
                      ("DOPTQ: B1 - B0", F["DOPTQ"], "B0", F["DOPTQ"], "B1", "DOPTQ"),
                      ("DOPTQ: B1 - HIST_D7", F["DOPTQ"], "HIST_D7", F["DOPTQ"], "B1", "DOPTQ"),
                      ("DOPTQ: B1 refit k4 - B1 k3 beta", F["DOPTQ"], "B1_k3beta", F["DOPTQ"], "B1", "DOPTQ")]
            for name, F1, a1, F2, a2, p in specs:
                ct.append(_rec(design, "ENVELOPE", f"{coh}: {name}", "B1", var, policy=p, row=row_name(f"{coh}|r5"),
                               label=label_for("B1", p, var, design, "different read budget"),
                               **contrast_row(ST, F1, a1, F2, a2, lopo=lopo and p in ("BISECT_R", "RANDOM", "HULL_BISECT"))))
            # per-block delta in pooled-R2 units, vs DOPTQ and vs 4 reads; Pareto question vs DOPTQ
            for ref_name, REF in (("DOPTQ", F["DOPTQ"]), ("4 reads", K3)):
                for p in ENV_POLICIES:
                    if ref_name == "DOPTQ" and p == "DOPTQ":
                        continue
                    P = pair_frame(REF, "B1", F[p], "B1")
                    rec = _rec(design, "ENVELOPE", coh, "B1", var, policy=p, vs=ref_name,
                               label=label_for("B1", p, var, design, "different read budget"))
                    for b in BLOCKS + ("all",):
                        P["_m"] = True if b == "all" else (P.block.astype(str) == b)
                        D = ST.delta(P, "se_r", "se_c", mask_col="_m")
                        rec.update({b: float(D["per_seed"].mean()), f"{b}_seeds_pos": int((D["per_seed"] > 0).sum()),
                                    f"{b}_lo": float(np.quantile(D["draws"], 0.025)),
                                    f"{b}_hi": float(np.quantile(D["draws"], 0.975))})
                    bl.append(rec)
                    if ref_name == "DOPTQ":
                        keeps = all(rec[b] > PARETO_FLOOR for b in BLOCKS)
                        improves = [b for b in BLOCKS if rec[b] > 0 and rec[f"{b}_lo"] > 0]
                        pa.append(_rec(design, "ENVELOPE", coh, "B1", var, policy=p, role=PARETO_ROLE[p],
                                       row=row_name(f"{coh}|r5"),
                                       label=label_for("B1", p, var, design, "different read budget; pre-declared Pareto "
                                                                              "question (descriptive)"),
                                       keeps_every_block_above_floor=keeps, blocks_improved_lo_gt_0=",".join(improves),
                                       pareto_dominates=bool(keeps and improves),
                                       **{b: rec[b] for b in BLOCKS}, **{f"{b}_lo": rec[f"{b}_lo"] for b in BLOCKS}))
            pa += pareto_answers([x for x in pa if x["frame"] == coh and x["variant"] == var and "answer" not in x["role"]],
                                 design, coh, var)
            # envelope width over the label-free family
            r2s = {p: r2_seeds(F[p], "B1") for p in ENV_LABEL_FREE}
            means = {p: float(v.mean()) for p, v in r2s.items()}
            allps = np.concatenate([v.to_numpy() for v in r2s.values()])
            ew.append(_rec(design, "ENVELOPE", coh, "B1", var, row=row_name(f"{coh}|r5"),
                           label=label_for("B1", "", var, design, "different read budget"),
                           envelope=max(means.values()) - min(means.values()), lowest=min(means, key=means.get),
                           highest=max(means, key=means.get), per_seed_span_min=float(np.nanmin(allps)),
                           per_seed_span_max=float(np.nanmax(allps)), RANDOM_as="exact expectation",
                           note="RANDOM enters as its exact expectation (uniform over the legal set, independent across "
                                "cells; ITEM_SPEC)"))
            if coh != "C3pop":
                continue
            # per pair against FLAT (pred 0): MAE skill; R2 within the pair type
            for p in ENV_POLICIES:
                X = F[p]
                for pt in PAIR_ORDER:
                    g = X[X.pair.astype(str) == pt]
                    if g.empty:
                        continue
                    mae_flat = seed_mean(g.assign(_f=g.y.abs()), "_f")
                    rec = _rec(design, "ENVELOPE", "C3pop|r5", "B1", var, policy=p, pair=pt,
                               label=label_for("B1", p, var, design, "different read budget"),
                               rows_per_seed=len(g) / g.split_seed.nunique(), R2=float(r2_seeds(g, "B1").mean()),
                               MAE=seed_mean(g, "ae_B1"), MAE_B0=seed_mean(g, "ae_B0"), MAE_FLAT=mae_flat,
                               skill_vs_FLAT=1.0 - seed_mean(g, "ae_B1") / mae_flat if mae_flat > 0 else np.nan,
                               hull_out_share=float(g.hull_out.mean()),
                               SSE_share=float(g.se_B1.sum() / X.se_B1.sum()))
                    pp.append(rec)
            for pt in PAIR_ORDER:                                    # the 4-read reference row
                g = K3[K3.pair.astype(str) == pt]
                if g.empty:
                    continue
                mae_flat = seed_mean(g.assign(_f=g.y.abs()), "_f")
                pp.append(_rec(design, "ENVELOPE", "C3pop|r4", "B1", var, policy="K3 (4 reads)", pair=pt,
                               level_note=EA_STRATUM_NOTE,
                               label=label_for("B1", "", var, design), rows_per_seed=len(g) / g.split_seed.nunique(),
                               R2=float(r2_seeds(g, "B1").mean()), MAE=seed_mean(g, "ae_B1"), MAE_B0=seed_mean(g, "ae_B0"),
                               MAE_FLAT=mae_flat, skill_vs_FLAT=1.0 - seed_mean(g, "ae_B1") / mae_flat if mae_flat > 0 else np.nan,
                               hull_out_share=float(g.hull_out.mean()), SSE_share=float(g.se_B1.sum() / K3.se_B1.sum())))
            # by k_max stratum and by cluster
            for p in ENV_POLICIES:
                X = F[p]
                for k, g in X.groupby("k_max"):
                    g0 = F["DOPTQ"][F["DOPTQ"].k_max == k]
                    km.append(_rec(design, "ENVELOPE", "C3pop|r5", "B1", var, policy=p, k_max=int(k),
                                   label=label_for("B1", p, var, design, "different read budget"),
                                   rows_per_seed=len(g) / g.split_seed.nunique(), cells=g.cell_id.nunique(),
                                   clusters=g.cluster_cpg.nunique(), R2=float(r2_seeds(g, "B1").mean()),
                                   R2_B0=float(r2_seeds(g, "B0").mean()),
                                   dR2_vs_DOPTQ=float((r2_seeds(g, "B1") - r2_seeds(g0, "B1")).mean())))
                sst = X.groupby("split_seed").y.apply(lambda v: float(((v - v.mean()) ** 2).sum()))
                for c, g in X.groupby("cluster_cpg", observed=True):
                    g0 = F["DOPTQ"][F["DOPTQ"].cluster_cpg == c]
                    contrib = ((g0.groupby("split_seed").se_B1.sum() - g.groupby("split_seed").se_B1.sum()) / sst).mean()
                    rest = X[X.cluster_cpg != c]
                    pc.append(_rec(design, "ENVELOPE", "C3pop|r5", "B1", var, policy=p, cluster_cpg=str(c),
                                   label=label_for("B1", p, var, design, "different read budget"),
                                   rows_per_seed=len(g) / g.split_seed.nunique(), cells=g.cell_id.nunique(),
                                   pubs=g.publication.nunique(), has_C5=bool((g.publication.astype(str) == paths.C5).any()),
                                   R2_within=float(r2_seeds(g, "B1").mean()),
                                   pooled_R2_without=float(r2_seeds(rest, "B1").mean()),
                                   contribution_vs_DOPTQ=float(contrib)))
            # RANDOM Monte Carlo over joint draws (B1)
            mc.append(_rec(design, "ENVELOPE", "C3pop|r5", "B1", var, policy="RANDOM",
                           label=label_for("B1", "RANDOM", var, design, "different read budget; Monte Carlo"),
                           **random_mc(cand, K3, design, var)))
            # the width with the Monte Carlo mean in place of the exact RANDOM expectation (disclosure: the seen BP
            # 0.0096 is 0.0095 with the exact expectation the item uses; the difference is expected, not a mismatch)
            mcm = dict(means, RANDOM=mc[-1]["mc_mean"])
            ew[-1].update({"envelope_with_RANDOM_MC_mean": max(mcm.values()) - min(mcm.values()),
                           "seen_envelope": SEEN_ENVELOPE.get(design, np.nan) if var == "all" else np.nan,
                           "note": ew[-1]["note"] + "; the seen value [P3] is quoted to four decimals; with the "
                                                    "RANDOM Monte Carlo mean in place of the exact expectation the width "
                                                    "is envelope_with_RANDOM_MC_mean, so a last-digit difference from "
                                                    "the seen value is expected (not a reproduction target)"})
        log(f"    envelope {design} {var}: {len(lv)} levels, {len(ct)} contrasts, {len(bl)} block rows")
    return {"envelope_levels": pd.DataFrame(lv), "envelope_contrasts": pd.DataFrame(ct), "envelope_blocks": pd.DataFrame(bl),
            "envelope_pareto": pd.DataFrame(pa), "envelope_per_pair": pd.DataFrame(pp), "envelope_by_kmax": pd.DataFrame(km),
            "envelope_by_cluster": pd.DataFrame(pc), "envelope_width": pd.DataFrame(ew), "envelope_random_mc": pd.DataFrame(mc)}


def random_mc(cand: pd.DataFrame, K3: pd.DataFrame, design: str, var: str, reps: int = MC_REPS) -> dict:
    """P3's Monte Carlo of the RANDOM legal 5th read (B1): independent uniform draws per cell, k_max = 3 cells fixed at
    4 reads; SST per seed from the identical C3 rows."""
    rng = np.random.default_rng(mc_seed(design, var))
    c = cand[cand.strict_legal.astype(bool)]
    c = variant(c, var) if var != "noC5cell" else c[c.cell_id.astype(str) != paths.C5_CELL]
    draws, ge = [], []
    for s in sorted(K3.split_seed.unique()):
        g3 = K3[K3.split_seed == s]
        sst = float(((g3.y - g3.y.mean()) ** 2).sum())
        tot = np.full(reps, float(g3[g3.k_max < 4].se_B1.sum()))
        cs = c[c.split_seed == s]
        for _, v in cs.groupby("cell_id", observed=True):
            vv = v.sse_B1.to_numpy(float)
            tot += vv[rng.integers(len(vv), size=reps)]
        r = 1.0 - tot / sst
        draws.append(r)
        ge.append(float((r >= 0.5).mean()))
    dm = np.mean(np.vstack(draws), 0)
    return {"mc_mean": float(dm.mean()), "mc_sd_seedmean": float(dm.std()), "mc_q05": float(np.quantile(dm, 0.05)),
            "mc_q95": float(np.quantile(dm, 0.95)), "P_seedmean_ge_050": float((dm >= 0.5).mean()),
            "P_seed_ge_050": ";".join(f"{x:.3f}" for x in ge), "E_n_seeds_ge_050": float(sum(ge)),
            "rng_seed": mc_seed(design, var), "reps": reps}


def nm_class(n: int) -> str:
    return "14" if n == 14 else "13" if n == 13 else "8-12" if n >= 8 else "<=7"


def density_tables(G, ST, design, variants=("all", "noC5"), log=print) -> dict:
    """Density and saturation (M18): population view, rows by their cell's density and longest hidden run, series-
    length strata, saturation strata k_max j at j + 1 reads with the density-invariance consistency check."""
    frames = cohort_frames(G)
    pv, db, hb, sl, sa, sc = [], [], [], [], [], []
    for var in variants:
        Fv = {k: variant(v, var) for k, v in frames.items()}
        envF = {p: variant(env_frame(G, "C3pop", p), var) for p in ENV_LABEL_FREE}
        recF = {(p, r): variant(rec_frame(G, p, r), var) for p in REC_POLICIES for r in REC_READS}
        base4 = {"C3pop": Fv["C3pop|r4"], **{c: Fv[f"{c}|r4"] for c in ("C4", "C5", "C6")}}
        srcs = {f"LADDER {k}": v for k, v in Fv.items()}
        srcs.update({f"ENVELOPE C3pop|r5|{p}": v for p, v in envF.items()})
        srcs.update({f"RECIPE C3pop|r{r}|{p}": v for (p, r), v in recF.items()})
        for name, F in srcs.items():
            coh = name.split()[1].split("|")[0]
            ref = base4.get(coh)
            rel = float(np.mean([1 - float(g.se_B1.sum()) / float(h.se_B1.sum())
                                 for (_, g), (_, h) in zip(F.groupby("split_seed"), ref.groupby("split_seed"))])) \
                if ref is not None and same_rows(F, ref) else np.nan
            rec = _rec(design, "DENSITY", name, "B1", var, row=row_name(name),
                       label=label_for("B1", "", var, design, budget_tag(reads_of(name))),
                       mean_density_rows=float((F.n_read / F.n_metals).mean()),
                       mean_reads_per_cell=float(F.groupby(["split_seed", "cell_id"], observed=True).n_read.first().mean()),
                       hull_out_share=float(F.hull_out.mean()), mean_longest_hidden_run=float(F.lhr.mean()),
                       mean_out_share=float(F.out_share.mean()), R2=float(r2_seeds(F, "B1").mean()),
                       R2_B0=float(r2_seeds(F, "B0").mean()), MSE=seed_mean(F, "se_B1"),
                       relSSE_vs_B1_4reads_same_rows=rel)
            pv.append(ea_blank(rec) if is_ea_cell(name) else rec)      # s0-fidelity-9: the C3 4-read cell
        # rows by their cell's density and longest hidden run
        bsrc = {"4 reads (C3 rows)": (Fv["C3pop|r4"], 4)}
        for p in ("DOPTQ", "MAXMIN_R", "BISECT_R", "RANDOM"):
            X = envF[p]
            bsrc[f"5 reads ({p} rows)"] = (X[X.n_read == 5], 5)
        bsrc["6 reads (C5 rows)"] = (Fv["C5|r6"], 6)
        bsrc["7 reads (C6 rows)"] = (Fv["C6|r7"], 7)
        bsrc["pooled 4+5 (C3 min(k, k_max) at 5 reads)"] = (Fv["C3pop|r5"], 5)
        for name, (F, nr) in bsrc.items():
            dens = (F.n_read / F.n_metals).to_numpy(float)
            for col, bins, labs, out in (("density", DBINS, DLAB, db), ("lhr", HBINS, HLAB, hb)):
                v = dens if col == "density" else F.lhr.to_numpy(float)
                cut = pd.cut(v, bins, labels=labs, right=False if col == "density" else True)
                for b_ in labs:
                    g = F[np.asarray(cut == b_)]
                    if g.empty:
                        continue
                    out.append(_rec(design, "DENSITY", name, "B1", var, label=label_for("B1", "", var, design, budget_tag(nr)),
                                    **({"level_note": EA_STRATUM_NOTE} if nr == 4 else {}),
                                    bin=b_, R2=float(r2_seeds(g, "B1").mean()), R2_B0=float(r2_seeds(g, "B0").mean()),
                                    MSE=seed_mean(g, "se_B1"), rows_per_seed=len(g) / g.split_seed.nunique(),
                                    clusters=g.cluster_cpg.nunique(), mean_density=float((g.n_read / g.n_metals).mean())))
        # series-length strata; relative SSE vs B1 at 4 reads on the same rows next to R2 (ITEM_SPEC)
        for name, F in list(srcs.items()):
            if not (name.startswith("LADDER") or name.endswith(("|DOPTQ", "|LADDER", "|HULL"))):
                continue
            ref = base4.get(name.split()[1].split("|")[0])
            refc = ref.n_metals.map(nm_class) if ref is not None and same_rows(F, ref) else None
            for nc, g in F.groupby(F.n_metals.map(nm_class)):
                sl.append(_rec(design, "DENSITY", name, "B1", var, row=row_name(name),
                               label=label_for("B1", "", var, design, budget_tag(reads_of(name))),
                               **({"level_note": EA_STRATUM_NOTE} if is_ea_cell(name) else {}),
                               series_length=nc, R2=float(r2_seeds(g, "B1").mean()), R2_B0=float(r2_seeds(g, "B0").mean()),
                               MSE=seed_mean(g, "se_B1"),
                               relSSE_vs_B1_4reads_same_rows=(rel_sse_same_rows(g, "B1", ref[(refc == nc).to_numpy()])
                                                              if refc is not None else np.nan),
                               rows_per_seed=len(g) / g.split_seed.nunique(),
                               cells=g.cell_id.nunique(), clusters=g.cluster_cpg.nunique(),
                               hull_out=float(g.hull_out.mean()), mean_density=float((g.n_read / g.n_metals).mean())))
        # saturation: reads = k_max + 1
        strata = []
        for j in (3, 4, 5, 6):
            F = Fv[f"C{j}|r{j + 1}"] if j > 3 else Fv["C3pop|r4"]
            g = F[F.k_max == j]
            if g.empty:
                continue
            L = ST.level(g, "se_B1")
            vc = g.cluster_cpg.value_counts(normalize=True)
            rec = _rec(design, "SATURATION", f"k_max {j} at {j + 1} reads", "B1", var,
                       label=label_for("B1", "", var, design, join_labels("CONSISTENCY CHECK", budget_tag(j + 1))),
                       **({"level_note": EA_STRATUM_NOTE} if j == 3 else {}), k_max=j, reads=j + 1,
                       R2=float(L["per_seed"].mean()), R2_B0=float(r2_seeds(g, "B0").mean()), MSE=seed_mean(g, "se_B1"),
                       rows_per_seed=len(g) / g.split_seed.nunique(), cells=g.cell_id.nunique(),
                       clusters=g.cluster_cpg.nunique(), largest_cluster_share=float(vc.iloc[0]),
                       mean_density=float((g.n_read / g.n_metals).mean()),
                       loco_min=float(np.nanmin(list(L["loco"].values()))) if L["loco"] else np.nan,
                       loco_max=float(np.nanmax(list(L["loco"].values()))) if L["loco"] else np.nan)
            sa.append(rec)
            strata.append(rec)
        elig = [s for s in strata if s["clusters"] >= SAT_MIN_CLUSTERS]
        vals = [s["R2"] for s in strata]
        low = [s["frame"] for s in elig if s["R2"] < SAT_FLOOR]
        spread = float(max(vals) - min(vals)) if vals else np.nan
        sc.append(_rec(design, "SATURATION", "density invariance at saturation", "B1", var,
                       label=label_for("B1", "", var, design, "CONSISTENCY CHECK (the k_max-4 value was known before the "
                                                               "rule was written; not a kill); strata at 5-7 reads = "
                                                               "different read budget"),
                       strata_values=";".join(f"{s['frame']}: {s['R2']:.4f} ({s['clusters']} clusters)" for s in strata),
                       rule=f"fires if a stratum with >= {SAT_MIN_CLUSTERS} clusters is below {SAT_FLOOR} or the spread "
                            f"is >= {SAT_SPREAD}", strata=len(strata), strata_low=",".join(low), spread=spread,
                       rule_fires=bool(low or (np.isfinite(spread) and spread >= SAT_SPREAD))))
        log(f"    density {design} {var}: {len(pv)} population rows, {len(sa)} saturation strata")
    return {"density_population": pd.DataFrame(pv), "density_bins": pd.DataFrame(db), "hidden_run_bins": pd.DataFrame(hb),
            "series_length": pd.DataFrame(sl), "saturation": pd.DataFrame(sa), "saturation_check": pd.DataFrame(sc)}


def recipe_tables(G, ST, design, selection: pd.DataFrame | None, variants=VARIANTS, lopo=True, log=print) -> dict:
    """recipe-2 row: policy levels at 2-5 reads by series length, Ladder - HULL (primary) and the other contrasts,
    coverage (static metals used, guard, LOCO applicability), the LOCO selections."""
    lv, ct, sl, cov = [], [], [], []
    for var in variants:
        F = {(p, r): variant(rec_frame(G, p, r), var) for p in REC_POLICIES for r in REC_READS}
        for (p, r), X in F.items():
            if not same_rows(X, F[("DOPTQ", r)]):
                raise AssertionError(f"REC {p} r{r}: rows differ")
            for arm in ("B1", "B0", "HIST_D7"):
                rec = _rec(design, "RECIPE2", f"C3pop|r{r}|{p}", arm, var, policy=p, reads=r, row=row_name(f"C3pop|r{r}"),
                           label=label_for(arm, p, var, design, "different read budget" if r != 4 else ""),
                           relSSE_vs_DOPTQ=float(np.mean([1 - float(g["se_" + arm].sum()) / float(h["se_" + arm].sum())
                                                          for (_, g), (_, h) in zip(X.groupby("split_seed"),
                                                                                    F[("DOPTQ", r)].groupby("split_seed"))])),
                           **level_row(ST, X, arm, lopo=lopo and arm == "B1" and p in ("LADDER", "HULL")))
                lv.append(ea_blank(rec) if is_ea_cell(f"C3pop|r{r}|{p}") else rec)   # s0-fidelity-9: DOPTQ at 4 reads
            for nc, g in X.groupby(X.n_metals.map(nm_class)):
                sl.append(_rec(design, "RECIPE2", f"C3pop|r{r}|{p}", "B1", var, policy=p, reads=r, series_length=nc,
                               **({"level_note": EA_STRATUM_NOTE} if is_ea_cell(f"C3pop|r{r}|{p}") else {}),
                               label=label_for("B1", p, var, design, budget_tag(r)), R2=float(r2_seeds(g, "B1").mean()),
                               R2_B0=float(r2_seeds(g, "B0").mean()), rows_per_seed=len(g) / g.split_seed.nunique(),
                               cells=g.cell_id.nunique(), clusters=g.cluster_cpg.nunique()))
        for r in REC_READS:
            specs = [("LADDER - HULL", "HULL", "LADDER", "B1", True), ("LADDER - HULL (B0)", "HULL", "LADDER", "B0", True),
                     ("LADDER - DOPTQ", "DOPTQ", "LADDER", "B1", False), ("HULL - RANDOM_SET", "RANDOM_SET", "HULL", "B1", False),
                     ("LOCO_STATIC - HULL", "HULL", "LOCO_STATIC", "B1", False),
                     ("LADDER_GUARD - LADDER", "LADDER", "LADDER_GUARD", "B1", False),
                     ("DOPTQ - HULL", "HULL", "DOPTQ", "B1", False)]
            for name, p0, p1, arm, primary in specs:
                ct.append(_rec(design, "RECIPE2", f"r{r}: {name}", arm, var, reads=r, primary=primary,
                               row=row_name(f"C3pop|r{r}"),
                               label=label_for(arm, p1, var, design, budget_tag(
                                   r, extra=join_labels("primary recipe-2 contrast" if primary else "",
                                                        "closure line: 'spread first' is definitional"
                                                        if p0 == "RANDOM_SET" else ""))),
                               **contrast_row(ST, F[(p0, r)], arm, F[(p1, r)], arm, lopo=lopo and primary)))
            Lf, Lc = F[("LADDER", r)], F[("LOCO_STATIC", r)]
            first = Lf.groupby(["split_seed", "cell_id"], observed=True).first()
            stat = first.read_set.astype(str).map(lambda s: set(s.split(",")) <= set(STATIC_ORDER[:r]))
            cov.append(_rec(design, "RECIPE2", f"r{r}", "", var, reads=r, label=label_for("B1", "", var, design, budget_tag(r)),
                            ladder_static_only_share=float(stat.mean()),
                            guard_cx_cells=int(Lf[Lf.cx.astype(bool)].cell_id.nunique()),
                            loco_fallback_share=float(Lc.groupby(["split_seed", "cell_id"], observed=True).fallback.first().mean()),
                            cells=int(Lf.cell_id.nunique())))
        log(f"    recipe-2 {design} {var}: {len(lv)} levels, {len(ct)} contrasts")
    out = {"recipe_levels": pd.DataFrame(lv), "recipe_contrasts": pd.DataFrame(ct), "recipe_series_length": pd.DataFrame(sl),
           "recipe_coverage": pd.DataFrame(cov)}
    if selection is not None:
        out["recipe_loco_selection"] = selection.assign(design=design, label=SEEN + "; LOCO selection by B0 SSE on the "
                                                                                   "other clusters' cells")
    return out


MAIN_UNIT_CONTRASTS = (("C3pop 5 vs 4 reads", ("LAD", "C3pop", 4), ("LAD", "C3pop", 5)),
                       ("C4 5 vs 4 reads", ("LAD", "C4", 4), ("LAD", "C4", 5)),
                       ("C6 7 vs 4 reads", ("LAD", "C6", 4), ("LAD", "C6", 7)),
                       ("BISECT_R - DOPTQ (C3pop, 5 reads)", ("ENV", "C3pop", "DOPTQ"), ("ENV", "C3pop", "BISECT_R")),
                       ("RANDOM - DOPTQ (C3pop, 5 reads)", ("ENV", "C3pop", "DOPTQ"), ("ENV", "C3pop", "RANDOM")),
                       ("BISECT_R - RANDOM (C3pop, 5 reads)", ("ENV", "C3pop", "RANDOM"), ("ENV", "C3pop", "BISECT_R")),
                       ("HULL_BISECT - DOPTQ (C3pop, 5 reads)", ("ENV", "C3pop", "DOPTQ"), ("ENV", "C3pop", "HULL_BISECT")),
                       ("LADDER - HULL (2 reads)", ("REC", "HULL", 2), ("REC", "LADDER", 2)),
                       ("LADDER - HULL (3 reads)", ("REC", "HULL", 3), ("REC", "LADDER", 3)),
                       ("LADDER - HULL (4 reads)", ("REC", "HULL", 4), ("REC", "LADDER", 4)),
                       ("LADDER - HULL (5 reads)", ("REC", "HULL", 5), ("REC", "LADDER", 5)))


def _get(G, spec):
    fam, a, b = spec
    return lad_frame(G, a, b) if fam == "LAD" else env_frame(G, a, b) if fam == "ENV" else rec_frame(G, a, b)


def _spec_reads(spec) -> int:
    """Read count of a (family, a, b) frame spec (ENV frames are 5-read)."""
    return 5 if spec[0] == "ENV" else int(spec[2])


def unit_tables(G, design, variants=("all", "noC5")) -> dict:
    """cluster_cpg unit tables of the main contrasts (B1) and their descriptive tests (DIAGNOSTIC: no in-corpus p is
    called significant)."""
    ut, tt = [], []
    for var in variants:
        for name, s0, s1 in MAIN_UNIT_CONTRASTS:
            P = pair_frame(variant(_get(G, s0), var), "B1", variant(_get(G, s1), var), "B1")
            us, tests = unit_table(P, f"ED|{design}|{var}|{name}")
            lab = label_for("B1", "", var, design, budget_tag(_spec_reads(s0), _spec_reads(s1),
                                                              extra="DIAGNOSTIC (descriptive test; never called significant)"))
            ut.append(us.assign(design=design, contrast=name, variant=var, label=lab))
            tt.append({"design": design, "contrast": name, "variant": var, "label": lab, "G": tests.get("G"),
                       "W/L/T": tests.get("W/L/T"), "SF_POOL_p": tests["SF_POOL"]["p"], "SIGN_p": tests["SIGN"]["p"],
                       "SF_LOG_p": tests["SF_LOG"]["p"] if tests.get("SF_LOG") else np.nan,
                       "IM_REL_p": tests["IM_REL"]["p"], "top3_unit_share": CM.top3_share(us.dSSE.to_numpy(float)),
                       "top3_unit_share_abs": CM.top3_share(np.abs(us.dSSE.to_numpy(float)))})
    return {"unit_tables": pd.concat(ut, ignore_index=True), "unit_tests": pd.DataFrame(tt)}


def duplicate_sets() -> dict:
    """Duplicate sensitivities: the round-3 V2 drop set (V2-1/2/3) and the second cell of every Stage-0 flagged
    bit-identity link (results/stage0/dupscan_links.csv, flag_bit)."""
    from . import population as PO
    out = {"noV2": set(PO.v2drop_set())}
    p = paths.RES / "stage0" / "dupscan_links.csv"
    if p.exists():
        d = pd.read_csv(p)
        flag = d.flag_bit.astype(str).str.lower() == "true"
        out["noBITDUP"] = set(d[flag].cell_b.astype(str))
    return out


def duplicate_tables(G, ST, design) -> pd.DataFrame:
    rows = []
    for v, cells in duplicate_sets().items():
        for name, s in (("C3pop|r4", ("LAD", "C3pop", 4)), ("C3pop|r5", ("LAD", "C3pop", 5)), ("C4|r4", ("LAD", "C4", 4)),
                        ("C4|r5", ("LAD", "C4", 5)), ("C3pop|r5|BISECT_R", ("ENV", "C3pop", "BISECT_R")),
                        ("C3pop|r2|LADDER", ("REC", "LADDER", 2)), ("C3pop|r2|HULL", ("REC", "HULL", 2))):
            F = variant(_get(G, s), v, cells)
            rec = _rec(design, "DUPLICATES", name, "B1", v, row=row_name(name),
                       label=label_for("B1", "", v, design, budget_tag(_spec_reads(s))),
                       dropped_cells=int(_get(G, s).cell_id.nunique() - F.cell_id.nunique()),
                       R2=float(r2_seeds(F, "B1").mean()), rows_per_seed=len(F) / max(F.split_seed.nunique(), 1))
            rows.append(ea_blank(rec) if is_ea_cell(name) else rec)         # s0-fidelity-9: the C3 4-read cell
        for name, s0, s1 in MAIN_UNIT_CONTRASTS[:2] + MAIN_UNIT_CONTRASTS[3:4] + MAIN_UNIT_CONTRASTS[7:8]:
            c = contrast_row(ST, variant(_get(G, s0), v, cells), "B1", variant(_get(G, s1), v, cells), "B1", lopo=False)
            rows.append(_rec(design, "DUPLICATES", name, "B1", v,
                             label=label_for("B1", "", v, design, budget_tag(_spec_reads(s0), _spec_reads(s1),
                                                                            extra="paired, identical queries")),
                             delta_R2=c["delta_R2"], seeds_pos=c["seeds_pos"], per_seed=c["per_seed"],
                             rows_per_seed=c["rows_per_seed"]))
    return pd.DataFrame(rows)


def analyse(R: pd.DataFrame, cand: pd.DataFrame, design: str, *, clusters, selection=None, lopo=True, log=print) -> dict:
    """Every table of the item for one design (row-level frames assembled on identical queries)."""
    t0 = time.time()
    G = split_rows(R)
    ST = Stats(clusters)
    T = {}
    T.update(ladder_tables(G, ST, design, lopo=lopo, log=log))
    T.update(envelope_tables(G, ST, cand, design, lopo=lopo, log=log))
    T.update(density_tables(G, ST, design, log=log))
    T.update(recipe_tables(G, ST, design, selection, lopo=lopo, log=log))
    T.update(unit_tables(G, design))
    T["duplicates"] = duplicate_tables(G, ST, design)
    log(f"    analysis {design}: {len(T)} tables ({time.time() - t0:.0f}s)")
    return T


# ------------------------------------------------------------------------------------------------------------
# reproduction targets (brief 6.0; TECHNICAL SMOKE)
# ------------------------------------------------------------------------------------------------------------
def repro_gate(out: dict) -> dict:
    """The identity part of the reproduction gate (ITEM_SPEC 'Reproduction targets'; a mismatch stops the item): every
    4-read row matched with B1 (and B0, HIST_D7) equal to round 4's REF within 8.9e-16, y exact, read sets identical;
    the 5-read B0 equal to round 3's R3-B0 PWL_R at k = 4 on every C4 row (same tolerance), read sets identical.  A
    missing value (NaN) fails."""
    tol = TARGETS["B1_vs_REF_max_abs"]
    d = out.get("max_abs_diff", {})

    def le(v):
        return bool(np.isfinite(v) and v <= tol)
    return {"rows_4reads_all_matched": bool(out["rows_matched"] == out["rows_REF"] == out["rows_K3"] > 0),
            "B1_4reads_vs_REF": le(d.get("B1", np.nan)), "B0_4reads_vs_REF": le(d.get("B0", np.nan)),
            "HIST_D7_4reads_vs_REF": le(d.get("HIST_D7", np.nan)), "y_exact": bool(d.get("y", np.nan) == 0),
            "read_sets_4reads_identical": bool(out["read_sets_equal_share"] == 1.0),
            "rows_5reads_all_matched": bool(out["rows_matched_r5"] == out["rows_C4_r5"] == out["rows_R3B0_k4"] > 0),
            "B0_5reads_vs_R3B0": le(out["max_abs_diff_B0_r5_vs_R3B0"]),
            "read_sets_5reads_identical": bool(out["read_sets_equal_share_r5"] == 1.0)}


def repro_checks(R: pd.DataFrame, design: str, *, scores: bool) -> dict:
    """B1 / B0 / HIST_D7 / y at 4 reads against round 4's REF parquet (every row), the 4-read read sets, the 5-read
    DOPTQ B0 against round 3's R3-B0 PWL_R at k = 4; with ``scores`` also P3's 5-read targets (C3 min(4, k_max) and
    C4 B1 seed means).  ``identity_ok`` (and ``ok``) require every identity of ``repro_gate``."""
    G = split_rows(R)
    K3 = lad_frame(G, "C3pop", 4)
    ref = CM.load_rows(design)
    cols = ["y", "B0", "B1", "HIST_D7", "read_set"]

    def keyed(t):
        return t.assign(**{c: t[c].astype(str) for c in ("cell_id", "A", "B")})
    m = keyed(K3).merge(keyed(ref)[KEY + cols].rename(columns={c: c + "_ref" for c in cols}), on=KEY)
    out = {"label": "TECHNICAL SMOKE (reproduction run; brief 6.0)", "design": design, "rows_K3": int(len(K3)),
           "rows_REF": int(len(ref)), "rows_matched": int(len(m)),
           "max_abs_diff": {c: float((m["p_" + c if c != "y" else "y"] - m[c + "_ref"]).abs().max())
                            for c in ("y", "B0", "B1", "HIST_D7")},
           "read_sets_equal_share": float((m.read_set.astype(str) == m.read_set_ref.astype(str)).mean())}
    r3 = pd.read_parquet(paths.R4PRED / f"R3-B0_{design}.parquet")
    r3 = r3[(r3.k == 4) & (r3.k_max >= 4)]
    out["R3B0_policies"] = sorted(map(str, r3.policy.unique())) if "policy" in r3 else []
    D5 = lad_frame(G, "C4", 5)
    mm = keyed(D5).merge(keyed(r3)[KEY + ["PWL_R", "read_set"]].rename(columns={"read_set": "read_set_r3"}), on=KEY)
    out.update({"rows_C4_r5": int(len(D5)), "rows_R3B0_k4": int(len(r3)), "rows_matched_r5": int(len(mm)),
                "max_abs_diff_B0_r5_vs_R3B0": float((mm.p_B0 - mm.PWL_R).abs().max()) if len(mm) else float("nan"),
                "read_sets_equal_share_r5": float((mm.read_set.astype(str) == mm.read_set_r3.astype(str)).mean())
                if len(mm) else float("nan")})
    out["identity_checks"] = repro_gate(out)
    ok = all(out["identity_checks"].values())
    out["identity_ok"] = bool(ok)
    if scores:
        c3 = float(r2_seeds(lad_frame(G, "C3pop", 5), "B1").mean())
        c4 = float(r2_seeds(D5, "B1").mean())
        out["targets"] = {"C3pop_r5_B1": {"got": c3, "want": TARGETS["C3pop_r5_B1"][design], "row": row_name("C3pop|r5")},
                          "C4_r5_B1": {"got": c4, "want": TARGETS["C4_r5_B1"][design], "row": row_name("C4|r5")}}
        out["targets_ok"] = bool(all(abs(v["got"] - v["want"]) <= TARGETS["tol"] for v in out["targets"].values()))
        out["ok"] = bool(ok and out["targets_ok"])
    return out


# ------------------------------------------------------------------------------------------------------------
# the frozen item specification (results/ED/ITEM_SPEC.md)
# ------------------------------------------------------------------------------------------------------------
def item_spec_text() -> str:
    def bl(xs):
        return ", ".join(xs)
    lines = [
        "# Item 1.6 specification: E-D read-budget ladder (M18), policy envelope (M17), recipe-2 row",
        "",
        "Frozen at the Stage-0 seal, before the item's first post-seal run (brief 1.2: Stage-0 seal -> item code seal -> "
        "reproduction run labelled TECHNICAL SMOKE -> arms; the pre-seal --smoke runs are PRE-SEAL ENGINEERING CHECKS, "
        "PROTOCOL 4 and 10).  Every in-corpus number is EXPLORATORY (seen); nothing here is a test.  Code: "
        "`r5/ed.py`, `scripts/s1_ed.py`, `scripts/s1_ed_poison.py`, `tests/test_ed.py`.  Sources: brief 6.6, 6.0, "
        "6.10, 8, 9; P3 and PC P3; IR M17 / M18; I2 recipe-2; E5 5.1.",
        "",
        "## Rows and frames (identical queries at every rung and under every policy)",
        "",
        "- Read counts in reads r; code budget k = r - 1; DOPT_Q read set = union of `cell['seqs']['DOPT_Q'][:k]` (brief 3.3).",
        f"- C3 population row \"C3 min(k, k_max), k = reads - 1\" (E-A's 1,802 queries per seed) at {POP_READS} reads; a "
        "cell that cannot take r reads keeps min(r, k_max + 1) reads (and that budget's beta).  The 4-read rung is E-A "
        "itself; every other rung is a different read budget, never an E-A value.  In any E-D table that shows C3 rungs, "
        "the C3 4-read cell reads \"(E-A rows)\" and no 4-read C3 level is printed (the writer blanks it and sets "
        "`level_note`); C3 levels at 5 or more reads appear only in such tables, labelled \"different read budget\", "
        "and as paired contrasts on E-A's identical queries.  Fixed-cohort C4-C6 levels may be printed at every rung, "
        "including 4 reads (C4 0.4290 is not E-A).",
        "- Fixed cohorts: " + "; ".join(f"{c} (k_max >= {c[1]}) at {COHORT_READS[c]} reads" for c in ("C4", "C5", "C6")) + ".",
        "- Frozen query rule for C4-C6 (E5 5.1): every strict-adjacent query of the cohort's cells at every budget; a "
        "pair with both ends read at the cohort's largest budget would be excluded at every budget.  Legal reads never "
        "close a strict-adjacent pair, so the excluded set is empty; the code asserts it (checks "
        "`cohort_largest_budget_closed_queries` = 0).",
        "- Legality: `gen20k.design.legal_metals` for every added read (measured, unread, not strict-adjacent to a read "
        "metal, maximum independent set still reachable).  LOOSE (no look-ahead) is a sensitivity.",
        "",
        "## Arms",
        "",
        f"- B1: beta refit per outer fold and per budget with `r4.refs.fit_reference(bench, rec, Y, 'B1', k = reads - 1)` "
        f"for k in {BETA_KS} (k = 3 uses the stored round-4 fit `results/REF/fitted_<design>.json`; its refit is an "
        "identity check); beta, lambda, rows and coverage logged per fold (`fits_<design>.csv`); one BLAS thread.",
        "- B0: the beta = 0 row.  HIST_D7: the fixed -0.072 (DIAGNOSTIC; not fold-independent).  B1_k3beta: the k = 3 "
        "beta applied at every budget (SENSITIVITY).",
        "",
        "## M17 policy envelope (5 reads; P3's rules and tie-breaks)",
        "",
        f"- Family (strict legality): {bl(ENV_LABEL_FREE)}.  ORACLE_B1 [O] (the legal 5th metal minimising the cell's B1 "
        "SSE; bound only).  LOOSE sensitivity: MAXMIN_R_LOOSE, RANDOM_LOOSE.",
        "- RANDOM: exact expectation (uniform over the legal set, independent across cells) plus a Monte Carlo of "
        f"{MC_REPS} joint draws, RNG `default_rng(20261023 + crc32('ED|RANDOM_MC|<design>|<variant>'))` (`r5.paths.seed_for`).",
        "- Reported: levels for C3 min(4, k_max) at 5 reads and C4; per-block dSSE (light La/Ce-Pr/Nd, mid Sm/Eu-Dy/Ho, "
        "heavy Ho/Er-Yb/Lu) in pooled-R2 units vs DOPTQ and vs 4 reads with cluster-bootstrap 95 % intervals; per-pair "
        "MAE and skill against FLAT (prediction 0); splits by k_max and by cluster; with and without C5-BTBP; envelope "
        "width over the family and the per-seed span; contrast-specific implied MDE = 2.8 x cluster-bootstrap sd.",
        f"- Pre-declared Pareto question (vs DOPTQ, C3 min(4, k_max)): does any member keep every block above "
        f"{PARETO_FLOOR} and improve one block with a cluster lower bound > 0?  Seen before the code: envelope 0.0091 "
        "(BPG) / 0.0096 (BP), no member Pareto-dominates [P3 5, 9].",
        "",
        "## Density and saturation (M18)",
        "",
        "- Series-length strata (14 / 13 / 8-12 / <= 7 metals); read-density index: reads per measured metal, longest "
        "hidden run, share of rows outside the read hull, share of unread metals outside the hull; MSE and relative SSE "
        "vs B1 at 4 reads on the same rows next to R2; P(R2 >= 0.50) as a cluster-bootstrap fraction (descriptive, never "
        "a posterior) with LOCO and leave-one-publication-out ranges, with and without C5-BTBP.",
        f"- Saturation strata: k_max j at j + 1 reads for j = 3, 4, 5, 6 (the 6- and 7-read strata P3 did not compute).  "
        f"Density invariance at saturation is a CONSISTENCY CHECK, not a kill: the rule fires if a stratum with >= "
        f"{SAT_MIN_CLUSTERS} clusters is below {SAT_FLOOR} or the spread is >= {SAT_SPREAD}; the k_max-4 value "
        "(0.444) was known before the rule was written [P3 7c; PC P3].",
        "",
        "## recipe-2 row (2-5 reads, C3 min rule)",
        "",
        f"- Policies: {bl(REC_POLICIES)}.  LADDER = the nested static order {' -> '.join(STATIC_ORDER)} (a static metal is "
        "skipped when it is not legal; the farthest-point HULL pick replaces it).  LADDER_GUARD = HULL for cation "
        f"exchangers ({bl(CX_CLASSES)}; `r5.supply.classify_extractant` on chem_family / extractant_name, blind to "
        "values), LADDER otherwise.  HULL = lightest legal metal, then the legal metal farthest in -CN8 radius, ties -> "
        "lower Z.  LOCO_STATIC = per requested size r and held-out cluster_cpg cluster, the static set (no strict-"
        "adjacent pair) minimising the summed B0 SSE over the other clusters' C3 cells (a cell on which the set is not "
        "legal counts with its HULL set), ties -> lexicographically smallest Z; HULL fallback on a held-out cell where "
        "the set is not legal.  RANDOM_SET = exact mean over every legal read set of size min(r, k_max + 1).",
        "- Primary contrast Ladder - HULL (B1; B0 beside it); also Ladder - DOPTQ, HULL - RANDOM_SET (closure line: "
        "'spread first' is definitional), LOCO_STATIC - HULL, LADDER_GUARD - LADDER, DOPTQ - HULL; by series length.",
        "",
        "## Report rules and statistics",
        "",
        "- Pooled strict-adjacent R2 within each seed, then the seed mean; MSE, MAE; relative SSE.  Every item reports "
        "C5-BTBP in / out (and the single cell c950dc2982e96b83 out for every row: ladder, M17 envelope and recipe-2), "
        "BPG and BP, per-seed values, the "
        "cluster_cpg unit table (SF_POOL exact over 2^13, SIGN, SF_LOG, IM_REL: DIAGNOSTIC), leave-one-publication-out "
        "(r5.common.lopo) and the top-3 unit share.  Duplicate sensitivities: V2-1/2/3 cells dropped; the second cell of "
        "every Stage-0 flagged bit-identity link dropped.",
        f"- Cluster bootstrap: one shared draw matrix `gen20k.inference._draw_counts(13, {REPS}, {BOOT_SEED})` over the 13 "
        "C3 cluster_cpg clusters (P3's); percentile intervals are descriptive (anti-conservative at Kish 2.24).",
        "- Labels: EXPLORATORY (seen) on every in-corpus number; ORACLE [O]; DIAGNOSTIC (HIST_D7, unit tests); "
        "SENSITIVITY (BP, C5-BTBP out, k = 3 beta, LOOSE, duplicates); CONSISTENCY CHECK (saturation rule); TECHNICAL "
        "SMOKE (reproduction run).  The 5- to 7-read numbers carry 'different read budget'.  No in-corpus p-value is "
        "called significant.",
        "",
        "## Reproduction targets (brief 6.0; a mismatch stops the item)",
        "",
        f"- B1 at 4 reads from the fold betas equals `round4_20261022/predictions/REF_<design>.parquet` within "
        f"{TARGETS['B1_vs_REF_max_abs']} on every row (18,020 over both designs); y exact; read sets identical.",
        f"- P3 5-read DOPTQ-nested B1: C3 min(4, k_max) {TARGETS['C3pop_r5_B1']['BPG']} (BPG) / "
        f"{TARGETS['C3pop_r5_B1']['BP']} (BP); C4 {TARGETS['C4_r5_B1']['BPG']} / {TARGETS['C4_r5_B1']['BP']} "
        f"(tolerance {TARGETS['tol']}); 5-read B0 equals round 3's R3-B0 PWL_R at k = 4.",
        "",
        "## Leakage (brief 8)",
        "",
        "- Isolation (`g21.evaluate.assert_test_isolation`) on every fold; support checks: no scored pair with both ends "
        "read, nested and legal ladders, identical row keys across rungs and policies, the DOPT_Q 5th read legal and "
        "equal to the cache; prediction intervention on every read set (unread metals scrambled; predictions "
        "bit-identical) with a block-Y peek positive control that must be caught; the enumerated static sets likewise.",
        f"- Fit poisoning of the k-specific beta refits (k in {BETA_KS}) on all 50 fold-fits: `scripts/s1_ed_poison.py` "
        "(r4.leakage.poisoned_bench; writes only `results/leakage/poison_ED_beta.json`); run after the seal.  The "
        "item's summary and ITEM_TABLE.md are written only when that file carries the current seal, every k of "
        "BETA_KS, 50 fold-fits, all_identical and matches_main_run (s0-mechanics-13); the first run writes rows, "
        "candidates, fits and checks and exits with status 3, and `s1_ed.py --from-rows` (seal and code digests "
        "checked) writes the tables after the poisoning run.",
        "- Fit poisoning of the recipe-2 LOCO static read-set selection (`r5.ed.poison_loco`): each cluster's labels "
        "replaced by seeded noise, the selection for that cluster bit-identical; leaky positive control (selection with "
        "the cluster's own rows) caught for at least one cluster; written to `results/leakage/poison_ED_loco.json`; a "
        "failure stops the item before any row is computed.",
        "- The block-Y peek positive control goes through the real level function and comparator: levels from every "
        "measured metal must change when the unread metals are scrambled (s0-mechanics-6).",
        "",
        "## RNG seeds",
        "",
        f"- Cluster bootstrap: {BOOT_SEED} (gen20k.inference.SEED).  RANDOM Monte Carlo: seed_for('ED|RANDOM_MC|<design>|"
        "<variant>').  Intervention scrambles: seed_for('ED|intervention|<design>'), seed_for('ED|static_intervention').  "
        "Fit poisoning: seed = split_seed * 10 + fold (round 4's).  Sign-flip tests: exact at G <= 13 "
        "(seed_for('<key>|SF_POOL') otherwise).",
        "",
        "## Command lines (run from D:\\ml_separator_gh)",
        "",
        "```",
        "set PYTHONIOENCODING=utf-8",
        "rem pre-seal engineering check on real data (no metric printed or written; y and every y-derived column "
        "replaced by seeded noise before any table runs; allowed before the seal)",
        ".venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --smoke --design BPG",
        "rem after the seal: reproduction run (TECHNICAL SMOKE), the arms (rows; exit 3), the fit-poisoning check, "
        "then the tables from the rows",
        ".venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --repro --design both",
        ".venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --design both",
        ".venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed_poison.py",
        ".venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --from-rows --design both",
        ".venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ed.py -q "
        "-p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest",
        "```",
        "",
    ]
    return "\n".join(lines)
