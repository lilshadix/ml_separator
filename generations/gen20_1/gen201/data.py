"""The frozen gen13 exact cohort as gen20.1 sees it: cohort assertions, cell metadata, the C3 cells,
strict-adjacent queries, the §2 scored populations (label-free row sets), the DMDPhPDA rule, and the
feature matrices.

Label discipline (IMPLEMENTATION_PLAN.md §2, §9 S3): nothing here reads a log D value except
(i) NaN masks of ``bench.Y`` (which metals were measured) and (ii) the NaN-aware equality assertion
``bench.Y == bench.frame[logD__*]``.  ``nrep__*`` replicate counts are identity information (how many
readings), never values.  The bundle is read only for identity/run columns (``safe_exp_id``,
``metal``), never ``log_D``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import bootstrap
from gen20k import data as GD  # noqa: E402
from gen20k import design as GDS  # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS  # noqa: E402
from gen13sep.features import _assert_allowed  # noqa: E402
from gen13sep.metals import ATOMIC_NUMBER, SHANNON_RADIUS_CN8  # noqa: E402
from gen13sep.metals import LANTHANIDES as _LN  # noqa: E402

LANTHANIDES: tuple[str, ...] = tuple(_LN)
N_LN = len(LANTHANIDES)
IDX = {m: i for i, m in enumerate(LANTHANIDES)}
Z = np.array([ATOMIC_NUMBER[m] for m in LANTHANIDES], dtype=float)
R = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], dtype=float)
LOGD_COLS = tuple(f"logD__{m}" for m in LANTHANIDES)
NREP_COLS = tuple(f"nrep__{m}" for m in LANTHANIDES)

BUNDLE_SHA256 = "fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd"
COHORT_PARQUET_SHA256 = "3f11ac4ccb7c9f6b1fa768f4406e1f249881cfd76689ba49a9115efc08bb7fd6"
GEN13_FINGERPRINT = "4c3c6628ea0be949"        # drift expected (gen20k AUDIT.md:75); recorded only

DMD_SMILES = "CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1"
DMD_COND_COLUMN = "cond__diluent__ch3cl"
SC009 = "sc009"

COND_BLOCKS: tuple[str, ...] = ("COND", "MASSACT")
DENY_EXACT: tuple[str, ...] = ("n_metals", "n_rows", "n_replicated_metals", "replicate_sd_median", "metals",
                               "experiment_series_ids", "safe_exp_ids")
DENY_PREFIX: tuple[str, ...] = ("recipe__",)

LOAD_RECORD: dict = {}


# --------------------------------------------------------------------------------------
# loading and the §1.1 cohort assertions
# --------------------------------------------------------------------------------------
def load(check: bool = True):
    """``gen20k.data.load_bench(check=True)`` plus the gen20.1 assertions (``assert_cohort``)."""
    bench = GD.load_bench(check=True)
    if check:
        LOAD_RECORD.clear()
        LOAD_RECORD.update(assert_cohort(bench))
    return bench


def observed(bench) -> np.ndarray:
    """cells x 14 boolean: metal measured in the cell (a NaN mask, no value)."""
    return ~np.isnan(np.asarray(bench.Y, dtype=float))


def assert_cohort(bench) -> dict:
    """IMPLEMENTATION_PLAN.md §1.1 "Cohort assertions"; any failure raises (STOP)."""
    from gen13sep import paths
    from gen13sep.cohort import Gen13Cohort
    from .run import sha256_file
    rec = GD.assert_matches_manifest(bench.frame)          # column-wise, atol 1e-9
    f = bench.frame
    Yf = f[list(LOGD_COLS)].to_numpy(dtype=float)
    Y = np.asarray(bench.Y, dtype=float)
    if Y.shape != Yf.shape or not np.array_equal(Y, Yf, equal_nan=True):
        raise AssertionError("bench.Y differs from bench.frame[logD__La..logD__Lu] (NaN-aware)")
    obs = observed(bench)
    if not np.array_equal(obs.sum(1), f["n_metals"].to_numpy()):
        raise AssertionError("frame.n_metals differs from the measured-metal mask")
    if not np.array_equal(np.asarray(bench.n_obs, dtype=float), obs.sum(1).astype(float)):
        raise AssertionError("bench.n_obs differs from the measured-metal mask")
    if not (np.asarray(bench.groups).astype(str) == f["chemotype"].astype(str).to_numpy()).all():
        raise AssertionError("bench.groups differs from frame.chemotype")
    bundle_sha = sha256_file(paths.BUNDLE_PARQUET)
    if bundle_sha != BUNDLE_SHA256 or paths.BUNDLE_SHA256 != BUNDLE_SHA256:
        raise AssertionError(f"bundle sha256 {bundle_sha} != {BUNDLE_SHA256}")
    parquet_sha = sha256_file(GD.MANIFEST_PARQUET)
    if parquet_sha != COHORT_PARQUET_SHA256:
        raise AssertionError(f"cohort parquet sha256 {parquet_sha} != {COHORT_PARQUET_SHA256}")
    try:
        fp = Gen13Cohort(frame=f, condition_columns=tuple(c for c in f.columns if c.startswith("cond__")),
                         key_mode="exact").fingerprint()
    except Exception as exc:  # pragma: no cover
        fp = f"unavailable: {exc!r}"
    return {"assert_matches_manifest": rec, "Y_equals_frame_logD": True,
            "bundle_sha256": bundle_sha, "cohort_parquet_sha256": parquet_sha,
            "bench_pkl": str(bootstrap.BENCH_PKL.relative_to(bootstrap.ROOT)).replace("\\", "/"),
            "bench_pkl_sha256": sha256_file(bootstrap.BENCH_PKL),
            "bench_pkl_bytes": int(bootstrap.BENCH_PKL.stat().st_size),
            "fingerprint_obtained": fp, "fingerprint_gen13_recorded": GEN13_FINGERPRINT,
            "fingerprint_drift": fp != GEN13_FINGERPRINT,
            "n_cells": int(len(f)), "n_entries": int(obs.sum())}


# --------------------------------------------------------------------------------------
# cells, queries, C3
# --------------------------------------------------------------------------------------
def q_adj(bench, ci: int) -> tuple[tuple[int, int], ...]:
    """Strict-adjacent pairs (|dZ| = 1) of cell ``ci``, lighter first (gen20k query 'adjacent')."""
    return GDS.make_design(GD.cell_view(bench, ci), "adjacent", "metal").query


def q_gen13(bench, ci: int) -> tuple[tuple[int, int], ...]:
    """gen13 adjacency (strict + Nd-Sm) pairs of cell ``ci``, lighter first."""
    return GDS.make_design(GD.cell_view(bench, ci), "adjacent_gen13", "metal").query


def dmd_mask(bench) -> np.ndarray:
    """The DMDPhPDA corrupt-copy rule: SMILES AND cond__diluent__ch3cl == 1 (never a name filter)."""
    f = bench.frame
    if DMD_COND_COLUMN not in f.columns:
        raise KeyError(f"{DMD_COND_COLUMN} missing from the cohort frame")
    return (f["extractant"].astype(str).to_numpy() == DMD_SMILES) & (f[DMD_COND_COLUMN].to_numpy(dtype=float) == 1)


def dmd_cells(bench) -> np.ndarray:
    return np.flatnonzero(dmd_mask(bench))


_META_CACHE: dict = {}


def cell_meta(bench) -> pd.DataFrame:
    """One row per cell (index = ci): identity, measured-metal count, query sizes and k_max under the
    gen20 metal model for strict and gen13 adjacency, C3/C1/ANY/NDSM flags, sc009 and DMD flags,
    replicated-entry count (nrep >= 2 among measured metals)."""
    if _META_CACHE.get("bench") is bench:
        return _META_CACHE["meta"]
    f = bench.frame
    obs = observed(bench)
    nrep = f[list(NREP_COLS)].to_numpy(dtype=float)
    rows = []
    for ci in range(len(f)):
        cv = GD.cell_view(bench, ci)
        ds = GDS.make_design(cv, "adjacent", "metal")
        dg = GDS.make_design(cv, "adjacent_gen13", "metal")
        rows.append({"n_qadj": len(ds.query), "k_max": int(ds.k_max),
                     "n_qgen13": len(dg.query), "k_max_gen13": int(dg.k_max),
                     "metals_in_qadj": len({m for p in ds.query for m in p})})
    m = pd.DataFrame(rows)
    out = pd.DataFrame({"ci": np.arange(len(f)), "cell_id": f["cell_id"].astype(str).to_numpy(),
                        "extractant": f["extractant"].astype(str).to_numpy(),
                        "chemotype": f["chemotype"].astype(str).to_numpy(),
                        "publication_id": f["publication_id"].astype(str).to_numpy(),
                        "n_obs": obs.sum(1).astype(int)})
    out = pd.concat([out, m], axis=1)
    out["is_any"] = out.n_qadj > 0
    out["is_c1"] = out.is_any & (out.k_max >= 1)
    out["is_c3"] = out.is_any & (out.k_max >= 3)
    out["is_ndsm"] = (out.n_qgen13 > 0) & (out.k_max_gen13 >= 3)
    out["is_sc009"] = out.chemotype == SC009
    out["is_dmd"] = dmd_mask(bench)
    out["n_rep_entries"] = ((nrep >= 2) & obs).sum(1).astype(int)
    _META_CACHE.clear()
    _META_CACHE.update({"bench": bench, "meta": out})       # keeps a reference: no id() reuse
    return out


def c3_cells(bench) -> np.ndarray:
    """gen20's C3 (gen20k/report.py:69): non-empty strict-adjacent query and metal-model k_max >= 3."""
    m = cell_meta(bench)
    return m.index[m.is_c3].to_numpy()


# --------------------------------------------------------------------------------------
# row sets (populations)
# --------------------------------------------------------------------------------------
@dataclass
class RowSet:
    """A label-free scored population: which (cell, metal) entries or (cell, a, b) pairs are scored."""
    name: str
    target: str                                  # "logd" or "logsf"
    rows: pd.DataFrame                           # see logd_rows / logsf_rows
    definition: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def cells(self) -> np.ndarray:
        return np.unique(self.rows["ci"].to_numpy())

    @property
    def n_rows(self) -> int:
        return int(len(self.rows))

    def minus_cells(self, cells, name: str | None = None) -> "RowSet":
        keep = ~self.rows["ci"].isin(np.asarray(cells))
        return RowSet(name or f"{self.name}@NODMD", self.target, self.rows[keep].reset_index(drop=True),
                      self.definition + " minus the DMDPhPDA cells", dict(self.meta))


def _id_frame(bench, cells) -> pd.DataFrame:
    f = bench.frame
    cells = np.asarray(cells, dtype=int)
    return pd.DataFrame({"ci": cells, "cell_id": f["cell_id"].astype(str).to_numpy()[cells],
                         "extractant": f["extractant"].astype(str).to_numpy()[cells],
                         "chemotype": f["chemotype"].astype(str).to_numpy()[cells],
                         "publication_id": f["publication_id"].astype(str).to_numpy()[cells],
                         "n_metals": f["n_metals"].to_numpy()[cells].astype(int)})


def logd_rows(bench, cells, *, metal_filter=None) -> pd.DataFrame:
    """Every measured (cell, metal) entry of ``cells`` (``metal_filter(ci) -> set[int]`` restricts)."""
    obs = observed(bench)
    ident = _id_frame(bench, cells).set_index("ci")
    out = []
    for ci in np.asarray(cells, dtype=int):
        ms = np.flatnonzero(obs[ci])
        if metal_filter is not None:
            keep = metal_filter(int(ci))
            ms = [m for m in ms if int(m) in keep]
        for m in ms:
            out.append((int(ci), int(m)))
    t = pd.DataFrame(out, columns=["ci", "im"])
    t = t.join(ident, on="ci")
    t["metal"] = [LANTHANIDES[i] for i in t["im"]]
    return t[["ci", "cell_id", "extractant", "chemotype", "publication_id", "n_metals", "metal", "im"]]


def logsf_rows(bench, cells, *, query: str = "adjacent") -> pd.DataFrame:
    """Within-cell pairs of ``cells`` (query 'adjacent' = strict |dZ| = 1, 'adjacent_gen13' adds Nd-Sm),
    lighter first, with ``strict`` and ``gen13adj`` flags."""
    ident = _id_frame(bench, cells).set_index("ci")
    out = []
    for ci in np.asarray(cells, dtype=int):
        qs = q_adj(bench, ci) if query == "adjacent" else q_gen13(bench, ci)
        for ia, ib in qs:
            out.append((int(ci), int(ia), int(ib)))
    t = pd.DataFrame(out, columns=["ci", "ia", "ib"])
    t = t.join(ident, on="ci")
    t["a"] = [LANTHANIDES[i] for i in t["ia"]]
    t["b"] = [LANTHANIDES[i] for i in t["ib"]]
    t["dZ"] = (Z[t["ib"].to_numpy()] - Z[t["ia"].to_numpy()]).astype(int)
    t["pair"] = t["a"] + "/" + t["b"]
    t["strict"] = [bool(GD.is_strict_adjacent(a, b)) for a, b in zip(t["ia"], t["ib"])]
    t["gen13adj"] = [bool(GD.is_gen13_adjacent(a, b)) for a, b in zip(t["ia"], t["ib"])]
    if (t["ia"] >= t["ib"]).any():
        raise AssertionError("pairs must be lighter first (ia < ib)")
    return t[["ci", "cell_id", "extractant", "chemotype", "publication_id", "n_metals",
              "a", "b", "ia", "ib", "dZ", "pair", "strict", "gen13adj"]]


#: the scored (test-side) populations of §2; ``<name>@NODMD`` adds COH-NODMD.  Training-side populations
#: (COH-TRAIN, COH-INNER, COH-REP) are per fold: ``train_logd_rows`` and ``folds.inner_folds``.
SCORED_POPULATIONS: tuple[str, ...] = (
    "COH-D-C3", "COH-SF-C3", "COH-D-C3M", "COH-D-ALL", "COH-SF-ANY", "COH-SF-C1", "COH-SF-NDSM",
    "COH-SC009-D", "COH-SC009-SF", "COH-NOSC009-D", "COH-NOSC009-SF")
PRIMARY_POPULATIONS = SCORED_POPULATIONS          # backward-compatible alias


def populations(bench, *, nodmd: bool = True) -> dict[str, RowSet]:
    """Every label-free §2 population as a RowSet (COH-REPCAP is label-derived and produced at S5).

    ``<name>@NODMD`` is the same population minus the 5 DMDPhPDA cells (COH-NODMD)."""
    m = cell_meta(bench)
    c3 = m.index[m.is_c3].to_numpy()
    anyc = m.index[m.is_any].to_numpy()
    c1 = m.index[m.is_c1].to_numpy()
    ndsm = m.index[m.is_ndsm].to_numpy()
    s9 = m.index[m.is_c3 & m.is_sc009].to_numpy()
    n9 = m.index[m.is_c3 & ~m.is_sc009].to_numpy()
    allc = m.index.to_numpy()

    def c3m_metals(ci):
        return {x for p in q_adj(bench, ci) for x in p}

    pops = {
        "COH-D-C3": RowSet("COH-D-C3", "logd", logd_rows(bench, c3), "every measured entry of the C3 cells"),
        "COH-SF-C3": RowSet("COH-SF-C3", "logsf", logsf_rows(bench, c3), "Q_adj pairs of the C3 cells"),
        "COH-D-C3M": RowSet("COH-D-C3M", "logd", logd_rows(bench, c3, metal_filter=c3m_metals),
                            "C3 entries whose metal is in >= 1 Q_adj pair of its cell"),
        "COH-D-ALL": RowSet("COH-D-ALL", "logd", logd_rows(bench, allc), "every entry of the 521 cells"),
        "COH-SF-ANY": RowSet("COH-SF-ANY", "logsf", logsf_rows(bench, anyc), "strict pairs of cells with >= 1 such pair"),
        "COH-SF-C1": RowSet("COH-SF-C1", "logsf", logsf_rows(bench, c1), "strict pairs of cells with k_max >= 1"),
        "COH-SF-NDSM": RowSet("COH-SF-NDSM", "logsf", logsf_rows(bench, ndsm, query="adjacent_gen13"),
                              "gen13-adjacency pairs (strict + Nd-Sm) of cells with a non-empty 'adjacent_gen13' "
                              "query and metal-model k_max >= 3 under it"),
        "COH-SC009-D": RowSet("COH-SC009-D", "logd", logd_rows(bench, s9), "COH-D-C3 restricted to sc009"),
        "COH-SC009-SF": RowSet("COH-SC009-SF", "logsf", logsf_rows(bench, s9), "COH-SF-C3 restricted to sc009"),
        "COH-NOSC009-D": RowSet("COH-NOSC009-D", "logd", logd_rows(bench, n9), "COH-D-C3 minus sc009"),
        "COH-NOSC009-SF": RowSet("COH-NOSC009-SF", "logsf", logsf_rows(bench, n9), "COH-SF-C3 minus sc009"),
    }
    if nodmd:
        dm = dmd_cells(bench)
        for k in list(pops):
            pops[f"{k}@NODMD"] = pops[k].minus_cells(dm, f"{k}@NODMD")
    return pops


def nrep_matrix(bench) -> np.ndarray:
    """cells x 14 replicate counts (``nrep__*``: how many readings, never their values; NaN = unmeasured)."""
    return bench.frame[list(NREP_COLS)].to_numpy(dtype=float)


def train_logd_rows(bench, train_index) -> pd.DataFrame:
    """COH-TRAIN of one fold: every measured (cell, metal) entry of the (masked) training cells, with the
    replicate count ``nrep`` (COH-REP = rows with nrep >= 2) and the DMDPhPDA flag (COH-NODMD drops them)."""
    t = logd_rows(bench, np.sort(np.asarray(train_index, dtype=int)))
    nrep = nrep_matrix(bench)
    t["nrep"] = nrep[t["ci"].to_numpy(), t["im"].to_numpy()].astype(int)
    t["is_dmd"] = dmd_mask(bench)[t["ci"].to_numpy()]
    return t


def kish(groups, weights=None) -> float:
    """Kish effective number of clusters (sum n_g)^2 / sum n_g^2 over cluster row counts."""
    g = pd.Series(np.ones(len(groups)) if weights is None else np.asarray(weights, dtype=float)).groupby(
        np.asarray(groups)).sum()
    return float(g.sum() ** 2 / (g ** 2).sum())


def card(rs: RowSet) -> dict:
    """Label-free card of a row set: rows, cells, extractants, chemotypes, publications, Kish, share."""
    t = rs.rows
    vc = t["chemotype"].value_counts()
    return {"n_rows": int(len(t)), "n_cells": int(t["ci"].nunique()), "n_extractants": int(t["extractant"].nunique()),
            "n_chemotypes": int(t["chemotype"].nunique()), "n_publications": int(t["publication_id"].nunique()),
            "kish_chemotypes": kish(t["chemotype"].to_numpy()) if len(t) else float("nan"),
            "largest_chemotype": str(vc.index[0]) if len(vc) else None,
            "largest_chemotype_share": float(vc.iloc[0] / vc.sum()) if len(vc) else float("nan")}


# --------------------------------------------------------------------------------------
# features (target-free)
# --------------------------------------------------------------------------------------
def assert_feature_columns(columns) -> None:
    """gen13's allow-list (features._assert_allowed) plus the gen20.1 deny list (§2)."""
    cols = list(columns)
    _assert_allowed(cols)
    bad = [c for c in cols if c in DENY_EXACT or c.startswith(DENY_PREFIX) or c.startswith("logD__")]
    if bad:
        raise RuntimeError(f"denied feature columns: {bad[:5]}")


def X_lean(bench) -> np.ndarray:
    """LEAN209 = COND64 + MASSACT8 + PHYSCHEM10 + DONORS13 + COORD114 (float64)."""
    cols = bench.columns(LEAN_BLOCKS)
    assert_feature_columns(cols)
    X = np.asarray(bench.matrix(LEAN_BLOCKS), dtype=np.float64)
    if X.shape != (len(bench.frame), 209):
        raise AssertionError(f"LEAN209 has shape {X.shape}")
    return X


def lean_columns(bench) -> list[str]:
    return list(bench.columns(LEAN_BLOCKS))


def X_cond(bench) -> np.ndarray:
    """COND64 + MASSACT8 (BD_COND's cell block; METAL_BASIC2 is appended per row by the learner)."""
    cols = bench.columns(COND_BLOCKS)
    assert_feature_columns(cols)
    X = np.asarray(bench.matrix(COND_BLOCKS), dtype=np.float64)
    if X.shape != (len(bench.frame), 72):
        raise AssertionError(f"COND64 + MASSACT8 has shape {X.shape}")
    return X


def metal_basic2() -> np.ndarray:
    """14 x 2 target-free metal descriptors (Z, CN8 Shannon radius) = gen13 metal_feature_matrix('basic')."""
    return np.c_[Z, R].astype(np.float64)


def ecfp(bench) -> np.ndarray:
    """Boolean ECFP fingerprints (``bench.frames['ECFP'] > 0.5``), used only by nearest-neighbour search."""
    return bench.frames["ECFP"].to_numpy(dtype=float) > 0.5


# --------------------------------------------------------------------------------------
# replicate readings (identity/run columns only; never log D)
# --------------------------------------------------------------------------------------
PROVENANCE_RUN_COLUMNS: tuple[str, ...] = ("experiment_id", "experiment_series_id", "replicate_id")


def reading_index(bench) -> pd.DataFrame:
    """One row per raw reading of the 521 cells: ci, safe_exp_id, metal (from the bundle), and the gen6
    provenance run identifiers.  Reads only identity columns (never ``log_D``); asserts that the number
    of readings per (cell, metal) equals ``nrep__`` and covers exactly the measured metals."""
    from gen13sep import paths
    b = pd.read_parquet(paths.BUNDLE_PARQUET, columns=["safe_exp_id", "metal"])
    prov = pd.read_parquet(paths.GEN6_PROVENANCE_PARQUET,
                           columns=["safe_exp_id", "metal_symbol", *PROVENANCE_RUN_COLUMNS,
                                    "experiment_id_status", "experiment_series_id_status", "replicate_id_status"])
    f = bench.frame
    rows = []
    for ci, ids in enumerate(f["safe_exp_ids"].astype(str)):
        for s in ids.split(";"):
            rows.append((ci, s))
    r = pd.DataFrame(rows, columns=["ci", "safe_exp_id"])
    r = r.merge(b, on="safe_exp_id", how="left", validate="one_to_one")
    r = r.merge(prov, on="safe_exp_id", how="left", validate="one_to_one")
    if r["metal"].isna().any():
        raise AssertionError("a cohort safe_exp_id is missing from the bundle")
    r["im"] = r["metal"].map(IDX)
    if r["im"].isna().any():
        raise AssertionError("non-lanthanide metal among cohort readings")
    r["im"] = r["im"].astype(int)
    nrep = f[list(NREP_COLS)].to_numpy(dtype=float)
    cnt = r.groupby(["ci", "im"]).size()
    obs = observed(bench)
    want = {(int(ci), int(im)): int(nrep[ci, im]) for ci, im in zip(*np.nonzero(obs))}
    got = {(int(k[0]), int(k[1])): int(v) for k, v in cnt.items()}
    if want != got:
        raise AssertionError("readings per (cell, metal) differ from nrep__ / the measured-metal mask")
    r["cell_id"] = f["cell_id"].astype(str).to_numpy()[r["ci"].to_numpy()]
    r["chemotype"] = f["chemotype"].astype(str).to_numpy()[r["ci"].to_numpy()]
    return r


def reppair_candidates(bench, readings: pd.DataFrame, key: str, *, cells=None) -> pd.DataFrame:
    """Strict-adjacent pairs (c, a, b) with both endpoints replicated (nrep >= 2) and, under run key
    ``key``, at least two run values in which a and b each have exactly one reading (a one-to-one
    run pairing; readings are never cross-multiplied).  Identity columns only."""
    f = bench.frame
    nrep = f[list(NREP_COLS)].to_numpy(dtype=float)
    cells = np.arange(len(f)) if cells is None else np.asarray(cells, dtype=int)
    by = {k: g for k, g in readings.groupby("ci")}
    out = []
    for ci in cells:
        for ia, ib in q_adj(bench, int(ci)):
            if nrep[ci, ia] < 2 or nrep[ci, ib] < 2:
                continue
            g = by[int(ci)]
            ka = g.loc[g.im == ia, key].astype(str).value_counts()
            kb = g.loc[g.im == ib, key].astype(str).value_counts()
            runs = sorted(set(ka.index[ka == 1]) & set(kb.index[kb == 1]))
            out.append({"ci": int(ci), "cell_id": str(f["cell_id"].iat[ci]), "chemotype": str(f["chemotype"].iat[ci]),
                        "ia": int(ia), "ib": int(ib), "nrep_a": int(nrep[ci, ia]), "nrep_b": int(nrep[ci, ib]),
                        "n_paired_runs": len(runs), "run_paired": len(runs) >= 2})
    return pd.DataFrame(out, columns=["ci", "cell_id", "chemotype", "ia", "ib", "nrep_a", "nrep_b",
                                      "n_paired_runs", "run_paired"])
