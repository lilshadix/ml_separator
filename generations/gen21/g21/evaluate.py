"""Long prediction tables and pooled metrics for gen21 methods on the cached folds.

A *method* is ``fn(block, cell, obs, k) -> level vector (14,)`` (NaN allowed only at metals the
cell does not have); the prediction for a strict-adjacent query ``(a, b)`` is ``v[a] - v[b]``
(A lighter, gen13's orientation).  ``obs`` is an ``Observed`` object exposing ONLY what the
measurement contract allows at prediction time: the read metals and their log D values (the masked
row), the gen20 support pairs, and the cell's metal identities.  Hidden metals are never passed
to a method; ``scramble_hidden`` replaces them by random numbers for the intervention test.

Every row carries ``a_read`` / ``b_read`` so the "both ends measured" rule is checked on the table
itself (``assert_no_known_pair``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cache as C
from . import paths
from gen20k import adapters as AD  # noqa: E402
from gen20k import cov as CV  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k.data import LANTHANIDES, N_LN, Z_VEC  # noqa: E402

KS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class Observed:
    read: tuple[int, ...]            # read metals (sorted by Z)
    y_masked: np.ndarray             # (14,) log D of read metals, NaN elsewhere
    support: DS.Support              # gen20 star-pair support (differences to the first read metal)
    metals: tuple[int, ...]          # metals the cell has (identities only)
    query: tuple[tuple[int, int], ...]


def observe(Y_row: np.ndarray, cell: dict, seq, k: int) -> Observed:
    sup = DS.reveal(Y_row, list(seq[:k]))
    order = sorted(sup.read)
    ym = np.full(N_LN, np.nan)
    for m in order:
        ym[m] = Y_row[m]
    return Observed(read=tuple(order), y_masked=ym, support=sup, metals=cell["metals"], query=cell["query"])


def scramble_hidden(Y_row: np.ndarray, read, rng) -> np.ndarray:
    out = Y_row.copy()
    hide = [i for i in range(N_LN) if np.isfinite(Y_row[i]) and i not in set(read)]
    out[hide] = rng.normal(0.0, 5.0, size=len(hide))
    return out


# --------------------------------------------------------------------------------------
# gen20 baselines through the same interface (the fold's own prior / lambda / covariance / s2)
# --------------------------------------------------------------------------------------
def gen20_baselines(bench):
    basis = bench.basis

    def ridge2(block, cell, obs, k):
        return AD.ridge_line(basis, obs.support, block["prior"], block["ridge_lambda"],
                             CV.noise_matrix(obs.support.pairs, block["s2"], "shared"))

    def line(block, cell, obs, k):
        return AD.naive_line(basis, obs.support)

    def g14_blup(block, cell, obs, k):
        return AD.blup(cell["zero_g14"], block["cov"], obs.support, CV.noise_matrix(obs.support.pairs, block["s2"], "shared"))

    def flat_blup(block, cell, obs, k):
        return AD.blup(cell["zero_flat"], block["cov"], obs.support, CV.noise_matrix(obs.support.pairs, block["s2"], "shared"))

    def g14_zero(block, cell, obs, k):
        return cell["zero_g14"]

    def flat_zero(block, cell, obs, k):
        return np.zeros(N_LN)

    return {"RIDGE2": ridge2, "LINE": line, "G14@BLUP20": g14_blup, "FLAT@BLUP20": flat_blup,
            "G14@ZERO": g14_zero, "FLAT": flat_zero}


# --------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------
def block_rows(bench, rec: dict, block: dict, methods: dict, *, policy: str = "DOPT_Q", ks=KS,
               role: str = "outer", Y=None, cells=None, logd_arms=(), logd_sink=None) -> list[dict]:
    """Query rows; with ``logd_sink`` also one row per (arm, unread metal) at k >= 1 with the arm's
    absolute log D: its level vector shifted by the mean (reading - level) over the read metals."""
    Y = bench.Y if Y is None else Y
    block = dict(block, Y=Y, policy=policy)          # sibling-pooled methods read sibling reads through this Y only
    rows = []
    for cell in (block["cells"] if cells is None else cells):
        seq = cell["seqs"][policy] if isinstance(policy, str) else policy(block, cell)
        sibs = siblings(block, cell)
        for k in ks:
            if k > len(seq):
                continue
            obs = observe(Y[cell["ci"]], cell, seq, k)
            DS.assert_no_closure(DS.make_design(_view(cell), "adjacent", "metal"), list(obs.support.pairs))
            preds = {n: np.asarray(fn(block, cell, obs, k), float) for n, fn in methods.items()}
            read = set(obs.read)
            sib_reads = [set(o.read) for _, o in sibling_obs(block, cell, k)] if sibs else []
            if logd_sink is not None and k >= 1:
                rr = list(obs.read)
                yr = obs.y_masked[rr]
                for n in list(logd_arms) + ["MEAN_OF_READS"]:
                    if n == "MEAN_OF_READS":
                        lvl = np.full(N_LN, float(yr.mean()))
                    else:
                        v = preds[n]
                        lvl = v + float((yr - v[rr]).mean())
                    for m in cell["metals"]:
                        if m in read:
                            continue
                        logd_sink.append({"design": rec["design"], "split_seed": rec["split_seed"], "fold": rec["fold"],
                                          "cell_id": cell["cell_id"], "extractant": cell["extractant"], "chemotype": cell["chemotype"],
                                          "k_max": cell["k_max"], "k": k, "policy": policy if isinstance(policy, str) else "CUSTOM",
                                          "metal": LANTHANIDES[m], "arm": n, "y": float(bench.Y[cell["ci"]][m]), "pred": float(lvl[m])})
            for a, b in cell["query"]:
                r = {"design": rec["design"], "split_seed": rec["split_seed"], "fold": rec["fold"], "role": role,
                     "cell_id": cell["cell_id"], "extractant": cell["extractant"], "chemotype": cell["chemotype"],
                     "publication": cell["publication"], "n_metals": len(cell["metals"]), "model": "metal",
                     "k_max": cell["k_max"], "policy": policy if isinstance(policy, str) else "CUSTOM", "draw": 0,
                     "k": k, "A": LANTHANIDES[a], "B": LANTHANIDES[b], "ia": a, "ib": b,
                     "dZ": int(Z_VEC[b] - Z_VEC[a]), "regime": "Q", "a_read": a in read, "b_read": b in read,
                     "lab_known": bool(a in read and b in read), "n_siblings": len(sibs),
                     "sib_pair_read": bool(any(a in r_ and b in r_ for r_ in sib_reads)),
                     "y": float(bench.Y[cell["ci"]][a] - bench.Y[cell["ci"]][b])}
                for n, v in preds.items():
                    r[n] = float(v[a] - v[b])
                rows.append(r)
    return rows


def siblings(block: dict, cell: dict) -> list[dict]:
    """Other held-out cells of the same extractant in the same publication (other conditions)."""
    key = (cell["extractant"], cell["publication"])
    return [c for c in block["cells"] if (c["extractant"], c["publication"]) == key and c["ci"] != cell["ci"]]


def sibling_obs(block: dict, cell: dict, k: int) -> list:
    """Each sibling's own reads at budget min(k, its k_max) under the block's policy, read from
    ``block['Y']`` (never anything a sibling did not read)."""
    pol = block.get("policy", "DOPT_Q")
    out = []
    for sib in siblings(block, cell):
        seq = sib["seqs"][pol] if isinstance(pol, str) else pol(block, sib)
        out.append((sib, observe(block["Y"][sib["ci"]], sib, seq, min(k, len(seq)))))
    return out


def duplicate_sibling_cells(bench, block: dict, tol: float = 1e-6) -> list[str]:
    """EVALUATION-ONLY data-quality flag (reads labels; never passed to a method): held-out cells
    with a sibling in the same series whose observed log D curve equals theirs up to a constant
    offset on every shared metal (an exact copy of one measurement series, e.g. re-reported with a
    shifted reference)."""
    out = []
    for cell in block["cells"]:
        for sib in siblings(block, cell):
            shared = [m for m in cell["metals"] if m in sib["metals"]]
            if len(shared) < 3:
                continue
            d = bench.Y[cell["ci"]][shared] - bench.Y[sib["ci"]][shared]
            if np.max(np.abs(d - d.mean())) < tol:
                out.append(cell["cell_id"])
                break
    return out


def scramble_block(Y: np.ndarray, block: dict, policy, k: int, rng) -> np.ndarray:
    """Every held-out cell of the block: metals not read at budget k replaced by random numbers."""
    Ys = Y.copy()
    for cell in block["cells"]:
        seq = cell["seqs"][policy] if isinstance(policy, str) else policy(block, cell)
        read = DS.reveal(Y[cell["ci"]], list(seq[:min(k, len(seq))])).read
        Ys[cell["ci"]] = scramble_hidden(Y[cell["ci"]], read, rng)
    return Ys


def _view(cell):
    from gen20k.data import CellView
    return CellView(index=cell["ci"], cell_id=cell["cell_id"], extractant=cell["extractant"], chemotype=cell["chemotype"],
                    publication=cell["publication"], metals=cell["metals"])


def outer_table(bench, design: str, seeds, methods: dict, *, policy="DOPT_Q", ks=KS, fit=None, log=None) -> pd.DataFrame:
    """Rows for every held-out cell of every cached fold.  ``fit(bench, rec) -> dict`` (optional)
    fits a method's free parameters on the fold's inner cross-fit blocks only; its result is passed
    to ``methods`` factories as ``block['fitted']``."""
    from gen20k.splits import folds_for
    out = []
    for f in folds_for(bench.frame, design, seeds=list(seeds)):
        rec = C.load(f.design, f.seed, f.fold)
        blk = dict(rec["outer"])
        if fit is not None:
            blk["fitted"] = fit(bench, rec)
        out += block_rows(bench, rec, blk, methods, policy=policy, ks=ks)
        if log:
            log(f"  {design} {f.seed} {f.fold}: {len(out)} rows")
    return pd.DataFrame(out)


def inner_table(bench, rec: dict, methods: dict, *, policy="DOPT_Q", ks=KS) -> pd.DataFrame:
    rows = []
    for blk in rec["inner"]:
        rows += block_rows(bench, rec, blk, methods, policy=policy, ks=ks, role=f"inner{blk['inner_fold']}")
    return pd.DataFrame(rows)


def assert_no_known_pair(table: pd.DataFrame) -> None:
    if bool(table["lab_known"].any()):
        raise AssertionError("a scored query pair has both metals read")


def assert_test_isolation(bench, rec: dict) -> dict:
    """Training/inner cells never share a cell, extractant, chemotype or publication (BP) with the
    fold's held-out cells; BPG also no copy group."""
    f = bench.frame
    tr, te = rec["outer"]["train_index"], rec["outer"]["test_index"]
    rep = {}
    for col in ("cell_id", "extractant", "chemotype", "publication_id"):
        rep[col] = len(set(f[col].astype(str).to_numpy()[tr]) & set(f[col].astype(str).to_numpy()[te]))
    if rec["design"] == "BPG":
        from gen20k.splits import copy_groups_of
        g = copy_groups_of(f)
        rep["copy_group"] = len(set(g[tr]) & set(g[te]))
    for blk in rec["inner"]:
        if not set(map(int, blk["train_index"])) <= set(map(int, tr)) or not set(map(int, blk["val_index"])) <= set(map(int, tr)):
            raise AssertionError("an inner block leaves the outer training fold")
    if any(rep.values()):
        raise AssertionError(f"fold {rec['design']} {rec['split_seed']} {rec['fold']} leaks: {rep}")
    return rep


# --------------------------------------------------------------------------------------
# metrics on the common-query cohort (gen20's wide_common / levels)
# --------------------------------------------------------------------------------------
def cohort_levels(table: pd.DataFrame, arms, *, K: int = 3, ks=None, replicates: int = 2000, policy=None) -> pd.DataFrame:
    from gen20k import report as RP
    pol = policy or table["policy"].iloc[0]
    t = table[(table.policy == pol) & (table.k_max >= K) & (table.k <= K)]
    ks = ks or sorted(t.k.unique())
    parts = []
    key = RP.KEY + ["draw"]
    for k in ks:
        tk = t[t.k == k].set_index(key)[list(arms)]
        tk.columns = [f"{a}|k{k}" for a in arms]
        parts.append(tk)
    wide = pd.concat(parts, axis=1, join="inner").reset_index()
    cols = [c for c in wide.columns if "|k" in c]
    return RP.levels(wide, cols, replicates=replicates), wide


def quick(table: pd.DataFrame, arms, *, K: int = 3, k: int | None = None) -> pd.DataFrame:
    """Per-seed pooled R2 / MAE / RMSE averaged over seeds (no bootstrap) on cohort C<K> at budget k."""
    k = K if k is None else k
    t = table[(table.k_max >= K) & (table.k == k)]
    rows = []
    for a in arms:
        per = []
        for s, g in t.groupby("split_seed"):
            y, p = g["y"].to_numpy(float), g[a].to_numpy(float)
            sst = ((y - y.mean()) ** 2).sum()
            per.append({"R2": 1 - ((y - p) ** 2).sum() / sst, "MAE": np.abs(y - p).mean(), "RMSE": np.sqrt(((y - p) ** 2).mean()),
                        "macro": g.assign(e=np.abs(y - p)).groupby("extractant")["e"].mean().mean()})
        d = pd.DataFrame(per)
        rows.append({"arm": a, "k": k, "cohort": f"C{K}", "n_rows_per_seed": len(t) / t.split_seed.nunique(),
                     "n_cells_per_seed": t.groupby("split_seed").cell_id.nunique().mean(),
                     **{m: float(d[m].mean()) for m in d.columns}, "R2_min_seed": float(d["R2"].min()),
                     "R2_max_seed": float(d["R2"].max())})
    return pd.DataFrame(rows)
