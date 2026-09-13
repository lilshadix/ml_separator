"""Gen14 negative controls: straw barcodes, restricted label permutations, AVE redundancy.

None of this can improve the MAE.  It exists to answer the one question a referee will ask of a
corpus whose 64 condition columns identify a cell's publication with 94 % 1-NN accuracy: is the
0.821 direction accuracy / 0.500 pairwise MAE under BP *chemistry*, or is it a memorised identity?

Three controls, following Chuang & Keiser's adversarial-control protocol
(Science 362:eaat8603, 2018; ACS Chem Biol 13:2819, 2018), Ojala & Garriga's two permutation
tests (ICDM 2009, doi:10.1109/icdm.2009.108) and Wallach & Heifets' AVE bias
(JCIM 58:916, 2018, doi:10.1021/acs.jcim.7b00403).

1. STRAW BARCODES.  Replace the 39 topology columns with 39 N(0,1) columns drawn once per
   *level* of a grouping key and held constant across every cell of that level.  A barcode carries
   zero chemistry and is a perfect identifier of its level, so it can only score by memorising
   that level -- which is exactly what the hold-out design is supposed to forbid.  Three
   granularities price three different leaks:

       extractant   leaks under NO design (A groups by extractant, B/BR/BQ/BP group by
                    chemotype, which is coarser).  This is the pure floor.
       chemotype    leaks under A only.  Prices the A-vs-B gap in a chemistry-free way.
       publication  leaks under A/B/BR/BQ, blocked under BP.  This is the control that makes
                    "BP is the deployment design" a measurement rather than an assertion.

2. RESTRICTED PERMUTATION.  Permute the extractant -> mean-amplitude map (a) globally
   (Ojala & Garriga Test 1: is there any feature-label relation at all?) and (b) only among
   extractants that share a publication (Test 2 in spirit: is the residual skill mediated by
   publication identity rather than by the ligand?).

3. AVE.  Asymmetric Validation Embedding bias of the *direction* task (heavy-selective =
   "active"), per design, over ECFP4 Tanimoto distance between extractant SMILES.  Reports the
   structural train/test redundancy each design leaves behind.

Usage:
    python generations/gen14_direction/scripts/g14_straw.py                    # all controls, 8 draws
    python generations/gen14_direction/scripts/g14_straw.py --draws 20         # publication-grade
    python generations/gen14_direction/scripts/g14_straw.py --only ave
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen13_separation"))
from gen14 import dirbench as db                                    # noqa: E402
from gen14 import models as M                                       # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame, cell_weights   # noqa: E402
from gen13sep.metrics import per_extractant, summarise               # noqa: E402
from gen13sep.models import tree_pipeline                            # noqa: E402
from gen13sep.splits import all_folds                                # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

DESIGNS = ("A", "B", "BR", "BQ", "BP")
BARCODE_LEVELS = ("extractant", "chemotype", "publication_id")
N_BARCODE_COLS = 39          # same width as TOPO39, so capacity is matched


# --------------------------------------------------------------------------------------
# a run() that takes an explicit cell x d matrix instead of column indices into LEAN
# --------------------------------------------------------------------------------------
def run_matrix(bench, name: str, fit_predict, X: np.ndarray, *, design: str,
               amp_override: np.ndarray | None = None) -> db.OOF:
    """Byte-for-byte db.run(), except X is given directly and the label may be overridden.

    Keeping the fold loop identical is the whole point: a control that uses a different fold
    plan, a different weight or a different training mask prices something other than the leak
    it claims to price.
    """
    amp = bench.coef[:, 0] if amp_override is None else np.asarray(amp_override, dtype=float)
    rich = bench.frame.n_metals.to_numpy() >= db.MIN_METALS
    rows, t0 = [], time.time()
    for f in all_folds(bench.frame, design=design):
        tr = f.train_index[rich[f.train_index]]
        te = f.test_index
        if len(tr) < db.MIN_TRAIN or len(te) < 1 or len(set(amp[tr] < 0)) < 2:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr], mode="balanced")
        ext_tr = bench.frame.extractant.to_numpy()[tr]
        p, mag = fit_predict(X[tr], amp[tr], w, bench.groups[tr], X[te], f.model_seed, ext_tr)
        p = np.asarray(p, dtype=float).ravel()
        mag = np.asarray(mag, dtype=float).ravel()
        for j, ci in enumerate(te):
            rows.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                         "cell_id": bench.frame.cell_id.iat[ci],
                         "extractant": bench.frame.extractant.iat[ci],
                         "chemotype": bench.frame.chemotype.iat[ci],
                         "n_metals": int(bench.frame.n_metals.iat[ci]),
                         "cell_index": int(ci), "amp": float(amp[ci]),
                         "y": int(bench.coef[ci, 0] < 0),     # truth is ALWAYS the real label
                         "p": float(p[j]), "mag": float(mag[j])})
    return db.OOF(cells=pd.DataFrame(rows), design=design, name=name, seconds=time.time() - t0)


def macro_accuracy(oof: db.OOF) -> float:
    """Extractant-macro hit rate, the gen14 headline statistic."""
    return float(db.unit_hits(oof)["hit"].mean())


# --------------------------------------------------------------------------------------
# 1. straw barcodes
# --------------------------------------------------------------------------------------
def barcode(bench, level: str, seed: int, d: int = N_BARCODE_COLS) -> np.ndarray:
    """39 N(0,1) columns drawn ONCE per level of ``level`` and broadcast to that level's cells."""
    key = bench.frame[level].astype(str).to_numpy()
    names = np.unique(key)
    table = np.random.default_rng(seed).standard_normal((len(names), d))
    lut = {n: i for i, n in enumerate(names)}
    return table[np.array([lut[k] for k in key])]


# --------------------------------------------------------------------------------------
# 2. restricted permutation of the extractant -> amplitude map
# --------------------------------------------------------------------------------------
def permuted_amp(bench, mode: str, seed: int) -> np.ndarray:
    """Permute which extractant owns which amplitude; return a cell-level amplitude vector.

    ``global``   permute freely over all extractants (Ojala & Garriga Test 1).
    ``within_pub`` permute only among extractants sharing a modal publication, so the
                 publication -> amplitude association survives and the ligand -> amplitude
                 association does not.
    """
    ext = bench.frame.extractant.astype(str).to_numpy()
    pub = bench.frame.publication_id.astype(str).to_numpy()
    amp = bench.coef[:, 0]
    units = np.unique(ext)
    modal = {e: pd.Series(pub[ext == e]).mode().iat[0] for e in units}
    rng = np.random.default_rng(seed)
    src = {}
    if mode == "global":
        perm = rng.permutation(units)
        src = dict(zip(units, perm))
    elif mode == "within_pub":
        by_pub: dict[str, list[str]] = {}
        for e in units:
            by_pub.setdefault(modal[e], []).append(e)
        for _, members in by_pub.items():
            m = np.array(members)
            src.update(dict(zip(m, rng.permutation(m))))
    else:
        raise ValueError(mode)
    # a cell's new amplitude: the donor extractant's amplitude at the same within-extractant rank
    out = amp.copy()
    idx_of = {e: np.flatnonzero(ext == e) for e in units}
    for e in units:
        tgt, don = idx_of[e], idx_of[src[e]]
        out[tgt] = amp[don][np.arange(len(tgt)) % len(don)]
    return out


# --------------------------------------------------------------------------------------
# 3. AVE bias  (Wallach & Heifets, JCIM 2018)
#    S(V,T,d) = mean_v 1[ NN-distance(v,T) < d ];  H(V,T) = mean_{d in D} S,  D = {0,.01,...,1}
#    B = H(Va,Ta) - H(Va,Ti) + H(Vi,Ti) - H(Vi,Ta)
# --------------------------------------------------------------------------------------
def _H(dist: np.ndarray, D: np.ndarray) -> float:
    if dist.size == 0:
        return float("nan")
    nn = dist.min(axis=1)                      # nearest training neighbour per validation molecule
    return float(np.mean([(nn < d).mean() for d in D]))


def ave_bias(dmat: np.ndarray, lut: dict[str, int], va, vi, ta, ti) -> float:
    D = np.arange(0.0, 1.001, 0.01)
    def sub(v, t):
        if not len(v) or not len(t):
            return np.empty((0, 0))
        return dmat[np.ix_([lut[x] for x in v], [lut[x] for x in t])]
    AA, AI = _H(sub(va, ta), D), _H(sub(va, ti), D)
    II, IA = _H(sub(vi, ti), D), _H(sub(vi, ta), D)
    return AA - AI + II - IA


def ecfp_distance(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fps.append(None if m is None else np.array(gen.GetFingerprint(m), dtype=bool))
    ok = np.array([f is not None for f in fps])
    F = np.zeros((len(fps), 2048), dtype=bool)
    for i, f in enumerate(fps):
        if f is not None:
            F[i] = f
    inter = (F.astype(np.int32) @ F.T.astype(np.int32))
    pop = F.sum(1)
    union = pop[:, None] + pop[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    dist = 1.0 - tan
    dist[~ok, :] = 1.0
    dist[:, ~ok] = 1.0
    return dist


def run_ave(bench) -> pd.DataFrame:
    frame = bench.frame
    ext = frame.extractant.astype(str).to_numpy()
    amp = bench.coef[:, 0]
    rich = frame.n_metals.to_numpy() >= db.MIN_METALS
    units = np.unique(ext[rich])
    # one direction label per extractant: sign of its chemotype-balanced mean amplitude
    w_all = cell_weights(bench.groups, bench.n_obs, mode="balanced")
    label = {}
    for e in units:
        m = (ext == e) & rich
        label[e] = int(np.average(amp[m], weights=w_all[m]) < 0)
    smiles = list(units)
    lut = {s: i for i, s in enumerate(smiles)}
    dmat = ecfp_distance(smiles)
    rows = []
    for design in DESIGNS:
        vals, nn_gap = [], []
        for f in all_folds(frame, design=design):
            tr = f.train_index[rich[f.train_index]]
            te = f.test_index[rich[f.test_index]]
            T = [e for e in np.unique(ext[tr]) if e in lut]
            V = [e for e in np.unique(ext[te]) if e in lut]
            if not T or not V:
                continue
            va = [e for e in V if label[e] == 1]; vi = [e for e in V if label[e] == 0]
            ta = [e for e in T if label[e] == 1]; ti = [e for e in T if label[e] == 0]
            b = ave_bias(dmat, lut, va, vi, ta, ti)
            if np.isfinite(b):
                vals.append(b)
            sub = dmat[np.ix_([lut[e] for e in V], [lut[e] for e in T])]
            nn_gap.append(float(sub.min(axis=1).mean()))
        rows.append({"design": design, "ave_bias": float(np.mean(vals)),
                     "ave_sd": float(np.std(vals, ddof=1)), "n_folds": len(vals),
                     "mean_nn_tanimoto_distance": float(np.mean(nn_gap))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 1b. the same straw control on the *endpoint* metric: extractant-macro MAE of log SF
# --------------------------------------------------------------------------------------
def straw_mae(bench, design: str, draws: int, full_model: bool = True) -> pd.DataFrame:
    """g14_value.py's pair table with straw arms alongside the real one.

    Two straw families, because the two deployed arms fail differently:
      * gen14's one-bit model with a barcode instead of TOPO39;
      * gen13's 209-column ExtraTrees regression with a barcode instead of the 209 columns
        (``full_model``).  Design A's headline 0.437 beats the *direction oracle*'s 0.478 there,
        which is only possible if the regression is reading the magnitude off near-duplicate
        training extractants -- exactly what a barcode can also do.
    """
    FS = db.feature_sets(bench)
    Xtopo = bench.matrix(LEAN_BLOCKS)[:, FS["TOPO39"]]
    X_lean = bench.matrix(LEAN_BLOCKS)
    amp = bench.coef[:, 0]
    rich = bench.frame.n_metals.to_numpy() >= db.MIN_METALS
    cand = M.candidate(M.dir_logistic(), M.amp_constant)

    straws = {f"STRAW1BIT_{lv}_{k}": (lv, k) for lv in ("chemotype", "publication_id")
              for k in range(draws)}
    oof = {}
    for nm, (lv, k) in straws.items():
        oof[nm] = run_matrix(bench, nm, cand, barcode(bench, lv, 20260900 + k),
                             design=design).cells.set_index(["split_seed", "fold", "cell_index"])
    oof["REAL_1BIT"] = run_matrix(bench, "REAL_1BIT", cand, Xtopo,
                                  design=design).cells.set_index(["split_seed", "fold", "cell_index"])

    parts = []
    for f in all_folds(bench.frame, design=design):
        tr, te = f.train_index, f.test_index
        pairs = _pair_frame(bench.frame, bench.Y, te, f.seed, f.fold)
        if pairs.empty:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr])
        rtr = tr[rich[tr]]
        wr = cell_weights(bench.groups[rtr], bench.n_obs[rtr])
        bmean = float(np.average(bench.coef[tr, 1], weights=w))
        amean = float(np.average(bench.coef[tr, 0], weights=w))
        mag_const = float(np.average(np.abs(amp[rtr]), weights=wr))
        curves = {"MEAN_CURVE": np.tile(np.array([amean, bmean]) @ bench.basis, (len(te), 1)),
                  "DIR_ORACLE": np.c_[np.where(amp[te] < 0, -mag_const, mag_const),
                                      np.full(len(te), bmean)] @ bench.basis}
        if any((f.seed, f.fold) not in o.index for o in oof.values()):
            continue                                    # a fold run_matrix's guards skipped
        for nm in list(straws) + ["REAL_1BIT"]:
            sub = oof[nm].loc[(f.seed, f.fold)].reindex(te)
            a = db.decision(sub["p"].to_numpy(), sub["mag"].to_numpy(), "hard")
            curves[nm] = np.c_[a, np.full(len(te), bmean)] @ bench.basis
        if full_model:
            m = tree_pipeline(f.model_seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
            m.fit(X_lean[tr], bench.coef[tr], extratreesregressor__sample_weight=w)
            curves["REAL_209REG"] = np.asarray(m.predict(X_lean[te]),
                                               dtype=float).reshape(len(te), 2) @ bench.basis
            for k in range(min(draws, 4)):          # the regression straw is the expensive one
                Bc = barcode(bench, "chemotype", 20260900 + k, d=209)
                m2 = tree_pipeline(f.model_seed, n_estimators=400, max_features=0.5,
                                   min_samples_leaf=2)
                m2.fit(Bc[tr], bench.coef[tr], extratreesregressor__sample_weight=w)
                curves[f"STRAW209_chemotype_{k}"] = np.asarray(
                    m2.predict(Bc[te]), dtype=float).reshape(len(te), 2) @ bench.basis
        ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
        t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
        for nm, cv in curves.items():
            t[nm] = cv[loc, ia] - cv[loc, ib]
        parts.append(t)

    table = pd.concat(parts, ignore_index=True)
    names = [c for c in table.columns if c in {"MEAN_CURVE", "DIR_ORACLE", "REAL_1BIT",
                                               "REAL_209REG"} or c.startswith("STRAW")]
    board = summarise(per_extractant(table, names), table, names)
    board.insert(0, "design", design)
    # collapse the per-draw straw arms into one row each
    board["family"] = board.arm.str.replace(r"_\d+$", "", regex=True)
    return board


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--only", choices=("barcode", "perm", "ave", "mae", "all"), default="all")
    ap.add_argument("--designs", default=",".join(DESIGNS))
    args = ap.parse_args()
    designs = tuple(args.designs.split(","))

    bench = db.load()
    FS = db.feature_sets(bench)
    Xtopo = bench.matrix(LEAN_BLOCKS)[:, FS["TOPO39"]]
    cand = M.candidate(M.dir_logistic(), M.amp_constant)
    db.RESULTS.mkdir(parents=True, exist_ok=True)
    out = []

    if args.only in ("barcode", "all"):
        for design in designs:
            real = macro_accuracy(run_matrix(bench, "REAL_TOPO39", cand, Xtopo, design=design))
            const = macro_accuracy(run_matrix(bench, "CONST", M.candidate(M.dir_always_heavy),
                                              Xtopo, design=design))
            out.append({"control": "real", "design": design, "level": "TOPO39",
                        "mean": real, "sd": 0.0, "n": 1})
            out.append({"control": "constant", "design": design, "level": "-",
                        "mean": const, "sd": 0.0, "n": 1})
            for level in BARCODE_LEVELS:
                accs = [macro_accuracy(run_matrix(bench, f"STRAW_{level}", cand,
                                                  barcode(bench, level, 20260900 + k),
                                                  design=design))
                        for k in range(args.draws)]
                out.append({"control": "straw", "design": design, "level": level,
                            "mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)),
                            "n": len(accs), "lo": float(np.min(accs)), "hi": float(np.max(accs))})
                print(f"  straw {design:3s} {level:15s} {np.mean(accs):.4f} "
                      f"+-{np.std(accs, ddof=1):.4f}  (real {real:.4f}, const {const:.4f})",
                      flush=True)

    if args.only in ("perm", "all"):
        for design in designs:
            for mode in ("global", "within_pub"):
                accs = [macro_accuracy(run_matrix(bench, f"PERM_{mode}", cand, Xtopo,
                                                  design=design,
                                                  amp_override=permuted_amp(bench, mode,
                                                                            20260900 + k)))
                        for k in range(args.draws)]
                out.append({"control": f"perm_{mode}", "design": design, "level": "TOPO39",
                            "mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)),
                            "n": len(accs), "lo": float(np.min(accs)), "hi": float(np.max(accs))})
                print(f"  perm  {design:3s} {mode:15s} {np.mean(accs):.4f} "
                      f"+-{np.std(accs, ddof=1):.4f}", flush=True)

    if out:
        t = pd.DataFrame(out)
        t.to_csv(db.RESULTS / "g14_straw.csv", index=False)
        print("\n" + t.round(4).to_string(index=False))

    if args.only in ("mae", "all"):
        boards = []
        for design in designs:
            b = straw_mae(bench, design, args.draws)
            boards.append(b)
            g = b.groupby("family")["macro_mae_extractant"].agg(["mean", "std", "min", "max"])
            print(f"\n=== straw on the MAE endpoint, design {design} ===")
            print(g.round(4).to_string())
        pd.concat(boards, ignore_index=True).to_csv(db.RESULTS / "g14_straw_mae.csv", index=False)

    if args.only in ("ave", "all"):
        a = run_ave(bench)
        a.to_csv(db.RESULTS / "g14_ave.csv", index=False)
        print("\n=== AVE bias of the direction task (0 = no structural redundancy) ===")
        print(a.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
