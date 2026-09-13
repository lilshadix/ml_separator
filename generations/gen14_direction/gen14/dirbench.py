"""Gen14 bench: the separation curve as a *direction* plus an *amplitude prior*.

Gen13 stage 3 measured that essentially all the ligand chemistry that transfers across
laboratories in this corpus is one bit per extractant -- the sign of the standardised-Shannon-radius
coefficient of the centred log-D curve -- and that two degrees of freedom (that bit, times the
training-fold mean amplitude magnitude) score as well on the pairwise metric as a 209-column
regression.  Gen14 takes that literally: a candidate is no longer a regression on the coefficient,
it is

    fit_predict(X_train, amp_train, weight_train, group_train, X_test, seed)
        -> (p_heavy_test, magnitude_test)

a probability that the held-out cell is heavy-selective, and a magnitude for it.  The predicted
amplitude is assembled from the two by a fixed, declared rule (``decision``), so the direction and
the prior can be improved independently and their contributions can be told apart.

Everything the gen13 ladder freezes is frozen here and imported, not re-implemented: the cohort,
the physics basis, the per-cell ridge, the fold plans of all five designs, the chemotype-balanced
weights and the pairwise metrics.  The direction target is scored on well-determined cells
(>= 5 measured metals, 289 cells / 82 extractants / 40 chemotypes) with the extractant as the unit
and the chemotype as the resampling block, exactly as ``analysis/stage3/s3_direction.py`` did, so a
number here is directly comparable with the stage-3 tables.
"""
from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "generations" / "gen13_separation") not in sys.path:
    sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))

from gen13sep.amplitude_bench import (ALL_BLOCKS, CHEM_BLOCKS, LEAN_BLOCKS,  # noqa: E402
                                      BenchData, cell_weights, load_bench)
from gen13sep.splits import all_folds  # noqa: E402

CACHE = ROOT / "generations" / "gen14_direction" / "cache" / "bench.pkl"
RESULTS = ROOT / "generations" / "gen14_direction" / "results"
DESIGNS: tuple[str, ...] = ("B", "BR", "BQ", "A", "BP")
MIN_METALS = 5                 # a cell's curve is "well determined" at >= 5 measured metals
MIN_TRAIN = 40                 # stage-3 guard: a fold with fewer training cells is skipped
BOOT_REPS = 10_000
BOOT_SEED = 8675309

#: candidate signature -> (probability heavy-selective, magnitude of the radius coefficient)
FitPredict = Callable[..., tuple[np.ndarray, np.ndarray]]


def load(rebuild: bool = False) -> BenchData:
    """The frozen gen13 bench, cached to disk (building it costs about a minute)."""
    if CACHE.exists() and not rebuild:
        with CACHE.open("rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    bench = load_bench()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as fh:
        pickle.dump(bench, fh, protocol=5)
    print(f"[dirbench] built and cached in {time.time() - t0:.0f}s", flush=True)
    return bench


# --------------------------------------------------------------------------------------
# feature sets
# --------------------------------------------------------------------------------------
def feature_sets(bench: BenchData) -> dict[str, np.ndarray]:
    """Named column-index sets over the LEAN block matrix.

    ``TOPO39`` is gen13's donor-topology family (``coord__dist__`` + ``coord__arm__``): counts of
    bonds on the 2D molecular graph between donor atoms, i.e. the chelate rings the ligand can
    close.  ``DONORS13`` is the frozen gen6 donor census -- the cheapest sensible alternative, and
    the baseline any new representation has to beat (stage-3 audit).
    """
    lean = bench.columns(LEAN_BLOCKS)
    idx = {c: i for i, c in enumerate(lean)}
    topo = [c for c in lean if c.startswith("coord__dist__") or c.startswith("coord__arm__")]
    donors = [c for c in lean if c.startswith("chem__")]
    physchem = [c for c in lean if not c.startswith(("cond__", "massact__", "chem__", "coord__"))]
    massact = [c for c in lean if c.startswith("massact__")]
    cond = [c for c in lean if c.startswith("cond__")]
    coord = [c for c in lean if c.startswith("coord__")]
    donor_elem = [c for c in lean if c.startswith("coord__donor__")]
    arch = [c for c in lean if c.startswith("coord__arch__")]
    motif = [c for c in lean if c.startswith("coord__motif__")]
    bite = ["coord__dist__frac_donor_pairs_within_3", "coord__dist__donor_pair_min",
            "coord__dist__donor_pair_mean", "coord__dist__n_five_membered_chelate_pairs",
            "coord__dist__n_donor_pairs_within_3", "coord__dist__donor_network_diameter"]
    sets = {
        "BITE6": [c for c in bite if c in idx],
        "TOPO39": topo,
        "DONORS13": donors,
        "PHYSCHEM10": physchem,
        "MASSACT8": massact,
        "COND64": cond,
        "COORD114": coord,
        "CHEM137": physchem + donors + coord,
        "LEAN209": lean,
        "TOPO_DONORS": topo + donors,
        "TOPO_ELEM": topo + donor_elem,
        "TOPO_ELEM_DONORS": topo + donor_elem + donors,
        "TOPO_ARCH": topo + arch,
        "TOPO_MOTIF": topo + motif,
        "TOPO_MASSACT": topo + massact,
        "TOPO_ELEM_MASSACT": topo + donor_elem + massact,
        "TOPO_ELEM_DONORS_MASSACT": topo + donor_elem + donors + massact,
    }
    return {k: np.array([idx[c] for c in v], dtype=int) for k, v in sets.items()}


# --------------------------------------------------------------------------------------
# out-of-fold evaluation
# --------------------------------------------------------------------------------------
@dataclass
class OOF:
    """Out-of-fold predictions of one candidate under one design."""
    cells: pd.DataFrame          # one row per (seed, fold, test cell): p, magnitude, truth
    design: str
    name: str
    seconds: float


def run(bench: BenchData, name: str, fit_predict: FitPredict, *, features: np.ndarray,
        design: str = "BP", rich_only_train: bool = True,
        weight_mode: str = "balanced") -> OOF:
    """Fit the candidate on every fold of ``design`` and collect held-out predictions.

    ``rich_only_train`` keeps stage 3's convention that a cell whose curve is not well determined
    is not a training example for the direction.  Predictions are made for *every* held-out cell,
    so the same run also feeds the pairwise-MAE scoring, where all cells count.
    """
    X = bench.matrix(LEAN_BLOCKS)[:, features]
    amp = bench.coef[:, 0]
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    rows = []
    t0 = time.time()
    for f in all_folds(bench.frame, design=design):
        tr = f.train_index[rich[f.train_index]] if rich_only_train else f.train_index
        te = f.test_index
        if len(tr) < MIN_TRAIN or len(te) < 1 or len(set(amp[tr] < 0)) < 2:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr], mode=weight_mode)
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
                         "y": int(amp[ci] < 0), "p": float(p[j]), "mag": float(mag[j])})
    return OOF(cells=pd.DataFrame(rows), design=design, name=name, seconds=time.time() - t0)


# --------------------------------------------------------------------------------------
# scoring: direction
# --------------------------------------------------------------------------------------
def unit_hits(oof: OOF, *, min_metals: int = MIN_METALS) -> pd.DataFrame:
    """One row per extractant: mean hit over its well-determined cells, averaged over seeds."""
    b = oof.cells[oof.cells.n_metals >= min_metals].copy()
    b["hit"] = ((b.p >= 0.5).astype(int) == b.y).astype(float)
    per_seed = b.groupby(["split_seed", "extractant", "chemotype"])["hit"].mean().reset_index()
    return per_seed.groupby(["extractant", "chemotype"])["hit"].mean().reset_index()


class Blocked:
    """Chemotype-blocked bootstrap with one shared set of resamples for every model compared."""

    def __init__(self, chemotypes: Sequence[str], reps: int = BOOT_REPS, seed: int = BOOT_SEED):
        self.names = sorted(set(chemotypes))
        rng = np.random.default_rng(seed)
        self.picks = rng.integers(0, len(self.names), size=(reps, len(self.names)))

    def draws(self, unit: pd.DataFrame, value: str = "hit") -> tuple[float, np.ndarray]:
        members = [np.flatnonzero(unit.chemotype.to_numpy() == c) for c in self.names]
        v = unit[value].to_numpy(dtype=float)
        take = [np.concatenate([members[j] for j in row]) for row in self.picks]
        return float(v.mean()), np.array([v[t].mean() for t in take])


def direction_board(oofs: Sequence[OOF], *, baseline: str | None = None) -> pd.DataFrame:
    """Macro accuracy with chemotype-blocked intervals; paired gains against ``baseline``."""
    units = {o.name: unit_hits(o) for o in oofs}
    boot = Blocked(next(iter(units.values())).chemotype)
    base = units[baseline].set_index("extractant")["hit"] if baseline is not None else None
    out = []
    for o in oofs:
        u = units[o.name]
        m, d = boot.draws(u)
        rich = o.cells[o.cells.n_metals >= MIN_METALS]
        rec = {"design": o.design, "model": o.name, "n_units": len(u),
               "n_chemotypes": int(u.chemotype.nunique()), "macro_accuracy": m,
               "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
               "pooled_accuracy": float(((rich.p >= 0.5).astype(int) == rich.y).mean()),
               "seconds": round(o.seconds, 1)}
        if base is not None and o.name != baseline:
            delta = u["hit"].to_numpy() - base.reindex(u.extractant.to_numpy()).to_numpy()
            du = u.assign(hit=delta)
            g, gd = boot.draws(du)
            rec.update({"gain": g, "gain_lo": float(np.quantile(gd, 0.025)),
                        "gain_hi": float(np.quantile(gd, 0.975)),
                        "p_two_sided": float(2 * min((gd <= 0).mean(), (gd >= 0).mean())),
                        "units_better": int((du.hit > 0).sum()),
                        "units_worse": int((du.hit < 0).sum())})
        out.append(rec)
    return pd.DataFrame(out).sort_values("macro_accuracy", ascending=False).reset_index(drop=True)


def paired_gain(a: OOF, b: OOF, reps: int = BOOT_REPS) -> dict:
    """``b - a`` macro accuracy per extractant, chemotype-blocked (positive favours ``b``)."""
    ua = unit_hits(a).set_index(["extractant", "chemotype"])
    ub = unit_hits(b).set_index(["extractant", "chemotype"])
    j = (ub["hit"] - ua["hit"]).dropna().reset_index()
    boot = Blocked(j.chemotype, reps=reps)
    point, d = boot.draws(j)
    loco = [float(j[j.chemotype != c].hit.mean()) for c in sorted(set(j.chemotype))]
    return {"design": a.design, "reference": a.name, "candidate": b.name, "gain": point,
            "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "n_units": len(j), "units_better": int((j.hit > 0).sum()),
            "units_worse": int((j.hit < 0).sum()),
            "loco_min": float(np.min(loco)), "loco_max": float(np.max(loco)),
            "loco_sign_stable": bool(np.all(np.sign(loco) == np.sign(point))) if point != 0 else False}


# --------------------------------------------------------------------------------------
# turning (direction, magnitude) into a curve
# --------------------------------------------------------------------------------------
def decision(p: np.ndarray, mag: np.ndarray, rule: str = "hard") -> np.ndarray:
    """Turn (probability heavy, magnitude) into a predicted radius coefficient.

    ``hard``     the stage-3 rule: sign from the 0.5 threshold, magnitude as given.
    ``expected`` the posterior mean, ``(1 - 2p) * magnitude``: shrinks toward the flat curve
                 exactly as far as the classifier is unsure.
    ``sqrt``     a middle rule, ``-sign(2p-1) * |2p-1|**0.5 * magnitude``.
    """
    p = np.asarray(p, dtype=float)
    mag = np.asarray(mag, dtype=float)
    if rule == "hard":
        return np.where(p >= 0.5, -1.0, 1.0) * mag
    if rule == "expected":
        return (1.0 - 2.0 * p) * mag
    if rule == "sqrt":
        s = 2.0 * p - 1.0
        return -np.sign(s) * np.sqrt(np.abs(s)) * mag
    raise ValueError(f"unknown decision rule {rule!r}")
