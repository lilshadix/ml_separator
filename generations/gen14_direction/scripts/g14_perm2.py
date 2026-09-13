"""Gen14 permutation controls, corrected.

The ``perm within_pub`` arm in ``g14_straw.py`` is degenerate.  It permutes the
extractant -> amplitude map only among extractants that share a modal publication, but 24 of the
90 extractants are the sole extractant of their publication, so they are frozen by construction,
and inside the groups that do move a uniform permutation still maps an extractant to itself with
probability 1/k.  Over the corpus that is E[fixed points] = 46.7 %.  A control that leaves half
the map intact and is then scored on *all* extractants -- including the 27 % that could never
move -- is not a null: it reports 0.8326 against a real 0.8346 and invites the reader to conclude
that the publication, not the ligand, carries the signal.  It shows nothing of the kind.

Three fixes, each of which is needed for the number to mean anything:

  1. DERANGEMENT.  Inside every modal-publication group of size >= 2, permute with no fixed point.
  2. RESTRICTED SCORING.  Score only the extractants that were actually free to move, and score
     the real model on that same subset, so the comparison is paired.
  3. SIGN-FLIP ACCOUNTING.  The endpoint is a *direction*, so a permutation that reshuffles
     amplitudes inside a publication whose extractants all share one sign changes no label at all.
     Report the realised flip rate; a within-publication null with a low flip rate is weak by
     construction and must be read as such, not as evidence of anything.

The matched control is a GLOBAL derangement restricted to the same movable set: same number of
units moved, same scoring subset, but the publication -> amplitude association is broken too.
Reading the two together separates "the ligand carries it" from "the publication carries it":

    global derangement    -> chance          and  within-pub derangement -> chance
        the map is learned from the ligand, and publication identity adds nothing.
    global derangement    -> chance          and  within-pub derangement -> stays high
        the model is reading publication identity out of the features.

Follows Ojala & Garriga's two permutation tests (ICDM 2009; JMLR 11:1833-1863, 2010), whose Test 2
is explicitly a *restricted* randomisation, and Chuang & Keiser's adversarial-control protocol
(Science 362:eaat8603, 2018; ACS Chem Biol 13:2819, 2018).

Usage:
    python generations/gen14_direction/scripts/g14_perm2.py --draws 20 --designs BP,A
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen13_separation"))
from gen14 import dirbench as db                                     # noqa: E402
from gen14 import models as M                                        # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g14_straw import run_matrix                                     # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def pub_groups(bench) -> tuple[dict[str, list[str]], list[str]]:
    """Modal-publication groups of extractants, and the subset free to move (group size >= 2)."""
    ext = bench.frame.extractant.astype(str).to_numpy()
    pub = bench.frame.publication_id.astype(str).to_numpy()
    units = np.unique(ext)
    modal = {e: pd.Series(pub[ext == e]).mode().iat[0] for e in units}
    by_pub: dict[str, list[str]] = {}
    for e in units:
        by_pub.setdefault(modal[e], []).append(e)
    movable = sorted(e for g in by_pub.values() if len(g) >= 2 for e in g)
    return by_pub, movable


def derange(items: list[str], rng: np.random.Generator, tries: int = 500) -> list[str]:
    """A permutation of ``items`` with no fixed point (cyclic shift as the guaranteed fallback)."""
    a = np.asarray(items, dtype=object)
    if len(a) < 2:
        return list(a)
    for _ in range(tries):
        p = rng.permutation(a)
        if not np.any(p == a):
            return list(p)
    return list(np.roll(a, 1))


def permuted_amp2(bench, mode: str, seed: int) -> tuple[np.ndarray, list[str], float]:
    """Cell-level amplitudes after a fixed-point-free permutation of the extractant -> amp map.

    ``within_pub`` deranges inside each modal-publication group, preserving publication -> amp.
    ``global``     deranges the same movable set freely, breaking publication -> amp as well.

    Returns the new amplitude vector, the movable extractants, and the realised *direction*
    flip rate over them -- the fraction whose sign actually changed, which is the only part of
    the permutation the direction task can even see.
    """
    ext = bench.frame.extractant.astype(str).to_numpy()
    amp = bench.coef[:, 0]
    by_pub, movable = pub_groups(bench)
    rng = np.random.default_rng(seed)

    src: dict[str, str] = {}
    if mode == "within_pub":
        for _, members in by_pub.items():
            if len(members) >= 2:
                src.update(dict(zip(sorted(members), derange(sorted(members), rng))))
    elif mode == "global":
        src.update(dict(zip(movable, derange(movable, rng))))
    elif mode.startswith("global_flipmatched"):
        # Same movable set, same number of realised sign flips as the within-publication
        # derangement, but donors are drawn without regard to publication.  This is the only
        # arm that isolates "does the publication boundary matter", because a permutation that
        # flips fewer direction labels degrades less for reasons that have nothing to do with
        # publications: the raw global derangement flips 0.55 of the signs and the within-pub
        # one flips 0.29, so their gap is mostly a flip-rate artefact.
        k = int(round(float(mode.split(":")[1]) * len(movable)))
        sgn = {e: float(np.mean(amp[np.flatnonzero(ext == e)])) < 0 for e in movable}
        neg = [e for e in movable if sgn[e]]
        pos = [e for e in movable if not sgn[e]]
        k = min(k, len(neg), len(pos))
        flip_set = list(rng.choice(np.asarray(movable, dtype=object),
                                   size=k, replace=False))
        # a flipped unit takes an opposite-sign donor; everyone else takes a same-sign donor
        for e in flip_set:
            pool = pos if sgn[e] else neg
            src[e] = str(rng.choice(np.asarray(pool, dtype=object)))
        rest_neg = [e for e in neg if e not in set(flip_set)]
        rest_pos = [e for e in pos if e not in set(flip_set)]
        for pool in (rest_neg, rest_pos):
            if len(pool) >= 2:
                src.update(dict(zip(sorted(pool), derange(sorted(pool), rng))))
    else:
        raise ValueError(mode)

    idx_of = {e: np.flatnonzero(ext == e) for e in np.unique(ext)}
    out = amp.copy()
    # one direction label per extractant, before and after, on the same convention as run_matrix
    sign_before = {e: float(np.mean(amp[idx_of[e]])) < 0 for e in movable}
    for tgt_e, don_e in src.items():
        tgt, don = idx_of[tgt_e], idx_of[don_e]
        out[tgt] = amp[don][np.arange(len(tgt)) % len(don)]
    sign_after = {e: float(np.mean(out[idx_of[e]])) < 0 for e in movable}
    flip = float(np.mean([sign_before[e] != sign_after[e] for e in movable]))
    return out, movable, flip


def macro_accuracy_on(oof: db.OOF, keep: list[str]) -> float:
    """Extractant-macro hit rate restricted to ``keep`` -- the units the permutation could move."""
    u = db.unit_hits(oof)
    u = u[u.extractant.astype(str).isin(set(keep))]
    return float(u["hit"].mean()) if len(u) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--designs", default="BP,A,B,BR,BQ")
    ap.add_argument("--modes", default="global,within_pub")
    args = ap.parse_args()

    bench = db.load()
    FS = db.feature_sets(bench)
    Xtopo = bench.matrix(LEAN_BLOCKS)[:, FS["TOPO39"]]
    cand = M.candidate(M.dir_logistic(), M.amp_constant)
    _, movable = pub_groups(bench)
    print(f"movable extractants (modal-publication group size >= 2): "
          f"{len(movable)} of {bench.frame.extractant.nunique()}", flush=True)

    rows = []
    for design in args.designs.split(","):
        real_oof = run_matrix(bench, "REAL", cand, Xtopo, design=design)
        real_all = float(db.unit_hits(real_oof)["hit"].mean())
        real_mov = macro_accuracy_on(real_oof, movable)
        rows.append({"design": design, "control": "real", "scope": "movable",
                     "mean": real_mov, "sd": 0.0, "n": 1, "flip_rate": np.nan})
        rows.append({"design": design, "control": "real", "scope": "all",
                     "mean": real_all, "sd": 0.0, "n": 1, "flip_rate": np.nan})
        print(f"{design:3s} real  all={real_all:.4f}  movable={real_mov:.4f}", flush=True)
        for mode in args.modes.split(","):
            accs, flips = [], []
            for k in range(args.draws):
                amp2, mov, flip = permuted_amp2(bench, mode, 20260900 + k)
                o = run_matrix(bench, f"PERM_{mode}", cand, Xtopo, design=design,
                               amp_override=amp2)
                accs.append(macro_accuracy_on(o, mov))
                flips.append(flip)
            rows.append({"design": design, "control": f"perm2_{mode}", "scope": "movable",
                         "mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)),
                         "n": len(accs), "lo": float(np.min(accs)), "hi": float(np.max(accs)),
                         "flip_rate": float(np.mean(flips))})
            print(f"{design:3s} perm2 {mode:11s} movable={np.mean(accs):.4f} "
                  f"+-{np.std(accs, ddof=1):.4f}  (real movable {real_mov:.4f}, "
                  f"realised sign-flip rate {np.mean(flips):.2f})", flush=True)

    t = pd.DataFrame(rows)
    db.RESULTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(db.RESULTS / "g14_perm2.csv", index=False)
    print("\n" + t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
