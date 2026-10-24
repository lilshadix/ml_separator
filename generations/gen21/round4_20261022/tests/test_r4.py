"""Focused tests for round 4's stage 0: support invariance, B1 = round-3 R14, the orphan rule, fit poisoning,
intervention with two positive controls, synthetic null / signal recovery, cluster nesting, task construction and
cache identity for one fold (the full task-identity check is scripts/r4_task_identity.py).  None compares a
held-out prediction with its target.

    .venv/Scripts/python.exe -m pytest generations/gen21/round4_20261022/tests -q -p no:cacheprovider -W ignore
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r4 import metrics as M4  # noqa: E402
from r4 import paths  # noqa: E402
from r4 import refs as RF  # noqa: E402
from r4 import registry as RG  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from r3 import harness as H  # noqa: E402
from r3 import interp  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

FOLD = ("BP", 130363, 2)


@pytest.fixture(scope="module")
def bench():
    return D.load_bench()


@pytest.fixture(scope="module")
def rec():
    return C.load(*FOLD)


def _random_reads(bench, rec, n_sets=4, seed=1):
    """(cell, y_masked) for held-out cells under DOPT_Q and random legal read sets (no target is used)."""
    rng = np.random.default_rng(seed)
    out = []
    for cell in rec["outer"]["cells"]:
        if cell["k_max"] < 3:
            continue
        sets = [frozenset(m for p in cell["seqs"]["DOPT_Q"][:3] for m in p)]
        sets += CO._random_legal(cell, 3, n_sets, f"test{seed}")
        for s in sets:
            ym = np.full(N_LN, np.nan)
            for m in s:
                ym[m] = rng.normal()
            out.append((cell, ym))
    return out


def test_gd_potential_equals_round3_gd_dip(bench, rec):
    for cell, ym in _random_reads(bench, rec):
        pot = RF.potentials(ym, cell["metals"], (RF.GD,))
        dip = CO.gd_dip(ym, cell["metals"])
        assert pot.keys() == dip.keys()
        for b in dip:
            assert pot[b][0] == pytest.approx(dip[b], abs=1e-15)


def test_potential_form_equals_interpolate_then_add_back(bench, rec):
    """level = I_S[y_S - u_S] + u  ==  PWL_R + sum_j u_j phi_j, and every support is kept exactly."""
    rng = np.random.default_rng(7)
    for cell, ym in _random_reads(bench, rec):
        u = rng.normal(size=N_LN)
        fit = {"spec": RF.SPECS["B2"] | {"geo": False}, "kind": "R", "coef": u}
        lev = RF.predict_levels(fit, SimpleNamespace(y_masked=ym), cell["metals"])
        shifted = np.where(np.isfinite(ym), ym - u, np.nan)
        direct = interp.pwl_levels(shifted, cell["metals"], "R") + u
        read = [m for m in range(N_LN) if np.isfinite(ym[m])]
        for m in cell["metals"]:
            assert lev[m] == pytest.approx(direct[m], abs=1e-12)
        for m in read:
            assert lev[m] == ym[m]


def test_infinite_shrinkage_returns_pwl(bench, rec):
    for name in ("B1", "B2"):
        fit = {"spec": RF.SPECS[name], "kind": "R", "coef": RF._solve((np.zeros((3, RF.n_features(RF.SPECS[name]))),
                                                                        np.zeros(3), np.ones(3)), math.inf, None)}
        for cell, ym in _random_reads(bench, rec, n_sets=1):
            lev = RF.predict_levels(fit, SimpleNamespace(y_masked=ym), cell["metals"])
            base = interp.pwl_levels(ym, cell["metals"], "R")
            assert np.array_equal(lev, base, equal_nan=True)


@pytest.fixture(scope="module")
def fits(bench, rec):
    return {n: RF.fit_reference(bench, rec, bench.Y, n) for n in ("B1", "B2")}


def test_b1_matches_round3_r14_when_finite(bench, rec, fits):
    r14 = CO.fit_corrector(bench, rec, bench.Y, fset="Dnc")
    b1 = fits["B1"]
    for lam, s in r14["inner_scores"].items():
        assert b1["inner_scores"][str(lam)] == pytest.approx(s, rel=0, abs=0)
    if not math.isinf(b1["lambda"]):
        assert b1["lambda"] == r14["lambda"]
        assert np.array_equal(b1["coef"], r14["coef"])


def test_fit_uses_outer_training_cells_only(bench, rec, fits):
    """Fit poisoning: every label-bearing field outside the outer training fold (log D, cached gen13
    coefficients, the frame's label columns) replaced by random numbers; the fitted coefficients, lambda and
    inner scores must be identical."""
    from r4.leakage import poisoned_bench
    pb, cols = poisoned_bench(bench, rec["outer"]["train_index"])
    assert len(cols) >= 29 and not np.array_equal(pb.Y, bench.Y, equal_nan=True)
    for n in ("B1", "B2"):
        p = RF.fit_reference(pb, rec, pb.Y, n)
        assert p["lambda"] == fits[n]["lambda"]
        assert np.array_equal(p["coef"], fits[n]["coef"])
        assert p["inner_scores"] == fits[n]["inner_scores"]


def test_intervention_passes_for_references_and_catches_a_leak(bench, rec, fits):
    cfg = RG.get("REF", bench)
    fitted = {(FOLD[1], FOLD[2]): {"ref:B1": fits["B1"], "ref:B2": fits["B2"]}}
    methods = {"B1": cfg.methods["B1"], "B2": cfg.methods["B2"], "HIST_D7": cfg.methods["HIST_D7"]}
    iv = H.intervention(bench, FOLD[0], methods, fitted, ks=(3,), fold_filter={(FOLD[1], FOLD[2])})
    assert iv["passed"] and iv["n_predictions"] > 600          # about 232 C3 queries x 3 methods in this fold

    def leaky(block, cell, obs, k):                      # positive control: peeks at an unread value
        v = RF.predict_levels(fits["B1"], obs, cell["metals"])
        hidden = [m for m in cell["metals"] if m not in obs.read]
        v[hidden[0]] = block["Y"][cell["ci"]][hidden[0]]
        return v
    iv2 = H.intervention(bench, FOLD[0], {"LEAK": leaky}, fitted, ks=(3,), fold_filter={(FOLD[1], FOLD[2])})
    assert not iv2["passed"]


def test_intervention_with_factory_catches_a_bench_closure(bench, fits):
    """Second positive control: a method that closes over bench.Y (not block['Y']).  It passes the plain
    intervention (documented gap) and is caught when the scrambled pass rebuilds methods from the scrambled bench,
    as r4_run.py does."""
    fitted = {(FOLD[1], FOLD[2]): {"ref:B1": fits["B1"]}}

    def make_leak(b):
        def leak(block, cell, obs, k):
            v = RF.predict_levels(fits["B1"], obs, cell["metals"])
            hidden = [m for m in cell["metals"] if m not in obs.read]
            v[hidden[0]] = b.Y[cell["ci"]][hidden[0]]
            return v
        return {"LEAK": leak}
    plain = H.intervention(bench, FOLD[0], make_leak(bench), fitted, ks=(3,), fold_filter={(FOLD[1], FOLD[2])})
    built = H.intervention(bench, FOLD[0], make_leak(bench), fitted, ks=(3,), fold_filter={(FOLD[1], FOLD[2])},
                           methods_factory=make_leak)
    assert plain["passed"] and not built["passed"]


def test_orphan_rule_covers_every_training_cell(bench):
    """BP 155921 fold 2: g21.cache skipped one inner block (408 training cells in no block); round 4 adds them."""
    from g21.cache import cell_view
    from gen20k import design as DS
    r = C.load("BP", 155921, 2)
    tr = set(map(int, r["outer"]["train_index"]))
    usable = {ci for ci in tr if (lambda dz: bool(dz.query) and dz.k_max >= 1)(DS.make_design(cell_view(bench, ci), "adjacent", "metal"))}
    cells = RF.training_cells(bench, r)
    assert {c["ci"] for c in cells} == usable and len(cells) == len(usable)    # cells without a query never train
    orph = RF.orphan_cells(bench, r)
    assert len(orph) == 235 and sum(c["k_max"] >= 3 for c in orph) == 148
    assert RF.orphan_cells(bench, C.load(*FOLD)) == []


def _synthetic_Y(bench, rec, u, noise, seed):
    """Training cells only: a cell-specific quadratic in -radius plus the potentials u plus noise, on the
    cell's own NaN pattern (held-out rows are irrelevant to the fit and left as they are)."""
    rng = np.random.default_rng(seed)
    Ys = np.array(bench.Y, copy=True)
    x = interp.X_R
    xs = (x - x.mean()) / x.std()
    for ci in np.asarray(rec["outer"]["train_index"]):
        a, b, c = rng.normal(0, [1.0, 1.0, 0.3])
        curve = a + b * xs + c * xs ** 2 + u + rng.normal(0, noise, N_LN)
        Ys[ci] = np.where(np.isfinite(bench.Y[ci]), curve, np.nan)
    return Ys


def test_synthetic_signal_is_recovered(bench, rec):
    u = np.zeros(N_LN)
    u[RF.GD] = -0.15
    Ys = _synthetic_Y(bench, rec, u, 0.03, 11)
    pb = copy.copy(bench)
    pb.Y = Ys
    b1 = RF.fit_reference(pb, rec, Ys, "B1")
    assert not math.isinf(b1["lambda"])
    assert b1["coef"][0] == pytest.approx(-0.15, abs=0.03)
    b2 = RF.fit_reference(pb, rec, Ys, "B2")
    assert not math.isinf(b2["lambda"])
    pot = b2["coef"][:N_LN]
    x = interp.X_R                   # gauge-invariant: Gd relative to the line through its neighbours Eu and Tb
    eu, tb = RF.GD - 1, RF.GD + 1
    t = (x[RF.GD] - x[eu]) / (x[tb] - x[eu])
    assert pot[RF.GD] - ((1 - t) * pot[eu] + t * pot[tb]) < -0.08


def test_synthetic_null_gives_no_material_correction(bench, rec):
    Ys = _synthetic_Y(bench, rec, np.zeros(N_LN), 0.03, 12)
    pb = copy.copy(bench)
    pb.Y = Ys
    b1 = RF.fit_reference(pb, rec, Ys, "B1")
    assert math.isinf(b1["lambda"]) or abs(b1["coef"][0]) < 0.02


def test_clusters_nest_chemotypes(bench):
    import pandas as pd
    t = pd.read_parquet(paths.R3DIR / "predictions" / "B0_BP.parquet", columns=["chemotype", "publication", "split_seed",
                                                                                 "k", "k_max", "cell_id", "extractant"])
    t = M4.add_clusters(t[(t.k == 3) & (t.k_max >= 3)], bench.frame)
    assert t.groupby("chemotype").cluster_cpg.nunique().max() == 1
    assert t.groupby("publication").cluster_cpg.nunique().max() == 1
    assert t.groupby("copy_group").cluster_cpg.nunique().max() == 1


def test_registry_ref_fits_only_declared_parts(bench):
    cfg = RG.get("REF", bench)
    assert cfg.parts == ("hbk", "cor:Dnc", "ref:B1", "ref:B2") and cfg.ks == (3,) and cfg.policy == "DOPT_Q"
    assert cfg.version == "V1" and cfg.K == 3
    assert copy.copy(RF.LAMBDA_GRID)[-1] == math.inf


def test_task_construction_and_cache_identity_one_fold(bench):
    import json
    sys.path.insert(0, str(paths.R3DIR / "scripts"))
    from r3_manifest import content_digest
    r = C.load(*FOLD)
    man = json.loads((paths.R3DIR / "manifests" / "inputs_sha256.json").read_text())
    assert content_digest(r) == man["cache_content"][f"{FOLD[0]}_{FOLD[1]}_{FOLD[2]}.pkl"]
    from gen20k.splits import folds_for
    n_cells, n_q = 0, 0
    for f in folds_for(bench.frame, "BP", seeds=[FOLD[1]]):
        rr = C.load("BP", f.seed, f.fold)
        c3 = [c for c in rr["outer"]["cells"] if c["k_max"] >= 3]
        n_cells += len(c3)
        n_q += sum(len(c["query"]) for c in c3)
    assert (n_cells, n_q) == (229, 1802)
