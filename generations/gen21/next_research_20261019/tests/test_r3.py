"""Round-3 contract tests (synthetic arrays only where marked SYNTHETIC; never used for results)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import corrector as CO  # noqa: E402
from r3 import interp as IP  # noqa: E402
from r3 import paths  # noqa: E402
from r3 import policy as PO  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import design as DS  # noqa: E402

FOLD = ("BP", 130363, 2)


@pytest.fixture(scope="module")
def bench():
    return D.load_bench()


@pytest.fixture(scope="module")
def rec():
    return C.load(*FOLD)


def _poisoned(bench, rec, value=55.0):
    te = rec["outer"]["test_index"]
    Yp = bench.Y.copy()
    Yp[te] = np.where(np.isfinite(Yp[te]), value, np.nan)
    return SimpleNamespace(Y=Yp, groups=bench.groups, n_obs=bench.n_obs, basis=bench.basis, frame=bench.frame, coef=bench.coef,
                           frames=bench.frames), Yp


def test_pwl_keeps_reads_and_ignores_hidden(bench, rec):
    rng = np.random.default_rng(0)
    for cell in rec["outer"]["cells"][:30]:
        for k in (1, 2, 3):
            if k > len(cell["seqs"]["DOPT_Q"]):
                continue
            o1 = E.observe(bench.Y[cell["ci"]], cell, cell["seqs"]["DOPT_Q"], k)
            o2 = E.observe(E.scramble_hidden(bench.Y[cell["ci"]], o1.read, rng), cell, cell["seqs"]["DOPT_Q"], k)
            for kind in ("R", "Z"):
                v1 = IP.pwl_levels(o1.y_masked, cell["metals"], kind)
                assert np.array_equal(v1, IP.pwl_levels(o2.y_masked, cell["metals"], kind), equal_nan=True)
                for m in o1.read:
                    assert v1[m] == bench.Y[cell["ci"]][m]
            assert np.array_equal(IP.pchip_levels(o1.y_masked, cell["metals"]), IP.pchip_levels(o2.y_masked, cell["metals"]), equal_nan=True)


def test_training_situations_only_training_cells(bench, rec):
    sits = CO.training_situations(bench, rec, bench.Y)
    tr = set(map(int, rec["outer"]["train_index"]))
    assert sits and all(s["ci"] in tr for s in sits)
    w = {}
    for s in sits:
        w[s["ci"]] = w.get(s["ci"], 0.0) + s["w"]
    from gen13sep.amplitude_bench import cell_weights
    trn = np.asarray(rec["outer"]["train_index"])
    ref = dict(zip(trn.tolist(), cell_weights(bench.groups[trn], bench.n_obs[trn]).tolist()))
    for ci, tot in w.items():                       # mask weights of a cell sum to its cell weight (if all masks have rows)
        assert tot <= ref[ci] + 1e-12


@pytest.mark.parametrize("which", ["corrector_G", "corrector_GP", "error_table"])
def test_fits_ignore_held_out_labels(bench, rec, which):
    pb, Yp = _poisoned(bench, rec)
    if which.startswith("corrector"):
        fset = which.split("_")[1]
        a, b = CO.fit_corrector(bench, rec, bench.Y, fset=fset), CO.fit_corrector(pb, rec, Yp, fset=fset)
        assert a["lambda"] == b["lambda"] and np.allclose(a["coef"], b["coef"])
    else:
        a, b = PO.fit_error_table(bench, rec, bench.Y), PO.fit_error_table(pb, rec, Yp)
        assert a["table"] == b["table"]


def test_inner_table_ignores_held_out_labels(bench, rec):
    from r3 import inner_preds as INP
    pb, Yp = _poisoned(bench, rec)
    a, la = INP.inner_table(bench, rec, kernels=("exp", "se"), noise_mult=(1.0,))
    b, lb = INP.inner_table(pb, rec, kernels=("exp", "se"), noise_mult=(1.0,))
    assert la == lb
    cols = [c for c in a.columns if c.startswith("HBK") or c.startswith("PWL")]
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)


def test_pol_err_legal_and_label_free(bench, rec):
    err = PO.fit_error_table(bench, rec, bench.Y)
    for cell in rec["outer"]["cells"][:25]:
        if cell["k_max"] < 3:
            continue
        S = PO.choose(cell, err, 3)
        read = set(S)
        assert len(S) == 4 and not any(a in read and b in read for a, b in cell["query"])
        dz = DS.make_design(E._view(cell), "adjacent", "metal")
        assert DS.max_independent_size(cell["metals"], dz.query, frozenset(S)) == DS.max_independent_size(cell["metals"], dz.query, frozenset())
    # the choice reads no label of the held-out cell: same table -> same sets whatever the cell's values
    pb, Yp = _poisoned(bench, rec)
    err2 = PO.fit_error_table(pb, rec, Yp)
    for cell in rec["outer"]["cells"][:25]:
        if cell["k_max"] >= 3:
            assert PO.choose(cell, err, 3) == PO.choose(cell, err2, 3)


def test_gd_dip_feature_consistent(bench, rec):
    """The Gd feature is +1 at an unread Gd and -w_Gd(b) at metals interpolated from a read Gd."""
    from r3 import interp as IPm
    for cell in rec["outer"]["cells"][:40]:
        if 6 not in cell["metals"] or cell["k_max"] < 3:
            continue
        o = E.observe(bench.Y[cell["ci"]], cell, cell["seqs"]["DOPT_Q"], 3)
        g = CO.gd_dip(o.y_masked, cell["metals"])
        if 6 in o.read:
            # shifting the read Gd by +1 moves every unread metal's PWL value by w_Gd(b) = -g[b]
            y2 = o.y_masked.copy()
            y2[6] += 1.0
            d = IPm.pwl_levels(y2, cell["metals"]) - IPm.pwl_levels(o.y_masked, cell["metals"])
            for b, v in g.items():
                assert abs(d[b] + v) < 1e-9
        else:
            assert g[6] == 1.0


@pytest.mark.parametrize("k", [5, 6])
def test_pol_err_k_legal(bench, rec, k):
    err = PO.fit_error_table(bench, rec, bench.Y, k=k)
    for cell in rec["outer"]["cells"][:30]:
        if cell["k_max"] < k:
            continue
        S = PO.choose(cell, err, k)
        read = set(S)
        assert len(S) == k + 1 and not any(a in read and b in read for a, b in cell["query"])


def test_v2_record(bench, rec):
    from r3 import datav2
    Y = datav2.matrix(bench)
    r2 = datav2.v2_record(rec, Y)
    q = set(datav2.V2_1_CELLS) | set(datav2.V2_2_DROP) | set(datav2.V2_3_CELLS)
    for blk in [r2["outer"]] + r2["inner"]:
        for cell in blk["cells"]:
            assert cell["cell_id"] not in q
            assert all(np.isfinite(Y[cell["ci"]][m]) for m in cell["metals"])
            read = {m for p in cell["seqs"]["DOPT_Q"][:3] for m in p}
            assert not any(a in read and b in read for a, b in cell["query"])


def _deep_equal(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    if hasattr(a, "__dict__") and not isinstance(a, type):   # fitted model objects compare by identity: compare state
        return type(a) is type(b) and _deep_equal(vars(a), vars(b))
    return bool(a == b)


@pytest.mark.parametrize("cid", ["R02", "R14", "R23"])
def test_candidate_fit_ignores_all_non_training_labels(bench, rec, cid):
    """Brief section 13 item 4: poisoning EVERY non-training cell (outer test and any cell outside the outer
    training fold) leaves the whole fitted chain of the frozen configurations (candidate R02, secondary R14 and
    R23: HBK fit, POL_ERR table, corrector, lambda) unchanged; the poisoned bench replaces bench.Y too, so
    closures over bench.Y are seen.  All 50 folds: scripts/r3_poison_chain.py."""
    from r3 import registry as RG
    tr = np.asarray(rec["outer"]["train_index"])
    Yp = np.where(np.isfinite(bench.Y), 77.0, np.nan)
    Yp[tr] = bench.Y[tr]
    pb = SimpleNamespace(Y=Yp, groups=bench.groups, n_obs=bench.n_obs, basis=bench.basis, frame=bench.frame, coef=bench.coef,
                         frames=bench.frames)
    cfg = RG.get(cid, bench)
    a, b = RG.fit(cfg, bench, rec, bench.Y), RG.fit(cfg, pb, rec, Yp)
    assert a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)


def test_intervention_catches_a_label_dependent_policy(bench, rec):
    """The intervention's read-sequence half can fail: a policy that peeks at the cell's hidden log D (here: its
    sum decides between the DOPT_Q and UNIFORM sequences) must be caught.  The second pass uses policy.fresh()."""
    from r3 import configs as CF, harness as H

    class Peeking:
        name = "PEEK"

        def __init__(self):
            self.memo = {}

        def __call__(self, block, cell):
            key = cell["ci"]
            if key not in self.memo:
                y = np.asarray(block["Y"][cell["ci"]], float)
                self.memo[key] = cell["seqs"]["DOPT_Q" if np.nansum(y) > 0 else "UNIFORM"]
            return self.memo[key]

        def fresh(self):
            return Peeking()
    methods = {"PWL_R": CF.baseline_methods(bench)["PWL_R"]}
    seed, fold = FOLD[1], FOLD[2]
    iv = H.intervention(bench, FOLD[0], methods, {}, policy=Peeking(), ks=(3,), seeds=[seed], fold_filter={(seed, fold)})
    assert not iv["passed"] and "read sequence moved" in iv["reason"]
    ok = H.intervention(bench, FOLD[0], methods, {}, policy="DOPT_Q", ks=(3,), seeds=[seed], fold_filter={(seed, fold)})
    assert ok["passed"]


def test_intervention_catches_a_predict_time_closure_over_bench_y(bench, rec):
    """A method that reads the bench's log D at prediction time (here: the true value of an unread metal) must be
    caught when the scrambled pass rebuilds its methods from a bench copy with the scrambled Y."""
    from r3 import configs as CF, harness as H

    def factory(b):
        pwl = CF.baseline_methods(b)["PWL_R"]

        def peek(block, cell, obs, k):
            v = np.array(pwl(block, cell, obs, k), float)
            hidden = [m for m in cell["metals"] if m not in obs.read]
            if hidden:
                v[hidden[0]] = b.Y[cell["ci"]][hidden[0]]
            return v
        return {"PEEK": peek}
    seed, fold = FOLD[1], FOLD[2]
    iv = H.intervention(bench, FOLD[0], factory(bench), {}, policy="DOPT_Q", ks=(3,), seeds=[seed],
                        fold_filter={(seed, fold)}, methods_factory=factory)
    assert not iv["passed"] or iv["max_abs_change"] > 0
