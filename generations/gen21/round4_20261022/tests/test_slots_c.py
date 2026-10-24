"""Tests for stage-1 block C (r4.an_data, r4.slots_c): union exclusions, control construction, identical Ln rows across
arms, support exactness, determinism and fit poisoning (with a shortened training schedule).  None compares a held-out
prediction with its target.

    .venv/Scripts/python.exe -m pytest generations/gen21/round4_20261022/tests -q -p no:cacheprovider -W ignore
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r4 import an_data as AD  # noqa: E402
from r4 import paths  # noqa: E402
from r4 import refs as RF  # noqa: E402
from r4 import slots_c as SC  # noqa: E402
from r4.leakage import deep_equal, poisoned_bench  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN  # noqa: E402
from gen20k.splits import copy_group_map, folds_for  # noqa: E402

FOLD = ("BPG", 130363, 2)
FORCED = (0.3, 0.3, 1.0)                  # (w_Ln, WD, w_X): a positive network, so the refit really trains


@pytest.fixture(scope="module")
def bench():
    return D.load_bench()


@pytest.fixture(scope="module")
def rec():
    return C.load(*FOLD)


def _short(monkeypatch):
    monkeypatch.setattr(SC, "EPOCHS", 15)
    monkeypatch.setattr(SC, "SEEDS", (0,))
    monkeypatch.setattr(SC, "NET_GRID", ((0.0, 0.3), (1.0, 0.3)))
    monkeypatch.setattr(SC, "W_AN_GRID", (0.0, 1.0))
    SC._S11_CACHE.clear()


def test_an_records_are_an3_compatible():
    an = AD.an3_records()
    assert set(an.metal_symbol) <= set(AD.AN3) and an.log_D.notna().all() and len(an) > 1000
    assert not an.solvent_key.astype(str).str.contains(r"\[", regex=True).any()
    assert an.rid.is_unique and an.extractant.notna().all()
    sys.path.insert(0, str(paths.ROOT / "generations" / "gen19_chem_transfer"))
    from gen19ct.data.load import load_archive
    a = load_archive().set_index("source_record_id").loc[an.source_record_id]
    hb = pd.to_numeric(a.holdback_concentration_M, errors="coerce")
    hb_smi = a.holdback_smiles_canonical.astype("string")
    assert not ((hb_smi.notna() & (hb_smi.str.len() > 0)) & ~(hb == 0)).fillna(False).any()   # no active hold-back
    assert not a.modifier_name.astype("string").str.contains(AD.ACIDIC_MODIFIERS, case=False, regex=True).fillna(False).any()
    assert not a.has_suspect_flag.fillna(False).astype(bool).any()
    assert (a.g19_tier == "MODEL").all() and (a.n_organic_extractants == 1).all() and not a.in_value_conflict.any()


def test_links_rebuilt_from_scratch(bench):
    """The chemotype link map against an independent rebuild (fresh generator, the frozen bench frame)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    f = bench.frame
    coh = [(gen.GetFingerprint(Chem.MolFromSmiles(str(e))), str(c)) for e, c in zip(f.extractant, f.chemotype)]
    links = AD.an_links()
    for e in sorted(set(AD.an3_records().extractant)):
        fp = gen.GetFingerprint(Chem.MolFromSmiles(e))
        sims = DataStructs.BulkTanimotoSimilarity(fp, [x for x, _ in coh])
        assert links[e] == frozenset(c for (_, c), s in zip(coh, sims) if s >= 0.7), e


def test_union_exclusions_on_every_fold_and_inner_block(bench):
    """All 50 dev fold-fits and every inner block; the four clauses recomputed independently of excluded_by."""
    f = bench.frame
    cg = copy_group_map()
    links = AD.an_links()

    def clean(an, held):
        pubs = {str(f.publication_id.iat[i]) for i in held}
        assert not (set(an.publication) & pubs)
        assert not (set(an.copy_group) & {cg.get(p, "own:" + p) for p in pubs})
        assert not (set(an.extractant) & {AD._canon(f.extractant.iat[i]) for i in held})
        chems = {str(f.chemotype.iat[i]) for i in held}
        assert not any(links[e] & chems for e in an.extractant)

    n_inner = 0
    for d in ("BP", "BPG"):
        for fo in folds_for(f, d, seeds=list(paths.DEV_SEEDS)):
            rec = C.load(d, fo.seed, fo.fold)
            an = AD.fold_an(bench, rec)
            train = set(map(int, rec["outer"]["train_index"]))
            held = [i for i in range(len(f)) if i not in train]
            clean(an, held)
            for blk in rec["inner"]:
                itr = set(map(int, blk["train_index"]))
                an_j = AD.inner_an(bench, rec, blk, an)
                assert set(an_j.rid) <= set(an.rid)
                clean(an_j, held + [i for i in train if i not in itr])
                n_inner += 1
    assert n_inner >= 190


def test_control_arms(bench, rec):
    an = AD.fold_an(bench, rec)
    tr = {int(c) for c in rec["outer"]["train_index"]}
    base = SC.aux_data("S11", bench, tr, an)
    for arm in ("S13", "S14", "S15", "S16", "S24"):
        a = SC.aux_data(arm, bench, tr, an)
        for x, y in zip(a["ln"], base["ln"]):                          # Ln aux rows identical in every arm
            assert np.array_equal(np.asarray(x), np.asarray(y)) if not isinstance(x, list) else x == y
    s15 = SC.aux_data("S15", bench, tr, an)["priv"]
    assert len(s15[1]) == len(an)                                      # matched row count
    assert np.isclose(s15[2].sum(), 1.0)
    n_units = an.publication.nunique()
    size = pd.Series(np.round(1.0 / (s15[2] * n_units)).astype(int))          # row weight = 1 / (units * unit size)
    units = sorted(int(k) for k, c in size.value_counts().items() for _ in range(c // k))
    assert units == sorted(an.groupby("publication").size().tolist())         # same per-unit sizes and weights
    assert set(s15[3]) <= set(bench.frame.extractant.iloc[sorted(tr)])       # training-Ln rows only
    assert len(SC.PERM_SEEDS) == 5
    for seed in SC.PERM_SEEDS:
        perm = SC._permuted_an(an, seed)
        assert not np.array_equal(perm.log_D.to_numpy(), an.log_D.to_numpy())
        assert (perm.metal_symbol == an.metal_symbol).all()
        vals = {(e, m): set(g.log_D) for (e, m), g in an.groupby(["extractant", "metal_symbol"])}
        own = sum(v in vals[(e, m)] for e, m, v in zip(an.extractant, an.metal_symbol, perm.log_D))
        assert own < 0.5 * len(an)                                     # targets come from other blocks
    cyc = SC.metal_cycled(an)
    for _, g in an.assign(c=cyc).groupby(list(SC.SERIES), dropna=False):
        assert sorted(g.metal_symbol) == sorted(g.c)                   # the series' metal multiset is kept
        if g.metal_symbol.nunique() > 1:
            assert (g.metal_symbol != g.c).all()
        else:
            assert (g.metal_symbol == g.c).all()
    s14 = SC.aux_data("S14", bench, tr, an)
    assert s14["priv"] is None and len(s14["an_in_ln"][1]) == len(an)
    assert SC._ln_with_x("S14", 0.0) == SC.S14_MIN_W_LN and SC._ln_with_x("S13", 0.0) is None


def test_deep_equal_compares_torch_values():
    a, b, a2 = SC._Net(0, 16), SC._Net(1, 16), SC._Net(0, 16)
    assert not deep_equal({"nets": [a]}, {"nets": [b]})
    assert deep_equal({"nets": [a]}, {"nets": [a2]})
    import torch
    assert not deep_equal(torch.zeros(3), torch.ones(3))


def test_descriptor_scale_floor_and_clip(bench):
    f = bench.frame
    cells = [i for i in range(len(f)) if SC._z_raw(str(f.extractant.iat[i]))[SC.SB.RD_DESC.index("NumHDonors")] == 0]
    assert cells
    zstat, _ = SC._stats([{"X": np.zeros((1, 30))}], bench, set(cells))
    assert np.all(zstat[1] > 0) and np.all(np.isfinite(zstat[1]))
    Z = SC._zmat(sorted(AD.an3_records().extractant.unique()), zstat)
    assert np.all(np.abs(Z) <= SC.Z_CLIP)


def test_zero_aux_weight_reproduces_s11(bench, rec, monkeypatch):
    _short(monkeypatch)
    an = AD.fold_an(bench, rec)
    tr = {int(c) for c in rec["outer"]["train_index"]}
    b1 = RF.fit_reference(bench, rec, bench.Y, "B1")
    sits = SC.SB._rows(bench, rec, bench.Y, {"geometry": None}, b1)
    zstat, rstat = SC._stats(sits, bench, tr | {s["ci"] for s in sits})
    main = SC._main_rows(sits, bench)
    X = np.vstack([s["X"][:, :-SC.N_CHEM] for s in sits[:5]])
    ext = [bench.frame.extractant.iat[s["ci"]] for s in sits[:5] for _ in range(len(s["t"]))]
    ref = SC.predict_corr(SC.train(main, SC.aux_data("S11", bench, tr, an), 0.3, 0.0, 0.3, zstat, rstat), X, ext, zstat, rstat)
    for arm in ("S13", "S14", "S15", "S16"):
        nets = SC.train(main, SC.aux_data(arm, bench, tr, an), 0.3, 0.0, 0.3, zstat, rstat,
                        ln_weight_with_x=SC._ln_with_x(arm, 0.3))
        assert np.array_equal(ref, SC.predict_corr(nets, X, ext, zstat, rstat))


def _refit(bench, rec, Y, arm="S13"):
    b1 = RF.fit_reference(bench, rec, Y, "B1")
    return SC._refit(bench, rec, Y, arm, b1, 0, FORCED, {}, AD.fold_an(bench, rec), {"geometry": None}, FORCED[:2])


def _an_poisoned(bench, rec):
    an = AD.an3_records()
    keep = set(AD.fold_an(bench, rec).rid)
    rng = np.random.default_rng(7)
    drop = ~an.rid.isin(keep).to_numpy()
    return an.assign(log_D=np.where(drop, rng.normal(0, 3, len(an)), an.log_D.to_numpy()))


def test_refit_poison_invariant_and_leak_detected(bench, rec, monkeypatch):
    """The outer refit (nets compared by value) is bit-identical under Ln poisoning of non-training cells plus An
    poisoning of the fold-excluded records; a refit that ignores the exclusions is caught (positive control)."""
    _short(monkeypatch)
    a = _refit(bench, rec, bench.Y)
    pb, _ = poisoned_bench(bench, rec["outer"]["train_index"], seed=4)
    an_p = _an_poisoned(bench, rec)
    real = AD.an3_records
    monkeypatch.setattr(AD, "an3_records", lambda: an_p)
    b = _refit(pb, rec, pb.Y)
    assert a["nets"] is not None and deep_equal(a, b)
    monkeypatch.setattr(AD, "fold_an", lambda bench_, rec_: AD.an3_records())             # deliberate leak
    leaky_p = _refit(pb, rec, pb.Y)
    monkeypatch.setattr(AD, "an3_records", real)
    leaky = _refit(bench, rec, bench.Y)
    assert not deep_equal(leaky, leaky_p)


def test_fit_deterministic_poison_invariant_and_support_exact(bench, rec, monkeypatch):
    _short(monkeypatch)
    b1 = RF.fit_reference(bench, rec, bench.Y, "B1")
    a = SC.fit_slot(bench, rec, bench.Y, "S13", b1)
    pb, _ = poisoned_bench(bench, rec["outer"]["train_index"], seed=4)
    b1p = RF.fit_reference(pb, rec, pb.Y, "B1")
    b = SC.fit_slot(pb, rec, pb.Y, "S13", b1p)
    assert deep_equal(a, b)
    fa = _refit(bench, rec, bench.Y)
    cell = next(c for c in rec["outer"]["cells"] if c["k_max"] >= 3)
    ym = np.full(N_LN, np.nan)
    for p in cell["seqs"]["DOPT_Q"][:3]:
        for m in p:
            ym[m] = 0.07 * m
    lv = SC.predict_levels(fa, SimpleNamespace(y_masked=ym), cell)
    for m in range(N_LN):
        if np.isfinite(ym[m]):
            assert lv[m] == ym[m]
    zero = dict(fa, nets=None)
    assert np.array_equal(SC.predict_levels(zero, SimpleNamespace(y_masked=ym), cell),
                          RF.predict_levels(fa["b1"], SimpleNamespace(y_masked=ym), cell["metals"]), equal_nan=True)
