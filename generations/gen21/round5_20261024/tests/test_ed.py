"""Tests for r5/ed.py (item 1.6, E-D): synthetic data and construction-only checks -- no score is computed.

* the set-enumeration decomposition P0 + beta * DD equals r4.refs.predict_levels (the exact B1 path);
* P3's 5th-read policies, HULL and the recipe-2 ladder: legal, nested, deterministic tie-breaks;
* legal_sets equals the sets reachable by sequential gen20k.design.legal_metals reads;
* LOCO static selection never reads the held-out cluster's labels;
* the statistics on squared-error columns agree with r5.common (R2, unit statistics), block deltas add up;
* the intervention check passes an honest predictor and catches a block-Y peek;
* frames: identical row keys across rungs and policies, the C3 min(k, k_max) rule, LADDER_GUARD;
* labels, RNG seed scheme and the frozen item specification;
* (construction-only, real metadata) HULL / LADDER sequences are legal on every C3 cell.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ed.py
        -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402,F401
from r5 import common as CM  # noqa: E402
from r5 import ed as ED  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k.data import LANTHANIDES, N_LN, CellView  # noqa: E402

IDX = ED.IDX


def design_for(metals, cid="t"):
    cv = CellView(index=0, cell_id=cid, extractant="e", chemotype="c", publication="p", metals=tuple(sorted(metals)))
    return DS.make_design(cv, "adjacent", "metal")


def reachable_sets(dz, r):
    """Every read set of size r reachable by sequential legal reads (BFS over gen20k.design.legal_metals)."""
    level = {frozenset()}
    for _ in range(r):
        level = {s | {m} for s in level for m in DS.legal_metals(dz, s)}
    return {tuple(sorted(s)) for s in level}


METAL_SETS = [tuple(range(N_LN)),
              tuple(IDX[m] for m in ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb")),
              tuple(IDX[m] for m in ("La", "Nd", "Sm", "Eu", "Gd", "Dy", "Er", "Lu")),
              tuple(IDX[m] for m in ("Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy"))]


# ------------------------------------------------------------------------------------------------------------
# levels
# ------------------------------------------------------------------------------------------------------------
def test_sf_parts_equals_predict_levels():
    rng = np.random.default_rng(1)
    worst = 0.0
    for metals in METAL_SETS:
        dz = design_for(metals)
        qa = np.array([a for a, b in dz.query])
        qb = np.array([b for a, b in dz.query])
        for r in (2, 3, 4, 5):
            for S in ED.legal_sets(dz, r)[:25]:
                y = np.full(N_LN, np.nan)
                y[list(metals)] = rng.normal(0, 1.5, len(metals))
                P0, DD = ED.sf_parts(y, metals, S, qa, qb)
                for beta in (0.0, -0.072, -0.13, 0.4):
                    L = ED.levels(y, metals, S, beta)
                    worst = max(worst, float(np.abs((L[qa] - L[qb]) - (P0 + beta * DD)).max()))
    assert worst <= 1e-12


def test_levels_keep_reads_and_ignore_unread():
    rng = np.random.default_rng(2)
    metals = METAL_SETS[0]
    y = rng.normal(size=N_LN)
    read = frozenset(IDX[m] for m in ("La", "Gd", "Er", "Lu"))
    L = ED.levels(y, metals, read, -0.09)
    assert np.allclose(L[list(read)], y[list(read)], atol=0, rtol=0)
    y2 = ED.scramble_unread(y, read, rng)
    assert not np.allclose(y2, y)
    assert np.array_equal(ED.levels(y2, metals, read, -0.09), L)          # honest predictor passes the intervention
    peek = lambda row: np.asarray(row, float).copy()                        # noqa: E731  block-Y peek
    assert not np.array_equal(peek(y)[list(metals)], peek(y2)[list(metals)])   # ... and the peek is caught


# ------------------------------------------------------------------------------------------------------------
# policies
# ------------------------------------------------------------------------------------------------------------
def test_static_candidates_and_legal_sets():
    assert len(ED.static_candidates(2)) == 91 - 12                         # 12 strict-adjacent pairs among 14 metals
    for metals in METAL_SETS:
        dz = design_for(metals)
        for r in (2, 3, 4):
            assert set(ED.legal_sets(dz, r)) == reachable_sets(dz, r)


def test_hull_and_ladder_sequences_are_legal_nested():
    for metals in METAL_SETS:
        dz = design_for(metals)
        n = min(dz.k_max, 6) + 1
        for seq in (ED.hull_sequence(dz, n), ED.ladder_sequence(dz, n)):
            assert len(seq) == n and len(set(seq)) == n
            ED.assert_legal_chain(dz, [seq[:r] for r in range(2, n + 1)])
    dz = design_for(METAL_SETS[0])
    h = ED.hull_sequence(dz, 5)
    assert [LANTHANIDES[m] for m in h[:2]] == ["La", "Lu"]
    lad = ED.ladder_sequence(dz, 5)
    assert [LANTHANIDES[m] for m in lad] == list(ED.STATIC_ORDER)           # 14-metal cell: the static order exactly
    dz2 = design_for(METAL_SETS[3])                                          # no La, no Ho, no Lu: fallbacks, still legal
    lad2 = ED.ladder_sequence(dz2, min(dz2.k_max, 6) + 1)
    assert LANTHANIDES[lad2[0]] == "Nd"


def test_env_policies_pick_legal_metals():
    for metals in METAL_SETS[:3]:
        dz = design_for(metals)
        if dz.k_max < 4:
            continue
        read4 = frozenset(ED.hull_sequence(dz, 4))
        legal = sorted(DS.legal_metals(dz, read4), key=ED._z)
        loose = ED.loose_metals(metals, read4, dz.query)
        assert set(legal) <= set(loose)
        picks = ED.env_picks(metals, read4, legal, loose, legal[0])
        for p, m in picks.items():
            assert m in (loose if p.endswith("LOOSE") else legal)
            assert not any(a in read4 | {m} and b in read4 | {m} for a, b in dz.query)


def test_maxmin_tie_break_lower_z():
    x = np.arange(N_LN, dtype=float)
    assert ED.maxmin([5], [3, 7], x) == 3                                    # equal distance -> lower Z


def test_read_density_helpers():
    metals = METAL_SETS[0]
    read = {IDX["La"], IDX["Gd"], IDX["Lu"]}
    assert ED.longest_hidden_run(metals, read) == 6                          # Tb .. Yb hidden between Gd and Lu
    assert ED.out_share(metals, {IDX["Ce"], IDX["Yb"]}) == pytest.approx(2 / 12)
    assert ED.outside_hull(IDX["La"], IDX["Ce"], {IDX["Ce"], IDX["Yb"]})
    assert not ED.outside_hull(IDX["Gd"], IDX["Tb"], {IDX["Ce"], IDX["Yb"]})


# ------------------------------------------------------------------------------------------------------------
# LOCO static selection
# ------------------------------------------------------------------------------------------------------------
def fake_bench(n_per=3, clusters=("g:a", "g:b", "g:c"), seed=0):
    rng = np.random.default_rng(seed)
    rows, Y, cmap = [], [], {}
    for c in clusters:
        for i in range(n_per):
            metals = METAL_SETS[(i + len(cmap)) % 3]
            y = np.full(N_LN, np.nan)
            y[list(metals)] = np.cumsum(rng.normal(0.2, 0.3, len(metals)))
            cid = f"{c[-1]}{i}"
            rows.append({"cell_id": cid, "extractant": "e" + cid, "chemotype": "sc" + c[-1], "publication_id": "pub_" + c[-1],
                         "chem_family": "diglycolamide", "extractant_name": "TODGA"})
            Y.append(y)
            cmap[cid] = c
    return SimpleNamespace(frame=pd.DataFrame(rows), Y=np.vstack(Y)), cmap


def test_loco_selection_ignores_held_out_labels():
    b, cmap = fake_bench()
    st = ED.static_tables(b, list(range(len(b.frame))), intervention=True)
    sel = ED.loco_static_selection(st, cmap)
    assert set(sel.reads_req) == set(ED.REC_READS) and sel.groupby("reads_req").size().eq(3).all()
    b2 = SimpleNamespace(frame=b.frame, Y=b.Y.copy())
    held = [i for i, c in enumerate(b.frame.cell_id) if cmap[c] == "g:a"]
    rng = np.random.default_rng(9)
    b2.Y[held] = np.where(np.isfinite(b2.Y[held]), rng.normal(0, 3, b2.Y[held].shape), np.nan)
    sel2 = ED.loco_static_selection(ED.static_tables(b2, list(range(len(b.frame))), intervention=False), cmap)
    a1 = sel[sel.cluster_cpg == "g:a"].set.tolist()
    a2 = sel2[sel2.cluster_cpg == "g:a"].set.tolist()
    assert a1 == a2


def test_poison_loco_identical_and_positive_control():
    """s0-mechanics-14: the LOCO static selection for a cluster is unchanged when that cluster's labels are replaced;
    the leaky selection (the cluster's own rows included) changes for at least one cluster."""
    b, cmap = fake_bench()
    cis = list(range(len(b.frame)))
    st = ED.static_tables(b, cis, intervention=False)
    sel = ED.loco_static_selection(st, cmap)
    rep = ED.poison_loco(b, cis, cmap, st, sel)
    assert rep["clusters"] == 3 and not rep["changed"] and rep["pc_caught"] > 0 and rep["passed"]
    bad = sel.copy()
    bad["set_idx"] = "99"
    assert not ED.poison_loco(b, cis, cmap, st, bad)["passed"]


def test_peek_control_leaky_caught_honest_not():
    """s0-mechanics-6: through the real level function, a predictor that reads every measured metal changes when the
    unread metals are scrambled; the honest 4-read predictor does not."""
    rng = np.random.default_rng(3)
    metals = tuple(range(N_LN))
    y = np.cumsum(rng.normal(0.2, 0.3, N_LN))
    rd = frozenset({IDX["La"], IDX["Nd"], IDX["Gd"], IDX["Lu"]})
    b_r = {"B1": -0.1, "B0": 0.0, "HIST_D7": -0.072, "B1_k3beta": -0.1}
    ys = ED.scramble_unread(y, rd, rng)
    H1, H2 = ED._arm_levels(y, metals, rd, b_r), ED._arm_levels(ys, metals, rd, b_r)
    assert all(np.array_equal(H1[a], H2[a], equal_nan=True) for a in H1)
    allm = frozenset(m for m in metals if np.isfinite(y[m]))
    L1, L2 = ED._arm_levels(y, metals, allm, b_r), ED._arm_levels(ys, metals, allm, b_r)
    assert not all(np.array_equal(L1[a][list(metals)], L2[a][list(metals)], equal_nan=True) for a in L1)


def test_static_tables_intervention_counts():
    b, cmap = fake_bench(n_per=1)
    chk = ED.Checks()
    ED.static_tables(b, list(range(len(b.frame))), intervention=True, checks=chk)
    assert chk.c["static_sets_intervention_checked"] > 0 and chk.c["static_sets_intervention_failed"] == 0


# ------------------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------------------
def synth_frame(seed=0, n_cells=26, seeds=(1, 2, 3)):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seeds:
        for c in range(n_cells):
            for j, pt in enumerate(ED.PAIR_ORDER[:6]):
                y = rng.normal(0.2, 0.3)
                p1, p2 = y + rng.normal(0, 0.15), y + rng.normal(0, 0.2)
                rows.append({"split_seed": s, "fold": c % 5, "cell_id": f"c{c:02d}", "A": pt.split("/")[0],
                             "B": pt.split("/")[1], "y": y, "p_X": p1, "p_Z": p2, "cluster_cpg": f"g{c % 13:02d}",
                             "publication": f"pub{c % 7}", "chemotype": f"sc{c % 13}", "k_max": 4, "n_metals": 14,
                             "pair": pt, "block": ED.BLOCK[pt], "extractant": f"e{c}"})
    F = pd.DataFrame(rows)
    for a in ("X", "Z"):
        F["se_" + a] = (F["p_" + a] - F.y) ** 2
        F["ae_" + a] = (F["p_" + a] - F.y).abs()
    return ED._sorted(F)


def test_r2_and_delta_match_common():
    F = synth_frame()
    ST = ED.Stats(sorted(F.cluster_cpg.unique()), reps=200)
    for a in ("X", "Z"):
        want = CM.r2_seeds(F, "p_" + a)
        assert np.allclose(ED.r2_seeds(F, a).to_numpy(), want.to_numpy(), atol=1e-12)
        assert np.allclose(ST.level(F, "se_" + a)["per_seed"].to_numpy(), want.to_numpy(), atol=1e-12)
    P = ED.pair_frame(F, "X", F, "Z")
    d = ST.delta(P, "se_r", "se_c")["per_seed"]
    assert np.allclose(d.to_numpy(), (ED.r2_seeds(F, "Z") - ED.r2_seeds(F, "X")).to_numpy(), atol=1e-12)
    tot = 0.0
    for b in ED.BLOCKS:
        P["_m"] = P.block == b
        tot += ST.delta(P, "se_r", "se_c", mask_col="_m")["per_seed"].to_numpy()
    assert np.allclose(tot, d.to_numpy(), atol=1e-12)
    us = ED.unit_stats_se(P)
    ref = CM.unit_stats(F.assign(X=F.p_X, Z=F.p_Z), "Z", "X", "cluster_cpg")
    assert np.allclose(us.dSSE.sort_index().to_numpy(), ref.dSSE.sort_index().to_numpy(), atol=1e-12)
    zero = ED.contrast_row(ST, F, "X", F, "X", lopo=True)
    assert zero["delta_R2"] == 0 and zero["boot_sd"] == 0 and zero["units_WLT"].endswith("/13")


def test_level_row_fields():
    F = synth_frame()
    ST = ED.Stats(sorted(F.cluster_cpg.unique()), reps=100)
    r = ED.level_row(ST, F, "X", lopo=True)
    for k in ("R2", "per_seed", "seeds_ge_050", "MSE", "MAE", "P_boot_ge_050", "boot_lo", "boot_hi", "loco_min",
              "loco_max", "lopo_min", "lopo_max", "lopo_n_ge_050", "top3_cluster_SSE_share"):
        assert k in r
    assert r["clusters"] == 13 and r["rows_per_seed"] == 26 * 6


# ------------------------------------------------------------------------------------------------------------
# frames
# ------------------------------------------------------------------------------------------------------------
def synth_rows():
    """Two cells: k_max 3 (rungs 2-4) and k_max 4 (rungs 2-5), 2 queries each, one seed; ENV and REC rows."""
    rows = []

    def add(cell, km, fam, pol, rr, nread, cx=False):
        for A, B in (("La", "Ce"), ("Gd", "Tb")):
            rows.append({"split_seed": 1, "fold": 0, "cell_id": cell, "A": A, "B": B, "k_max": km, "family": fam,
                         "policy": pol, "reads_req": rr, "n_read": nread, "cx": cx, "y": 0.1, "se_B1": 0.01})
    for r in (2, 3, 4):
        add("a", 3, "LAD", "DOPTQ", r, r, cx=True)
    for r in (2, 3, 4, 5):
        add("b", 4, "LAD", "DOPTQ", r, r)
    add("b", 4, "ENV", "BISECT_R", 5, 5)
    for pol in ("LADDER", "HULL"):
        for r in ED.REC_READS:
            add("a", 3, "REC", pol, r, min(r, 4), cx=True)
            add("b", 4, "REC", pol, r, r)
    return pd.DataFrame(rows)


def test_frames_identical_rows_and_min_rule():
    G = ED.split_rows(synth_rows())
    k4 = ED.lad_frame(G, "C3pop", 4)
    k5 = ED.lad_frame(G, "C3pop", 5)
    assert ED.same_rows(k4, k5)
    assert set(k5[k5.cell_id == "a"].n_read) == {4} and set(k5[k5.cell_id == "b"].n_read) == {5}
    assert set(ED.lad_frame(G, "C4", 5).cell_id) == {"b"}
    env = ED.env_frame(G, "C3pop", "BISECT_R")
    assert ED.same_rows(env, k4)
    g = ED.rec_frame(G, "LADDER_GUARD", 3)
    assert ED.same_rows(g, ED.rec_frame(G, "LADDER", 3))
    assert set(g[g.cell_id == "a"].policy) == {"HULL"} and set(g[g.cell_id == "b"].policy) == {"LADDER"}
    with pytest.raises(ValueError):
        ED.lad_frame(G, "C4", 6)


# ------------------------------------------------------------------------------------------------------------
# labels, seeds, specification
# ------------------------------------------------------------------------------------------------------------
def test_labels_name_section_9_1():
    for arm, pol, var, d in itertools.product(ED.ARMS, ("DOPTQ", "ORACLE_B1", "RANDOM_LOOSE"), ED.VARIANTS, ("BPG", "BP")):
        lab = ED.label_for(arm, pol, var, d)
        assert any(x in lab for x in CM.LABELS) or "ORACLE" in lab
    assert ED.label_for("B1", "ORACLE_B1").startswith("ORACLE [O]")
    assert "DIAGNOSTIC" in ED.label_for("HIST_D7")
    assert "SENSITIVITY (C5-BTBP out)" in ED.label_for(variant="noC5")


def test_row_labels():
    R = pd.DataFrame({"design": ["BPG", "BP", "BPG"], "policy": ["DOPTQ", "ORACLE_B1", "RANDOM_LOOSE"]})
    lab = ED.row_labels(R).astype(str).tolist()
    assert lab[0].startswith(ED.SEEN) and "SENSITIVITY (design BP)" in lab[1] and lab[1].startswith("ORACLE [O]")
    assert "LOOSE" in lab[2]


def test_rng_seed_scheme():
    assert ED.mc_seed("BPG", "all") == paths.seed_for("ED|RANDOM_MC|BPG|all")
    assert ED.mc_seed("BPG", "all") != ED.mc_seed("BP", "all")


def test_read_budget_tag_and_row_names():
    """ITEM_SPEC labels: 'different read budget' on every rung but 4 reads (contrasts: when either side is not 4);
    the C3 row is named 'C3 min(k, k_max), k = reads - 1' (brief 6.6); no empty label separators."""
    assert ED.budget_tag(4) == "" and ED.budget_tag(5) == ED.RB and ED.budget_tag(2) == ED.RB
    assert ED.budget_tag(4, 5) == ED.RB and ED.budget_tag(4, 4, extra="x") == "x"
    assert ED.budget_tag(7, extra="x") == f"{ED.RB}; x"
    assert ED.join_labels("a", "", "b") == "a; b" and "; ;" not in ED.label_for("B1", "", "all", "BP", ED.budget_tag(4))
    assert [ED.reads_of(n) for n in ("C3pop|r5", "LADDER C4|r4", "RECIPE C3pop|r2|HULL", "ENVELOPE C3pop|r5|DOPTQ")] == [5, 4, 2, 5]
    assert ED.row_name("C3pop|r5") == "C3 min(k, k_max), k = reads - 1 (5 reads)" == f"{ED.C3_ROW} (5 reads)"
    assert ED.row_name("LADDER C4|r4") == "C4 fixed cohort (k_max >= 4), 4 reads"
    assert ED.row_name("RECIPE C3pop|r2|HULL").startswith(ED.C3_ROW)


def test_ea_cell_blanked_in_every_writer_spelling():
    """s0-fidelity-9: the C3 4-read cell (E-A rows) reads '(E-A rows)' and prints no level, under every frame spelling
    the writers use (ladder / duplicates 'C3pop|r4', recipe 'C3pop|r4|DOPTQ', density 'LADDER C3pop|r4' and
    'RECIPE C3pop|r4|DOPTQ'); other 4-read rows (recipe policies, fixed cohorts) and 5-read rows are not E-A."""
    for n in ("C3pop|r4", "C3pop|r4|DOPTQ", "LADDER C3pop|r4", "RECIPE C3pop|r4|DOPTQ"):
        assert ED.is_ea_cell(n), n
    for n in ("C3pop|r4|LADDER", "C3pop|r4|HULL", "C4|r4", "LADDER C4|r4", "C3pop|r5", "ENVELOPE C3pop|r5|DOPTQ",
              "RECIPE C3pop|r4|LOCO_STATIC", "C3pop|r2"):
        assert not ED.is_ea_cell(n), n
    rec = {"frame": "C3pop|r4", "rows_per_seed": 1802.0, "cells": 229, "R2": 0.45, "R2_B0": 0.41, "MSE": 0.03,
           "per_seed": "0.4;0.5", "lopo_min": 0.44, "relSSE_vs_B1_4reads": 0.0, "relSSE_vs_DOPTQ": 0.0,
           "label": ED.SEEN}
    b = ED.ea_blank(rec)
    assert b["level_note"] == ED.EA_NOTE == "(E-A rows)"
    assert all(np.isnan(b[k]) for k in ("R2", "R2_B0", "MSE", "per_seed", "lopo_min", "relSSE_vs_B1_4reads"))
    assert b["rows_per_seed"] == 1802.0 and b["cells"] == 229 and b["label"] == ED.SEEN and b["relSSE_vs_DOPTQ"] == 0.0


def _pareto_row(p, dominates, keeps=None):
    return {"design": "BPG", "frame": "C3pop", "variant": "all", "policy": p, "role": ED.PARETO_ROLE[p],
            "pareto_dominates": dominates, "keeps_every_block_above_floor": dominates if keeps is None else keeps}


def test_pareto_answer_is_over_the_written_family_only():
    """REGISTRY 1.6-R01 asks 'does any member ...': ORACLE_B1 [O] (bound) and LOOSE (sensitivity) are never members;
    each gets its own answer row."""
    rows = [_pareto_row(p, False) for p in ED.ENV_LABEL_FREE if p != "DOPTQ"]
    rows += [_pareto_row("ORACLE_B1", True), _pareto_row("MAXMIN_R_LOOSE", False), _pareto_row("RANDOM_LOOSE", False)]
    next(r for r in rows if r["policy"] == "HULL_BISECT")["keeps_every_block_above_floor"] = True   # keeps, improves nothing
    ans = {a["policy"]: a for a in ED.pareto_answers(rows, "BPG", "C3pop", "all")}
    fam = ans["ANSWER: written label-free family"]
    assert fam["pareto_dominates"] is False and fam["dominating_policies"] == ""
    assert fam["n_policies"] == len(ED.ENV_LABEL_FREE) - 1 and "ORACLE_B1" not in fam["policies"]
    assert "LOOSE" not in fam["policies"] and fam["label"].startswith(ED.SEEN)
    orc = ans["ANSWER: ORACLE_B1 [O] bound"]
    assert orc["pareto_dominates"] is True and orc["label"].startswith("ORACLE [O]") and "never a member" in orc["role"]
    assert ans["ANSWER: LOOSE legality"]["pareto_dominates"] is False
    assert "SENSITIVITY (LOOSE legality)" in ans["ANSWER: LOOSE legality"]["label"]
    rows[0]["pareto_dominates"] = True                         # a member that dominates is reported as one
    fam2 = ED.pareto_answers(rows, "BPG", "C3pop", "all")[0]
    assert fam2["pareto_dominates"] is True and fam2["dominating_policies"] == rows[0]["policy"]
    S1 = _load_s1_ed()
    po = S1.pareto_outcome(pd.DataFrame(rows[1:] + ED.pareto_answers(rows[1:], "BPG", "C3pop", "all")))
    assert po["family_any_dominates"] is False and po["family_dominating"] == []
    assert po["oracle_bound_dominates"] is True and po["loose_sensitivity_dominating"] == []


def test_repro_gate_enforces_every_registered_identity():
    """ITEM_SPEC reproduction targets: 4-read B1 (and B0, HIST_D7) vs REF within 8.9e-16, y exact, read sets
    identical, every row matched; the 5-read B0 equal to R3-B0 PWL_R at k = 4 on every C4 row; NaN fails."""
    ok = {"rows_K3": 9010, "rows_REF": 9010, "rows_matched": 9010, "read_sets_equal_share": 1.0,
          "max_abs_diff": {"y": 0.0, "B0": 0.0, "B1": 0.0, "HIST_D7": 0.0}, "rows_C4_r5": 7580, "rows_R3B0_k4": 7580,
          "rows_matched_r5": 7580, "max_abs_diff_B0_r5_vs_R3B0": 0.0, "read_sets_equal_share_r5": 1.0}
    assert all(ED.repro_gate(ok).values())
    bad = [("max_abs_diff_B0_r5_vs_R3B0", 1e-9), ("max_abs_diff_B0_r5_vs_R3B0", float("nan")), ("rows_matched_r5", 7579),
           ("read_sets_equal_share_r5", 0.99), ("rows_matched", 9009), ("read_sets_equal_share", 0.5)]
    for k, v in bad:
        assert not all(ED.repro_gate({**ok, k: v}).values()), k
    for arm in ("B0", "HIST_D7", "B1", "y"):
        assert not all(ED.repro_gate({**ok, "max_abs_diff": {**ok["max_abs_diff"], arm: 1e-6}}).values()), arm


def _load_s1_ed():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ed_s1", Path(__file__).resolve().parents[1] / "scripts" / "s1_ed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compute_line_and_next_step_are_computed(tmp_path, monkeypatch):
    """The per-item table's compute line sums the item's sealed passes under the current seal from RUN_RECORD.md (no
    stale text), and the next registered step is read from REGISTRY.csv."""
    S1 = _load_s1_ed()
    rr = tmp_path / "RUN_RECORD.md"
    rr.write_text("\n".join([
        "- 2026-10-24T11:09:44+00:00 START ED repro seal=abc",
        "- 2026-10-24T11:13:45+00:00 END ED seal=abc post-run check ok (ok) repro status=0",
        "- 2026-10-24T11:14:10+00:00 START ED run seal=abc",
        "- 2026-10-24T11:29:18+00:00 END ED seal=abc post-run check ok (ok) run status=3",
        "- 2026-10-24T11:29:44+00:00 START ED_POISON shard 0/1 seal=abc",
        "- 2026-10-24T11:50:07+00:00 END ED_POISON seal=abc post-run check ok (ok) shard 0/1 status=0",
        "- 2026-10-24T12:00:00+00:00 START ED run seal=old",
        "- 2026-10-24T12:10:00+00:00 END ED seal=old post-run check ok (ok) run status=3", ""]), encoding="utf-8")
    monkeypatch.setattr(CM, "RUN_RECORD", rr)
    ps = S1.pass_seconds("abc")
    assert ps == {"repro": 241.0, "run": 908.0, "poison": [1223.0]}
    line = S1.compute_line({"BPG": {"seconds": 95.0}, "BP": {"seconds": 94.0}}, 190.0, "abc", "from-rows")
    assert "reproduction (TECHNICAL SMOKE) 241 s" in line and "1223 s" in line and "total 2562 s" in line
    assert "no completed pass" in S1.compute_line({"BPG": {"seconds": 1.0}}, 1.0, "new", "from-rows")
    nxt = S1.next_registered("1.6")
    assert nxt.startswith("Registry item 1.7") and "if not yet run" not in nxt
    assert S1._designs({"BPG": True, "BP": True}) == "BPG and BP" and S1._designs({"BPG": False, "BP": False}) == "neither design"


def test_item_spec_lists_the_frozen_arms():
    txt = ED.item_spec_text()
    for name in ED.ENV_POLICIES + ED.REC_POLICIES + ED.ARMS:
        assert name in txt
    for s in ("8675309", "RANDOM_MC", "TECHNICAL SMOKE", "CONSISTENCY CHECK", "s1_ed_poison.py", "0.4968", "0.4821"):
        assert s in txt
    assert "\r" not in txt


# ------------------------------------------------------------------------------------------------------------
# construction-only on real metadata (metal sets; no label compared)
# ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench():
    from gen20k import data as D
    return D.load_bench()


def test_sequences_legal_on_every_c3_cell(bench):
    from gen20k.data import cell_view
    ref = CM.load_rows("BPG")
    cid2ci = {c: i for i, c in enumerate(bench.frame.cell_id.astype(str))}
    cells = sorted(set(ref.cell_id.astype(str)))
    assert len(cells) == 229
    for c in cells:
        dz = DS.make_design(cell_view(bench, cid2ci[c]), "adjacent", "metal")
        n = min(dz.k_max, 6) + 1
        for seq in (ED.hull_sequence(dz, min(n, 5)), ED.ladder_sequence(dz, min(n, 5))):
            ED.assert_legal_chain(dz, [seq[:r] for r in range(2, len(seq) + 1)])
            assert len(seq) == min(n, 5)
