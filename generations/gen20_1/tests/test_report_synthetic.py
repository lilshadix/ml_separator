"""S10 end to end on synthetic long tables (injected loader, status, manifests and selection records; small
bootstrap replicate count): verdicts, thresholds, the exploratory family with BH, result files and the markdown.
No real prediction or target is read."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from gen201 import report as RP
from gen201 import rules as R

import g201_synth as S

DMD = (0, 1)


def build_world(skill: float, designs=RP.DESIGNS, n_chem: int = 14, seed: int = 0):
    cells = S.make_cells(n_chem, seed=seed)
    world = {}
    for d in designs:
        world[(d, "primary")] = S.make_tables(cells, design=d, skill=skill)
        world[(d, "nodmd")] = S.make_tables(cells, design=d, variant="nodmd", skill=skill, drop_cells=DMD)
    man = S.manifests_for(*world[(designs[0], "primary")], dmd_cells=DMD)
    sels = {}
    for d in designs:
        for v in ("primary", "nodmd"):
            t = world[(d, v)][0]
            sels[(d, v)] = {(int(s), int(f)): {"A": {"chosen": "K2", "inner_scores": {"K1": 0.3, "K2": 0.2, "K3": 0.25,
                                                                                  "K4": 0.26}},
                                               "BL_D_INNER": {"chosen": "BD_METAL"}, "BL_SF_INNER": "FLAT"}
                            for s, f in t[["split_seed", "fold"]].drop_duplicates().itertuples(index=False)}
    return world, man, sels


def scorer(world, man, sels, label="synthetic", designs=RP.DESIGNS, replicates=200):
    return RP.Scorer(label, loader=lambda lab, d, v: world[(d, v)],
                     status=lambda lab, d, v: {"status": "complete" if (d, v) in world else "not run"},
                     manifests=man, selections=lambda lab, d, v: sels.get((d, v), {}), designs=designs,
                     replicates=replicates, log=lambda *a: None)


@pytest.fixture(scope="module")
def strong():
    world, man, sels = build_world(skill=1.0)
    sc = scorer(world, man, sels)
    return sc, sc.score()


def test_strong_signal_passes_and_sentences(strong):
    sc, st = strong
    reg = st["registered"]
    assert reg["H1"]["verdict"] == "PASS" and reg["H2"]["verdict"] == "PASS"
    assert reg["H3"]["verdict"] in R.VERDICT_ORDER and reg["H4"]["verdict"] in R.H4_ORDER
    assert [p["id"] for p in st["sentences"]["pat"]][0] == "PAT-2"
    assert st["sentences"]["easier"]["sentence"] == R.EASIER_NEITHER       # H1 and H2 both PASS
    for h in ("H1", "H2"):
        assert R.FEATURE_LABEL in reg[h]["annotations"] and R.CONFIRMATION_PENDING in reg[h]["annotations"]
    # every comparator's (h) used the five other designs
    for ev in reg["H1"]["evidence"]:
        assert set(ev["design_points"]) == set(R.OTHER_DESIGNS) and all(v is not None for v in ev["design_points"].values())
    # the composed EX-11 references exist and the family has 83 tests with BH q
    ex = st["exploratory"]
    assert len(ex) == 83 and all("q_bh" in r for r in ex)
    ex11 = [r for r in ex if r["ex"] == "EX-11"]
    assert all(r["status"] == "run" for r in ex11)
    rep = [r for r in ex if r["population"].startswith("COH-REPCAP")]
    assert len(rep) == 11 and all(r["status"] == "not run" and r["p_used_in_bh"] == 1.0 for r in rep)
    # thresholds carry classes for every arm on the same rows
    th = st["thresholds"]
    assert {th[k]["classes"]["0.4"]["class"] for k in ("T1", "T1w", "T2", "T3")} <= set(R.CLASS_ORDER) | {"NA"}
    assert "level_attributable" in th["T1"]["classes"]["0.4"]
    assert any(k.startswith("COH-D-C3|BD_LEVEL|") for k in th["_others"])


def test_results_files_and_markdown(strong, tmp_path):
    sc, st = strong
    paths = RP.write_results(st, tmp_path / "res")
    names = {p.name for p in paths}
    for n in ("tab05_verdicts.csv", "tab06_thresholds.csv", "tab12_exploratory.csv", "tab03_logd.csv",
              "tab04_logsf.csv", "tab07_cross_target.csv", "tab08_designs.csv", "tab09_secondary.csv",
              "tab10_types.csv", "tab11_folds.csv", "tab13_not_selectable.csv", "tab14_context.csv",
              "report_state.json", "verdicts.json"):
        assert n in names, n
    v = pd.read_csv(tmp_path / "res" / "tab05_verdicts.csv")
    assert len(v) == 4 + 3 + 3 + 2 and {"cond_a", "cond_k", "p_wcr", "sign_A"} <= set(v.columns)
    tab03 = pd.read_csv(tmp_path / "res" / "tab03_logd.csv")
    assert {"R2_pooled_reason", "R2_pooled_n_seeds_defined", "card_y_sd0"} <= set(tab03.columns)
    state = json.loads((tmp_path / "res" / "report_state.json").read_text(encoding="utf-8"))
    md = RP.render(state)
    lines = [ln for ln in md.splitlines() if ln.startswith("- ")]
    assert lines[0].startswith("- " + R.EASIER_NEITHER) and "PAT-2" in lines[1]
    # (h) design points survive the JSON round trip for the confirmation
    hp = RP.h_points_from_json(state)
    assert set(hp[("H1", "BD_MEAN")]) == set(R.OTHER_DESIGNS) and "_nodmd" in hp
    # the confirmation render shows the confirmation status
    md2 = RP.render(state, confirmation=state)
    assert "confirmed (" + R.CONFIRMATION_LABEL in md2


def test_designs_not_run_give_insufficient_compute_or_failed_h():
    world, man, sels = build_world(skill=1.0, designs=("BP", "B"))
    sc = scorer(world, man, sels, designs=RP.DESIGNS)
    reg = sc.registered()
    assert reg["H1"]["verdict"] == "INSUFFICIENT_COMPUTE"
    assert set(reg["H1"]["designs_not_run"]) == {"BR", "BQ", "BPG", "A"}
    for ev in reg["H1"]["evidence"]:
        assert ev["design_points"]["B"] is not None and ev["design_points"]["A"] is None


def test_weak_signal_is_not_pass():
    world, man, sels = build_world(skill=0.0, designs=("BP",), seed=5)
    sc = scorer(world, man, sels, designs=("BP",))
    reg = sc.registered()
    assert reg["H1"]["verdict"] != "PASS"
    assert R.easier_sentence({h: reg[h]["label"] for h in reg})["sentence"] in (R.EASIER_NEITHER, R.EASIER_SF, R.EASIER_D)


def test_c_not_run_gives_na():
    world, man, sels = build_world(skill=0.5, designs=("BP",), n_chem=8)
    for k in list(world):
        logd, logsf = world[k]
        world[k] = (logd, logsf.drop(columns=["C"]))
    sc = scorer(world, man, sels, designs=("BP",))
    reg = sc.registered()
    assert reg["H3"]["verdict"] == "NA" and reg["H4"]["verdict"] == "NA"


def test_bp_not_complete_gives_no_verdict():
    world, man, sels = build_world(skill=0.5, designs=("B",), n_chem=6)
    sc = scorer(world, man, sels)
    st = sc.score()
    assert st["status"].startswith("design BP not complete") and st["registered"]["H1"]["verdict"] == "NA"
    assert "design BP not complete" in RP.render(st)
