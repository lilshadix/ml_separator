"""LT-23: RULE-T / RULE-C / RULE-H4 boundaries (values equal to τ or δ, NA draws, NA conditions) and the verdict
order; NC1; BH family size; the pre-written sentences.  Hand-built evidence dicts (no data)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gen201 import inference as I
from gen201 import rules as R

NAN = float("nan")


def tev(point=0.45, low=0.42, bca=0.42, pub_low=0.42, seeds=(0.45, 0.45, 0.45, 0.45, 0.45), bal=0.45,
        high=0.5, pub_high=0.5):
    return {"point": point, "per_seed": list(seeds), "chem_pct95_low": low, "chem_bca95_low": bca,
            "pub_pct95_low": pub_low, "r2_bal": bal, "chem_pct95_high": high, "pub_pct95_high": pub_high}


def test_rule_t_values_equal_to_tau_reach():
    tau = 0.4
    r = R.rule_t(tev(point=tau, low=tau, bca=tau, pub_low=tau, seeds=(tau, tau, tau, tau, 0.1), bal=tau), tau)
    assert r["class"] == "REACHED" and all(r["conditions"].values())


@pytest.mark.parametrize("field,value", [("low", 0.3999999999), ("bca", NAN), ("pub_low", NAN), ("bal", 0.39),
                                         ("seeds", (0.45, 0.45, 0.45, 0.1, 0.1))])
def test_rule_t_point_only(field, value):
    r = R.rule_t(tev(**{field: value}), 0.4)
    assert r["class"] == "POINT_ONLY"


def test_rule_t_ruled_out_not_reached_na_and_overlap():
    assert R.rule_t(tev(point=0.3, high=0.39, pub_high=0.399), 0.4)["class"] == "RULED_OUT"
    assert R.rule_t(tev(point=0.3, high=0.4, pub_high=0.39), 0.4)["class"] == "NOT_REACHED"   # 97.5th = τ: not < τ
    assert R.rule_t(tev(point=0.3, high=NAN, pub_high=0.39), 0.4)["class"] == "NOT_REACHED"   # NA (degenerate)
    assert R.rule_t(tev(point=NAN), 0.4)["class"] == "NA"
    both = R.rule_t(tev(point=0.41, low=0.1, high=0.39, pub_high=0.38), 0.4)
    assert both["class"] == "POINT_ONLY" and both["also_ruled_out"] and both["ruled_out"]
    assert R.rule_t(tev(point=0.399999), 0.4)["class"] in ("NOT_REACHED", "RULED_OUT")


def test_level_attributable_clauses():
    assert R.level_attributable("POINT_ONLY", "REACHED", 0.2, 0.1)["level_attributable"]
    assert not R.level_attributable("REACHED", "POINT_ONLY", 0.2, 0.1)["level_attributable"]
    la = R.level_attributable("REACHED", "NA", 0.2, 0.1)
    assert la["class_clause"] == R.NOT_EVALUABLE and not la["level_attributable"]
    assert R.level_attributable("REACHED", "NA", 0.0, NAN)["level_attributable"]               # R2_within ≤ 0
    assert R.level_attributable("REACHED", "RULED_OUT", 0.1, 0.1)["level_attributable"]        # ≤ BD_LEVEL_CURVE
    # POINT_ONLY + RULED_OUT overlap is ranked as POINT_ONLY (the printed class)
    assert R.level_attributable("POINT_ONLY", "POINT_ONLY", 0.3, 0.1)["level_attributable"]


def cev(**kw):
    ev = {"target": "logsf", "X": "B", "Y": "FLAT", "delta": 0.03, "met01_Y": 0.17,
          "delta_seeds": [0.03] * 5, "chem_pct9875_low": 0.01, "chem_pct9875_high": 0.05, "chem_bca9875_low": 0.01,
          "p_wcr": 0.001, "wcr_mean": 0.03, "pub_pct9875_low": 0.01, "pub_pct9875_high": 0.05,
          "loco_all_positive": True, "design_points": {d: 0.02 for d in R.OTHER_DESIGNS}, "delta_mae_bal": 0.01,
          "delta_r2_bal": 0.06, "delta_r2_bal_chem_low": 0.01, "delta_r2_bal_pub_low": 0.01, "r2_bal_X": 0.1,
          "skill_flat_X": 0.1, "mde80_dec": 0.01, "mde80_cr": 0.01}
    ev.update(kw)
    return ev


def test_rule_c_boundaries():
    assert R.delta_margin(0.17) == 0.02 and R.delta_margin(0.2) == pytest.approx(0.02) and R.delta_margin(0.5) == 0.05
    assert R.rule_c(cev(delta=0.02))["conditions"]["a"]                          # Δ = δ_Y satisfies (a)
    assert not R.rule_c(cev(delta=0.0199999999))["conditions"]["a"]
    assert R.rule_c(cev(delta=0.05, met01_Y=0.5))["conditions"]["a"]             # δ = 0.10·MET-01(Y) = 0.05
    assert not R.rule_c(cev(chem_pct9875_low=0.0))["conditions"]["b"]            # lower bound must be > 0
    assert not R.rule_c(cev(chem_pct9875_low=-np.inf))["conditions"]["b"]        # NA draws as −∞
    assert not R.rule_c(cev(chem_bca9875_low=NAN))["conditions"]["c"]            # NA fails
    assert not R.rule_c(cev(p_wcr=0.0125))["conditions"]["d"]                    # p must be < 0.0125
    assert not R.rule_c(cev(wcr_mean=-0.01))["conditions"]["d"]
    assert not R.rule_c(cev(pub_pct9875_low=NAN))["conditions"]["e"]
    assert R.rule_c(cev(delta_seeds=[0.03, 0.03, 0.03, 0.03, -0.01]))["conditions"]["f"]
    assert not R.rule_c(cev(delta_seeds=[0.03, 0.03, 0.03, 0.0, -0.01]))["conditions"]["f"]
    assert not R.rule_c(cev(loco_all_positive=False))["conditions"]["g"]
    assert R.rule_c(cev(design_points={**{d: 0.02 for d in R.OTHER_DESIGNS}, "A": None}))["conditions"]["h"] is None
    assert not R.rule_c(cev(design_points={**{d: 0.02 for d in R.OTHER_DESIGNS}, "BQ": 0.0}))["conditions"]["h"]
    assert not R.rule_c(cev(design_points={**{d: 0.02 for d in R.OTHER_DESIGNS}, "BQ": NAN}))["conditions"]["h"]
    assert not R.rule_c(cev(delta_mae_bal=0.0))["conditions"]["i"]
    assert R.rule_c(cev(delta_r2_bal=0.05))["conditions"]["j"]                   # ΔR2_balanced = 0.05 holds
    assert not R.rule_c(cev(r2_bal_X=0.0))["conditions"]["j"]
    assert not R.rule_c(cev(skill_flat_X=0.0))["conditions"]["k"]
    d = R.rule_c(cev(target="logd", Y="BD_MEAN"))
    assert d["conditions"]["j"] is None and d["conditions"]["k"] is None
    bb = R.rule_c(cev(chem_pct9875_high=-0.001, p_wcr=0.01, wcr_mean=-0.02))
    assert bb["baseline_better"]
    assert not R.rule_c(cev(chem_pct9875_high=0.0, p_wcr=0.01, wcr_mean=-0.02))["baseline_better"]


def test_verdict_order():
    ok = [R.rule_c(cev(Y=y)) for y in R.SF_COMPARATORS]
    assert R.verdict_h(ok)["verdict"] == "PASS"
    assert R.verdict_h(ok, candidate_run=False)["verdict"] == "NA"
    # BASELINE_BETTER outranks a passing component
    bb = ok[:2] + [R.rule_c(cev(Y="G14_ZERO", chem_pct9875_high=-0.01, p_wcr=0.001, wcr_mean=-0.03))]
    assert R.verdict_h(bb)["verdict"] == "BASELINE_BETTER"
    # a design not run with (a)-(g) holding everywhere -> INSUFFICIENT_COMPUTE
    miss = {**{d: 0.02 for d in R.OTHER_DESIGNS}, "A": None}
    ic = [R.rule_c(cev(Y=y, design_points=miss)) for y in R.SF_COMPARATORS]
    assert R.verdict_h(ic)["verdict"] == "INSUFFICIENT_COMPUTE"
    # a design not run while some (a)-(g) fails -> (h) counts as failed, order continues to INCONCLUSIVE
    ic2 = ic[:2] + [R.rule_c(cev(Y="G14_ZERO", design_points=miss, delta=0.01))]
    v = R.verdict_h(ic2)
    assert v["verdict"] == "INCONCLUSIVE" and "ENCOURAGING" not in v["sublabels"]
    # (a)-(h) hold but (i) fails -> ESTIMAND_INCONSISTENT
    ei = ok[:2] + [R.rule_c(cev(Y="G14_ZERO", delta_mae_bal=-0.001))]
    assert R.verdict_h(ei)["verdict"] == "ESTIMAND_INCONSISTENT"
    # every point Δ ≥ δ but (b) fails somewhere, with a large MDE -> INCONCLUSIVE (ENCOURAGING; UNDERPOWERED)
    enc = ok[:2] + [R.rule_c(cev(Y="G14_ZERO", chem_pct9875_low=-0.01, mde80_dec=0.05))]
    v = R.verdict_h(enc)
    assert v["label"] == "INCONCLUSIVE (ENCOURAGING; UNDERPOWERED)"
    # MDE exactly δ_Y is not > δ_Y
    enc2 = ok[:2] + [R.rule_c(cev(Y="G14_ZERO", chem_pct9875_low=-0.01, mde80_dec=0.02, mde80_cr=NAN))]
    assert R.verdict_h(enc2)["label"] == "INCONCLUSIVE (ENCOURAGING)"
    # PASS with a large MDE carries the low-power annotation
    lp = [R.rule_c(cev(Y=y, mde80_cr=0.03)) for y in R.SF_COMPARATORS]
    v = R.verdict_h(lp)
    assert v["verdict"] == "PASS" and R.LOW_POWER in R.annotate(v, hid="H2")
    assert R.FEATURE_LABEL in R.annotate(v, hid="H2") and R.NC1_FLAG in R.annotate(v, hid="H2", nc1_flag=True)
    assert any(R.NODMD_FLIP in a for a in R.annotate(v, hid="H2", nodmd_label="INCONCLUSIVE"))


def test_rule_h4_order_and_equivalence():
    assert R.rule_h4(None, None, c_run=False)["verdict"] == "NA"
    cb = cev(X="C", Y="B", met01_Y=0.16)
    bc = cev(X="B", Y="C", delta=-0.03, delta_seeds=[-0.03] * 5, chem_pct9875_low=-0.05, chem_pct9875_high=-0.01,
             chem_bca9875_low=-0.05, wcr_mean=-0.03, pub_pct9875_low=-0.05, loco_all_positive=False,
             design_points={d: -0.02 for d in R.OTHER_DESIGNS}, delta_mae_bal=-0.01, delta_r2_bal=-0.06)
    assert R.rule_h4(cb, bc)["verdict"] == "C_BETTER"
    assert R.rule_h4(bc | {"X": "C", "Y": "B"}, cb | {"X": "B", "Y": "C"})["verdict"] == "B_BETTER"
    miss = {**{d: 0.02 for d in R.OTHER_DESIGNS}, "BPG": None}
    assert R.rule_h4(cev(X="C", Y="B", design_points=miss), bc)["verdict"] == "INSUFFICIENT_COMPUTE"
    eq = cev(X="C", Y="B", delta=0.001, chem_pct9875_low=-0.019, chem_pct9875_high=0.019, pub_pct9875_low=-0.0199,
             pub_pct9875_high=0.0199, p_wcr=0.5)
    eq_b = cev(X="B", Y="C", delta=-0.001, chem_pct9875_low=-0.019, chem_pct9875_high=0.019, p_wcr=0.5)
    assert R.rule_h4(eq, eq_b)["verdict"] == "EQUIVALENT"
    edge = dict(eq, chem_pct9875_low=-0.02)                          # an endpoint at −0.02 is not inside
    v = R.rule_h4(edge, eq_b)
    assert v["verdict"] == "NOT_ESTABLISHED" and v["label"] == "NOT_ESTABLISHED"
    v = R.rule_h4(dict(edge, mde80_cr=0.021), eq_b)
    assert v["label"] == "NOT_ESTABLISHED (UNDERPOWERED)"
    assert R.rule_h4(dict(eq, pub_pct9875_high=NAN), eq_b)["verdict"] == "NOT_ESTABLISHED"


def test_nc1_flag():
    a = cev(target="logd", X="A", Y="A_PRIME", delta=0.0001)          # (a) not required by NC1
    b = cev(target="logd", X="A_PRIME", Y="A", chem_pct9875_low=-0.01)
    res = R.nc1(a, b)
    assert res["flag"] and res["A_over_A_PRIME"]["meets_b_to_g"] and not res["A_PRIME_over_A"]["meets_b_to_g"]
    assert not R.nc1(dict(a, p_wcr=0.2), b)["flag"]


def test_na_draws_unfavourable_in_contrast_bounds():
    d = np.full(10_000, 0.05)
    d[:70] = np.nan                                                    # 0.7 % NA draws: kept as −∞ / +∞
    b = I.interval_block(d, 0.05, None, "chem")
    assert b["chem_pct9875_low"] == -np.inf and b["chem_pct9875_high"] == np.inf and not b["chem_degenerate"]
    assert not R.rule_c(cev(chem_pct9875_low=b["chem_pct9875_low"]))["conditions"]["b"]
    assert not R.rule_c(cev(chem_pct9875_high=b["chem_pct9875_high"], p_wcr=0.001, wcr_mean=-0.1))["baseline_better"]
    d[:101] = np.nan
    b = I.interval_block(d, 0.05, None, "chem")
    assert b["chem_degenerate"] and math.isnan(b["chem_pct9875_low"])


def test_bh_family_size_and_na():
    fam = R.exploratory_family()
    assert len(fam) == 83 and len({f["id"] for f in fam}) == 83
    counts = {}
    for f in fam:
        counts[f["ex"]] = counts.get(f["ex"], 0) + 1
    assert counts == {"EX-01": 1, "EX-02": 1, "EX-03": 1, "EX-04": 1, "EX-05": 1, "EX-06": 3, "EX-07": 1, "EX-08": 1,
                      "EX-09": 40, "EX-10": 22, "EX-11": 3, "EX-12": 2, "EX-13": 2, "EX-14": 4}
    p = [0.001] + [NAN] * 82
    out = R.bh(p)
    assert out["p_used"][1] == 1.0 and out["q"][0] == pytest.approx(0.083) and not out["reject"][0]
    out = R.bh([0.0001] * 83)
    assert out["reject"].all()
    with pytest.raises(AssertionError):
        R.bh([0.01] * 82)


def test_sentences_verbatim():
    lab = {"H1": "PASS", "H2": "INCONCLUSIVE (ENCOURAGING)", "H3": "BASELINE_BETTER", "H4": "NOT_ESTABLISHED"}
    e = R.easier_sentence(lab)
    assert e["sentence"] == R.EASIER_D and "H2: INCONCLUSIVE (ENCOURAGING)" in e["text"] and "G14_ZERO" in e["text"]
    pats = R.pat_sentences(lab)
    assert [p["id"] for p in pats] == ["PAT-1"]
    assert pats[0]["text"].startswith("Variant A was shown to beat every pre-named log D baseline")
    assert "(H2: INCONCLUSIVE (ENCOURAGING), H3: BASELINE_BETTER)" in pats[0]["text"]
    none = {"H1": "INCONCLUSIVE", "H2": "INCONCLUSIVE", "H3": "NA", "H4": "NA"}
    assert R.easier_sentence(none)["sentence"] == R.EASIER_NEITHER
    p5 = R.pat_sentences(none)
    assert [p["id"] for p in p5] == ["PAT-5"] and "<" not in p5[0]["text"]
    assert [p["id"] for p in R.pat_sentences({"H1": "NA", "H2": "PASS", "H3": "PASS"})] == ["PAT-3"]
    assert R.easier_sentence({"H1": "NA", "H2": "PASS", "H3": "PASS"})["sentence"] == R.EASIER_SF
    assert [p["id"] for p in R.pat_sentences({"H1": "PASS", "H2": "PASS", "H3": "NA"})] == ["PAT-2"]
    assert [p["id"] for p in R.pat_sentences({"H1": "INCONCLUSIVE", "H2": "INCONCLUSIVE", "H3": "PASS"})] == ["PAT-4"]
    t = R.threshold_sentence(0.4, {"T1": "REACHED", "T1w": "POINT_ONLY", "T2": "RULED_OUT", "T3": "NOT_REACHED"},
                             "level-attributable")
    assert t["id"] == "PAT-6" and "(T1 REACHED; T1w: POINT_ONLY; level-attributable)" in t["text"]
    assert "(T2: RULED_OUT, T3: NOT_REACHED)" in t["text"]
    assert R.threshold_sentence(0.4, {"T1": "RULED_OUT", "T1w": "NA", "T2": "REACHED", "T3": "NA"}, "")["id"] \
        == "PATTERN-NOT-ANTICIPATED"
    assert R.threshold_sentence(0.5, {"T1": "RULED_OUT", "T1w": "NA", "T2": "NOT_REACHED", "T3": "NA"}, "") is None
    assert R.h4_sentence({"verdict": "EQUIVALENT", "label": "EQUIVALENT"})["text"].endswith(
        "the two routes are equivalent within ±0.02 log10.")


def test_confirmation_logic():
    assert R.confirm_h("H1", {"verdict": "PASS"}, {"verdict": "PASS"})["confirmed"]
    assert not R.confirm_h("H1", {"verdict": "PASS"}, {"verdict": "INCONCLUSIVE"})["confirmed"]
    assert R.confirm_h("H4", {"verdict": "EQUIVALENT"}, {"verdict": "EQUIVALENT"})["confirmed"]
    assert not R.confirm_h("H2", {"verdict": "PASS"}, None)["confirmed"]
    assert R.confirm_h("H2", {"verdict": "PASS"}, None)["status"] == R.NOT_CONFIRMED_LABEL
    assert R.confirm_t("REACHED", "REACHED")["final"] == "REACHED"
    assert R.confirm_t("RULED_OUT", "NOT_REACHED")["final"] is None
    assert R.confirm_t("POINT_ONLY", "POINT_ONLY")["final"] is None
    assert R.prnd_status({"FLAT": 0.02, "G14_ZERO": 0.01}, {"FLAT": 0.0}) == "verified"
    assert R.prnd_status({"FLAT": 0.03}, {"FLAT": 0.021}) == "failed"
    assert R.prnd_status({"FLAT": 0.03}, {"FLAT": 0.01}) == "not verified"
