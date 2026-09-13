"""Tests of ``gen18proc.optimize`` (DESIGN.md section 12.2, row ``test_optimize``; section 9).

Rows covered: two runs of ``lhs_pareto(n=50, seed=18)`` produce byte-identical CSV; ``n_failed``
(and ``n_invalid_spec``) are counted; OOD candidates are present in the ``all`` table / front and
absent from the ``in_domain_only`` front.  Extras: the integer decode rule of section 9.1, the
levels addition and ``from_config`` with a domain (addendum WB4b), the feed / scrub-return offset
shift rule, ``pareto_rank`` on a known matrix, ``epsilon_knees``, and the GP-BO gate.

Everything runs on the synthetic cation-exchange fixture of ``testsystems`` with 1-3 stages per
section (cheap cascades).  Run:
    .venv/Scripts/python.exe -m pytest generations/gen18_process/tests/test_optimize.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import optimize as OPT  # noqa: E402
from gen18proc import testsystems as ts  # noqa: E402
from gen18proc.metrics import Prices  # noqa: E402
from gen18proc.types import OOD_FLAGS, ApplicabilityDomain, AqStream, CascadeSpec  # noqa: E402

CE = ts.two_metal_cation_exchange()
PRICES = Prices.load()
LIMITS = {"purity_min": 0.9, "recovery_min": 0.5}


def base_spec() -> CascadeSpec:
    scrub = AqStream(0.3, {"Pr": 0.0, "Nd": 0.0}, 0.2, 0.2, 0.0, 0.0)
    strip = AqStream(0.5, {"Pr": 0.0, "Nd": 0.0}, 3.0, 3.0, 0.0, 0.0)
    return CascadeSpec(2, 1, 1, ts.feed_prnd(), scrub, strip, 1.0, {ts.CE_LIGAND: ts.CE_LT_DIMER},
                       0.0, target="Nd")


def small_space(**widen) -> OPT.DesignSpace:
    """Stage counts 1-3, offsets 0-2, the ratios and acids of the config, no bleed / dilution /
    complexant; ``widen`` replaces bounds."""
    return OPT.DesignSpace.from_config(
        family="acidic_organophosphorus", ligand=ts.CE_LIGAND,
        widen={"n_ext": (1, 3), "n_scr": (1, 3), "n_str": (1, 3), "feed_stage_offset": (0, 2),
               "scrub_return_offset": (0, 2), **widen},
        fixed={"strip_anion_M": 0.0, "scrub_complexant_M": 0.0, "feed_dilution": 0.0,
               "f_bleed": 0.0, "scrub_target_mM": 0.0})


# ---------------------------------------------------------------------------------------------
# 9.1  design space
# ---------------------------------------------------------------------------------------------

def test_integer_decode_rule_and_levels():
    space = OPT.DesignSpace(bounds={"n_ext": (1, 4), "oa_ext": (0.5, 2.5)}, integer=("n_ext",),
                            fixed={"n_scr": 2.0},
                            levels={"saponification_degree": (0.0, 0.3, 0.5)})
    assert space.variables == ("n_ext", "oa_ext", "saponification_degree")
    # floor(u (hi - lo + 1)) + lo, clipped
    for u, expect in ((0.0, 1), (0.2499, 1), (0.25, 2), (0.5, 3), (0.9999, 4), (1.0, 4)):
        assert space.decode([u, 0.0, 0.0])["n_ext"] == expect
    assert space.decode([0.0, 0.5, 0.0])["oa_ext"] == pytest.approx(1.5)
    assert space.decode([0.0, 0.0, 0.0])["saponification_degree"] == 0.0
    assert space.decode([0.0, 0.0, 0.34])["saponification_degree"] == 0.3
    assert space.decode([0.0, 0.0, 0.99])["saponification_degree"] == 0.5
    assert space.decode([0.0, 0.0, 0.0])["n_scr"] == 2.0
    with pytest.raises(ValueError):
        OPT.DesignSpace(bounds={"not_a_variable": (0.0, 1.0)})
    with pytest.raises(ValueError):
        space.decode([0.5, 0.5])


def test_from_config_uses_domain_and_family():
    dom = ApplicabilityDomain(anion="chloride", diluent_family="aliphatic_hydrocarbon",
                              modifiers=(), log_acid=(-1.42, -1.02),
                              log_ligand={"PC88A": (-0.699, 0.0)}, log_metal_total_mM=None,
                              loading_fraction=None, log_complexant=None, oa_ratio=None,
                              temperature_C=None, saponification_degree=None,
                              hull_vertices=None, n_records=0, n_publications=0)
    space = OPT.DesignSpace.from_config(family="acidic_organophosphorus", domain=dom,
                                        ligand="PC88A")
    assert space.bounds["scrub_acid_M"] == pytest.approx((10 ** -1.42, 10 ** -1.02))
    assert space.bounds["strip_acid_M"] == pytest.approx((10 ** -1.42, 10 ** -1.02))
    assert space.bounds["ligand_total_M"] == pytest.approx((10 ** -0.699, 1.0))
    assert space.bounds["n_ext"] == (1.0, 30.0) and "n_ext" in space.integer
    widened = OPT.DesignSpace.from_config(family="acidic_organophosphorus", domain=dom,
                                          widen={"strip_acid_M": (0.5, 6.0)})
    assert widened.bounds["strip_acid_M"] == (0.5, 6.0)
    dga = OPT.DesignSpace.from_config(family="diglycolamide")
    assert dga.bounds["saponification_degree"] == (0.0, 0.0)
    assert dga.bounds["ligand_total_M"] == (0.05, 0.4)


def test_offset_shift_rule_keeps_every_stage_wet():
    spec, _ = OPT.candidate_spec({"n_ext": 6, "n_scr": 2, "n_str": 1, "feed_stage_offset": 3,
                                  "scrub_return_offset": 1}, base_spec(), CE, "Nd")
    assert spec.scrub_return_stage == 5 and spec.feed_stage == 3     # shifted by min = 1
    spec, _ = OPT.candidate_spec({"n_ext": 6, "n_scr": 0, "n_str": 1, "feed_stage_offset": 3,
                                  "scrub_return_offset": 0}, base_spec(), CE, "Nd")
    assert spec.feed_stage == 5                                        # no scrub: feed on top
    vals = {"n_ext": 4, "n_scr": 1, "n_str": 1, "feed_stage_offset": 2, "scrub_return_offset": 2}
    spec, _ = OPT.candidate_spec(vals, base_spec(), CE, "Nd")
    assert spec.feed_stage == 3 and spec.scrub_return_stage == 3
    assert vals["feed_stage_offset"] == 0.0 and vals["scrub_return_offset"] == 0.0


def test_candidate_spec_maps_variables():
    vals = {"n_ext": 3, "n_scr": 2, "n_str": 1, "oa_ext": 1.5, "s_over_a": 0.4, "w_over_a": 0.6,
            "scrub_acid_M": 0.3, "scrub_target_mM": 20.0, "strip_acid_M": 4.0,
            "strip_anion_M": 1.0, "ligand_total_M": 1.0, "saponification_degree": 0.3,
            "feed_dilution": 1.0, "f_bleed": 0.05}
    spec, extras = OPT.candidate_spec(vals, base_spec(), CE, "Nd")
    assert spec.feed.flow_L_h == pytest.approx(2.0)          # diluted 1:1
    assert spec.feed.metals["Nd"] == pytest.approx(0.0375)
    assert spec.organic_flow_L_h == pytest.approx(3.0)       # oa_ext x diluted feed flow
    assert spec.scrub.flow_L_h == pytest.approx(0.8) and spec.strip.flow_L_h == pytest.approx(1.2)
    assert spec.scrub.metals["Nd"] == pytest.approx(0.020)
    assert spec.scrub.anion == pytest.approx(0.3 + 3 * 0.020)
    assert spec.strip.h == 4.0 and spec.strip.anion == pytest.approx(5.0)
    assert spec.ligand_total[ts.CE_LIGAND] == pytest.approx(0.5)   # formal 1.0 M -> dimer 0.5
    assert spec.saponification_degree == 0.3 and spec.f_bleed == 0.05
    assert spec.fresh_organic is not None and spec.fresh_organic.ligand_total[ts.CE_LIGAND] == 0.5
    assert extras["feed_dilution_L_h"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# 9.2  lhs_pareto
# ---------------------------------------------------------------------------------------------

def test_lhs_pareto_byte_identical_and_counts():
    space = small_space()
    kw = dict(n=50, seed=18)
    a = OPT.lhs_pareto(space, base_spec(), CE, "Nd", ["Pr"], LIMITS, PRICES, **kw)
    b = OPT.lhs_pareto(space, base_spec(), CE, "Nd", ["Pr"], LIMITS, PRICES, **kw)
    assert a.to_csv(index=False) == b.to_csv(index=False)
    assert len(a) == 50
    assert a.attrs["n_failed"] + a.attrs["n_invalid_spec"] + a.attrs["n_converged"] == 50
    assert set(a["status"]).issubset({"converged_newton", "converged_ss", "failed",
                                      "invalid_spec"})
    assert a.attrs["n_converged"] >= 40
    # sorted by front then the variable tuple
    assert (np.diff(a["front"].to_numpy()) >= 0).all()
    assert a["front"].min() == 1
    for col in ("purity_mol", "recovery_from_feed", "consumption_index", "n_stages_total",
                "regime_status", "flags", "on_spec", "in_domain"):
        assert col in a.columns
    conv = a.loc[a["status"].str.startswith("converged")]
    assert conv["balance_rel_max"].max() < 1e-8
    assert conv["n_stages_total"].to_numpy().tolist() == (conv["n_ext"] + conv["n_scr"]
                                                          + conv["n_str"]).tolist()


def test_failed_and_invalid_rows_are_counted_not_raised():
    # a solver budget of one Newton iteration and one sweep leaves some rows unconverged:
    # they are rows with status ``failed``, NaN metrics and INADMISSIBLE, and they are counted
    space = small_space()
    df = OPT.lhs_pareto(space, base_spec(), CE, "Nd", ["Pr"], LIMITS, PRICES, n=8, seed=18,
                        solver_kwargs={"max_newton": 1, "max_sweeps": 1})
    failed = df["status"] == "failed"
    assert failed.any()
    assert df.attrs["n_failed"] == int(failed.sum())
    assert df.loc[failed, "purity_mol"].isna().all()
    assert (df.loc[failed, "regime_status"] == "INADMISSIBLE").all()
    assert not df.loc[failed, "in_domain"].any()
    assert df.loc[failed, "front"].min() > df.loc[~failed, "front"].max()   # ranked last
    # a zero organic flow is a malformed specification -> invalid_spec, counted
    zero = space.with_bounds(oa_ext=(0.0, 0.0))
    df2 = OPT.lhs_pareto(zero, base_spec(), CE, "Nd", ["Pr"], LIMITS, PRICES, n=4, seed=18)
    assert df2.attrs["n_invalid_spec"] == 4
    assert (df2["status"] == "invalid_spec").all()
    assert df2["reason"].str.contains("organic flow").all()
    assert df2["n_ext"].notna().all()          # the variables survive on invalid rows
    assert df2["front"].min() >= 1


def test_ood_candidates_in_all_but_not_in_domain_front():
    # strip acid up to 30 M: candidates above the fixture's acid box (log acid <= 1) are OOD
    space = small_space(strip_acid_M=(0.5, 30.0))
    df = OPT.lhs_pareto(space, base_spec(), CE, "Nd", ["Pr"], LIMITS, PRICES, n=50, seed=18)
    ood_names = {f.value for f in OOD_FLAGS}
    is_ood = df["flags"].map(lambda s: bool(set(str(s).split("|")) & ood_names))
    assert is_ood.any(), "the widened strip acid must produce OOD candidates"
    assert (~is_ood).any()
    # an OOD row is OUT_OF_DOMAIN unless an inadmissible flag takes precedence (section 1.4)
    assert df.loc[is_ood, "regime_status"].isin({"OUT_OF_DOMAIN", "INADMISSIBLE"}).all()
    assert not df.loc[is_ood, "in_domain"].any()
    fronts = OPT.pareto_fronts(df)
    assert len(fronts["all"]) >= 1 and len(fronts["in_domain_only"]) >= 1
    assert fronts["in_domain_only"]["in_domain"].all()
    assert not set(fronts["in_domain_only"]["candidate"]) & set(df.loc[is_ood, "candidate"])
    assert set(df.loc[is_ood, "candidate"]) <= set(df["candidate"])      # present in ``all``
    assert (df.loc[is_ood, "front_in_domain"] == 0).all()


def test_pareto_rank_known_matrix():
    F = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0], [3.0, 0.5], [np.nan, 1.0], [1.5, 1.5]])
    ranks = OPT.pareto_rank(F)
    assert ranks.tolist() == [1, 3, 1, 1, 0, 2]


def test_epsilon_knees_and_consumption_index():
    df = pd.DataFrame({
        "status": ["converged_newton"] * 4 + ["failed"],
        "in_domain": [True, False, True, True, False],
        "purity_mol": [0.99, 0.99, 0.96, 0.90, np.nan],
        "recovery_from_feed": [0.9, 0.95, 0.9, 0.9, np.nan],
        "consumption_index": [5.0, 1.0, 2.0, 0.5, np.nan],
        "n_stages_total": [10, 12, 8, 4, 6],
    })
    knees = OPT.epsilon_knees(df, [0.95, 0.99], [0.85])
    k_all_99 = knees.loc[(knees.subset == "all") & (knees.purity_min == 0.99)].iloc[0]
    assert k_all_99["reachable"] and k_all_99["n_feasible"] == 2
    assert k_all_99["consumption_index"] == 1.0                    # the OOD row wins in ``all``
    k_dom_99 = knees.loc[(knees.subset == "in_domain_only") & (knees.purity_min == 0.99)].iloc[0]
    assert k_dom_99["consumption_index"] == 5.0                    # only the in-domain row
    k_dom_95 = knees.loc[(knees.subset == "in_domain_only") & (knees.purity_min == 0.95)].iloc[0]
    assert k_dom_95["n_feasible"] == 2 and k_dom_95["consumption_index"] == 2.0
    ci, incomplete = OPT.consumption_index({"acid_mol_per_kg_oxide": 1.0,
                                            "base_mol_per_kg_oxide": 2.0,
                                            "complexant_mol_per_kg_oxide": math.nan})
    assert ci == 3.0 and incomplete
    ci, incomplete = OPT.consumption_index({})
    assert math.isnan(ci) and incomplete


# ---------------------------------------------------------------------------------------------
# 9.3  GP-BO gate
# ---------------------------------------------------------------------------------------------

def test_gp_bo_is_gated_on_reliability():
    space = small_space()
    res = OPT.gp_bo(space, base_spec(), CE, "Nd", ["Pr"], LIMITS, PRICES, n_init=4, n_iter=1,
                    batch=2, reliability={"interpretable": {"n": True, "p_eff": False}})
    assert res.status == "gated" and res.table.empty
    ok, reason = OPT.reliability_passes(CE, {"interpretable": {"n": True, "p_eff": True}})
    assert not ok and "literature" in reason        # fixture params are not a fitted set
    res = OPT.gp_bo(space, base_spec(), CE, "Nd", ["Pr"], LIMITS, PRICES, n_init=4, n_iter=1,
                    batch=2, seed=18, force=True)
    assert res.status == "done" and len(res.table) == 6
    assert set(res.table["source"]) == {"init", "bo"}
    assert "front" in res.table.columns and res.table["front"].min() == 1
