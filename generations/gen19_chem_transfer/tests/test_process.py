"""Phase H process integration (brief section 27: "process Monte Carlo reproducibility", "Gen18 regression tests",
"mass-balance invariance"; pre-registration sections 10 F5, 14): the Gen19 -> gen18 adapter, the Monte Carlo, the
lexicographic ranking, S2(d) and the runner's gate.  Synthetic prediction tables and gen18's own test systems only
(``gen18proc.testsystems``); gen18 is imported read-only; nothing reads the corpus, a fold file or a discovery record.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

G19 = Path(__file__).resolve().parents[1]
if str(G19) not in sys.path:
    sys.path.insert(0, str(G19))

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import support as SUP  # noqa: E402
from gen19ct.manifest import write_csv, write_json  # noqa: E402
from gen19ct.process import gen18_adapter as GA  # noqa: E402
from gen19ct.process import monte_carlo as MC  # noqa: E402
from gen19ct.process import robust_optimize as RO  # noqa: E402

paths.add_gen18_to_path()

from gen18proc import optimize as OPT  # noqa: E402
from gen18proc import testsystems as ts  # noqa: E402
from gen18proc.cascade import check_balances, solve_cascade  # noqa: E402
from gen18proc.dmodel import ASSUMPTIONS, LN10, SystemModel, tracer_state  # noqa: E402
from gen18proc.metrics import compute_metrics  # noqa: E402
from gen18proc.types import ASSUMED_LABEL, ModelBuildError, ProvStatus  # noqa: E402

REL_FD = 1e-6
REL_EXACT = 1e-10


def _runner():
    name = "g19_run_process"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, G19 / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# --------------------------------------------------------------------------------------------- #
# synthetic tables
# --------------------------------------------------------------------------------------------- #

def table_rows(*, std: float = 0.3, statuses=None, constant: dict[str, float] | None = None, arm: str = "M2",
               system_id: str = "sys_test_solvating", la=None, ll=None) -> list[dict]:
    """log D = a_m + 1.5 log_acid + 2.5 log_ligand + 0.2 log_acid log_ligand (tracer), or ``constant`` per metal."""
    la = np.linspace(-2.0, 0.7, 7) if la is None else np.asarray(la)
    ll = np.linspace(-1.5, -0.5, 5) if ll is None else np.asarray(ll)
    rows = []
    for m, a in (("Pr", 1.0), ("Nd", 1.3)):
        for i, x in enumerate(la):
            for j, y in enumerate(ll):
                mu = constant[m] if constant else a + 1.5 * x + 2.5 * y + 0.2 * x * y
                st = statuses(m, i, j) if statuses else "IN_DOMAIN"
                rows.append({"metal": m, "system_id": system_id, "log_acid": float(x), "log_ligand": float(y),
                             "mean_logD": float(mu), "std_logD": std, "lower_95": mu - 2.0 * std, "upper_95": mu + 2.0 * std,
                             "domain_status": st, "support_score": 0.7, "nearest_support": "Nd|TODGA", "arm": arm})
    return rows


def synthetic_table(**kw) -> GA.PredictionTable:
    return GA.PredictionTable.from_frame(pd.DataFrame(table_rows(**kw)))


def small_space(ligand: str, *, max_ext: int = 4) -> OPT.DesignSpace:
    return OPT.DesignSpace.from_config(
        family="diglycolamide", domain=None, ligand=ligand,
        widen={"n_ext": (1, max_ext), "n_scr": (1, 2), "n_str": (1, 2), "feed_stage_offset": (0, max_ext - 1),
               "scrub_return_offset": (0, max_ext - 1), "scrub_acid_M": (0.05, 4.0), "strip_acid_M": (0.01, 0.3),
               "ligand_total_M": (0.05, 0.3), "oa_ext": (0.5, 2.0), "s_over_a": (0.2, 1.0), "w_over_a": (0.5, 2.0)},
        fixed={"saponification_degree": 0.0, "strip_anion_M": 0.0, "scrub_complexant_M": 0.0, "feed_dilution": 0.0,
               "f_bleed": 0.0, "scrub_target_mM": 0.0})


def setup_for(entry, table: GA.PredictionTable, **kw) -> MC.CaseSetup:
    lig = entry.organic_ligands[0].name
    feed = ts.feed_prnd_nitrate(acid_M=3.0)
    return MC.CaseSetup(entry=entry, feed=feed, feed_anion="nitrate", temperature_C=25.0, target="Nd", impurities=("Pr",),
                        purity_grid=(0.95, 0.97, 0.99), recovery_grid=(0.80, 0.85, 0.90), space=small_space(lig),
                        base_spec=MC.base_spec_for(feed, "Nd", lig, 0.1), prices=None, n_prior=2.5, **kw)


# --------------------------------------------------------------------------------------------- #
# the table and the adapter
# --------------------------------------------------------------------------------------------- #

def test_status_vocabulary_matches_section_13_module():
    assert GA.DOMAIN_STATUSES == tuple(SUP.DOMAIN_STATUS_ORDER)


def test_prediction_table_validation_and_views():
    tab = synthetic_table()
    assert tab.shape == (2, 7, 5) and tab.metals == ("Nd", "Pr") and tab.arm == "M2"
    assert tab.box()["log_acid"] == (-2.0, 0.7) and tab.statuses() == {"IN_DOMAIN": 70}
    rec = tab.record("Nd", 0, 0)
    assert rec["model_derived"] is True and rec["domain_status"] == "IN_DOMAIN" and rec["lower_95"] < rec["mean_logD"]
    rows = table_rows()
    with pytest.raises(ValueError, match="rectangular"):
        GA.PredictionTable.from_frame(pd.DataFrame(rows[:-1]))
    with pytest.raises(ValueError, match="duplicate"):
        GA.PredictionTable.from_frame(pd.DataFrame(rows + rows[:1]))
    with pytest.raises(ValueError, match="lacks columns"):
        GA.PredictionTable.from_frame(pd.DataFrame(rows).drop(columns=["std_logD"]))
    with pytest.raises(ValueError, match="vocabulary"):
        GA.PredictionTable.from_frame(pd.DataFrame(rows).assign(domain_status="MODEL_PREDICTED"))
    sub = tab.restrict(log_acid=(-1.0, 0.7))
    assert sub.shape[1] < tab.shape[1] and sub.box()["log_acid"][0] >= -1.0
    # a round trip through the long frame reproduces the table
    tab2 = GA.PredictionTable.from_frame(tab.to_frame())
    np.testing.assert_allclose(tab2.mean, tab.mean)


def test_unsupported_refuses_unless_allowed_and_supported_box_trims_the_edge():
    def st(m, i, j):
        return "UNSUPPORTED" if i == 0 else ("CONDITION_EXTRAPOLATION" if i == 1 else "IN_DOMAIN")
    tab = synthetic_table(statuses=st)
    assert tab.has_unsupported and tab.worst_status() == "UNSUPPORTED"
    with pytest.raises(GA.UnsupportedPredictionError, match="UNSUPPORTED"):
        GA.Gen19DModel(tab, "TODGA")
    model = GA.Gen19DModel(tab, "TODGA", allow_unsupported=True)          # F5(ii) audit only
    assert model.allow_unsupported
    sb = tab.supported_box()
    assert sb["ok"] and sb["removed_lines"] == {"log_acid": 1, "log_ligand": 0}
    trimmed = tab.supported()
    assert not trimmed.has_unsupported and trimmed.shape == (2, 6, 5)
    assert trimmed.box()["log_acid"][0] > tab.box()["log_acid"][0]
    GA.Gen19DModel(trimmed, "TODGA")                                       # now allowed
    every = synthetic_table(statuses=lambda m, i, j: "UNSUPPORTED" if j == 2 else "IN_DOMAIN")   # a middle column
    sb2 = every.supported_box()
    assert sb2["ok"] and sb2["removed_lines"]["log_ligand"] >= 3           # trims from an edge up to the column


def test_provenance_is_assumed_with_range_and_model_derived():
    tab = synthetic_table()
    model = GA.Gen19DModel(tab, "TODGA", model_id="M2:cfg-7", fit_manifest_sha256="ab" * 32)
    prov = model.provenance
    assert prov.status is ProvStatus.ASSUMED and prov.assumed_label == ASSUMED_LABEL
    assert prov.range == (float(tab.lower.min()), float(tab.upper.max()))
    assert prov.source.kind == "model" and "M2" in prov.source.locator
    assert prov.model_id == "M2:cfg-7" and prov.fit_manifest_sha256 == "ab" * 32
    assert prov.note.startswith("MODEL_DERIVED")
    assert all(r["model_derived"] for r in model.cell_records())


def test_adapter_reproduces_constant_d_when_std_is_zero():
    """A constant table with std 0 and no depletion / stoichiometry equals gen18's ConstantD on the reference 6/3/3."""
    entry = ts.solvating_entry()
    lig = entry.organic_ligands[0].name
    const = {"Pr": 1.0, "Nd": 1.3}
    tab = synthetic_table(std=0.0, constant=const)
    assert tab.meta["n_intervals_repaired"] == 0
    g19 = GA.Gen19DModel(tab, lig, n_prior=0.0, p=0.0, z=0.0)
    sys_g19 = SystemModel(entry=None, dmodels={lig: g19}, complexant=None, activity=None, temperature_C=25.0,
                          assumptions=ASSUMPTIONS, params_source="gen19")
    sys_const = ts.constant_d_system(const)
    feed = ts.feed_prnd_nitrate(acid_M=3.0)
    spec_g19 = MC.base_spec_for(feed, "Nd", lig, 0.1)
    spec_const = MC.base_spec_for(feed, "Nd", ts.CONSTANT_LIGAND, 1.0)
    assert (spec_g19.n_ext, spec_g19.n_scr, spec_g19.n_str) == (6, 3, 3)
    r1, r2 = solve_cascade(spec_g19, sys_g19), solve_cascade(spec_const, sys_const)
    assert r1.status.startswith("converged") and r2.status.startswith("converged")
    m1 = compute_metrics(r1, spec_g19, sys_g19, "Nd", ["Pr"], None)
    m2 = compute_metrics(r2, spec_const, sys_const, "Nd", ["Pr"], None)
    assert rel(m1.purity_mol, m2.purity_mol) < REL_EXACT
    assert rel(m1.recovery_from_feed, m2.recovery_from_feed) < REL_EXACT
    assert max(abs(v) for v in check_balances(spec_g19, r1, sys_g19).values()) < 1e-8
    # the table's draw values also enter: with_values shifts log D and the cascade follows
    g_hi = GA.Gen19DModel(tab.with_values(tab.mean + 1.0), lig, n_prior=0.0, p=0.0, z=0.0)
    ev = g_hi.evaluate("Nd", tracer_state(sys_g19, {lig: 0.1}, 1.0, 1.0))
    assert rel(ev.log_d, 2.3) < REL_EXACT


def central_difference(model, metal, state, var, eps=1e-5):
    lig = model.ligand

    def perturbed(scale):
        f = math.exp(scale)
        if var == "L":
            return replace(state, ligand_free={lig: state.ligand_free[lig] * f})
        if var == "h":
            return replace(state, h=state.h * f)
        if var == "nu":
            return replace(state, anion=state.anion * f)
        return replace(state, c_free=state.c_free * f)

    return (model.evaluate(metal, perturbed(eps)).log_d - model.evaluate(metal, perturbed(-eps)).log_d) / (2.0 * eps)


def test_partials_vs_finite_differences_and_stoichiometry():
    entry = ts.solvating_entry()
    lig = entry.organic_ligands[0]
    tab = synthetic_table()
    system = GA.build_gen19_system(tab, entry, feed_anion="nitrate", temperature_C=25.0, n_prior=2.5)
    assert not isinstance(system, ModelBuildError) and system.params_source == "gen19"
    model = system.dmodels[lig.name]
    # stoichiometry: q = n_prior (ligands per metal), p = 0 (solvating), z from the entry's ligand spec
    assert model.q == {"Nd": 2.5, "Pr": 2.5} and model.p == {"Nd": 0.0, "Pr": 0.0} and model.z["Nd"] > 0
    for h, lt, frac in ((0.5, 0.1, 0.6), (0.02, 0.2, 0.9), (2.0, 0.06, 0.3)):
        st = replace(tracer_state(system, {lig.name: lt}, h, h, 0.0), ligand_free={lig.name: frac * lt})
        for metal in ("Pr", "Nd"):
            ev = model.evaluate(metal, st)
            assert rel(ev.dlogd_dlnL, 2.5 / LN10) < REL_EXACT
            for var, an in (("L", ev.dlogd_dlnL), ("h", ev.dlogd_dlnh), ("nu", ev.dlogd_dlnanion), ("c", ev.dlogd_dlnc)):
                fd = central_difference(model, metal, st, var)
                if an == 0.0:
                    assert abs(fd) < 1e-9, (var, fd)
                else:
                    assert rel(an, fd) < REL_FD, (var, an, fd)
            # the acid partial is the table's own slope: d/dlog10 h of (1.5 x + 0.2 x y) = 1.5 + 0.2 log10 L_T
            assert rel(ev.dlogd_dlnh, (1.5 + 0.2 * math.log10(lt)) / LN10) < 1e-9
    # outside the grid: clamped value, zero acid slope, OUTSIDE_TABLE recorded
    model.reset_usage()
    st = tracer_state(system, {lig.name: 0.1}, 50.0, 50.0, 0.0)
    ev = model.evaluate("Nd", st)
    assert ev.dlogd_dlnh == 0.0 and ev.ood_distance["gen19_table"] > 0.9
    assert GA.OUTSIDE_TABLE in model.statuses_used()
    # cross-anion and a missing feed metal are refused like gen18's build_system_model
    assert isinstance(GA.build_gen19_system(tab, entry, feed_anion="chloride", temperature_C=25.0), ModelBuildError)
    assert isinstance(GA.build_gen19_system(tab, entry, feed_anion="nitrate", temperature_C=25.0, feed_metals=("Nd", "Eu")),
                      ModelBuildError)


def test_usage_tracking_counts_cache_hits_and_extrapolation_cells():
    def st(m, i, j):
        return "CONDITION_EXTRAPOLATION" if i >= 5 else "IN_DOMAIN"
    tab = synthetic_table(statuses=st)
    model = GA.Gen19DModel(tab, "TODGA")
    model.core(0.05, 0.5, 0.5, 0.0, 0.1)                         # log acid -0.3: interior IN_DOMAIN cells
    assert model.statuses_used() == frozenset({"IN_DOMAIN"})
    model.reset_usage()
    model.core(0.05, 0.5, 0.5, 0.0, 0.1)                         # same key: a cache hit must still record usage
    assert model.statuses_used() == frozenset({"IN_DOMAIN"}) and model.n_evaluations == 1
    model.core(0.05, 4.0, 4.0, 0.0, 0.1)                         # log acid 0.6 lies in the extrapolation rows
    assert model.statuses_used() == frozenset({"IN_DOMAIN", "CONDITION_EXTRAPOLATION"})


# --------------------------------------------------------------------------------------------- #
# draws and the Monte Carlo
# --------------------------------------------------------------------------------------------- #

def test_draws_are_truncated_correlated_and_deterministic():
    tab = synthetic_table(std=0.3)
    values, draws = MC.draw_tables(tab, 200, seed=7, rho=0.9)
    for v in values:
        assert np.all(v >= tab.lower - 1e-12) and np.all(v <= tab.upper + 1e-12)
    corr = np.corrcoef(draws["z_Nd"], draws["z_Pr"])[0, 1]
    assert corr > 0.8
    # one common quantile per metal: the offset is monotone in u
    order = np.argsort(draws["u_Nd"].to_numpy())
    off = draws["mean_offset_Nd"].to_numpy()[order]
    assert np.all(np.diff(off) >= -1e-12)
    v2, d2 = MC.draw_tables(tab, 200, seed=7, rho=0.9)
    pd.testing.assert_frame_equal(draws, d2)
    _, d3 = MC.draw_tables(tab, 200, seed=8, rho=0.9)
    assert not np.allclose(draws["z_Nd"], d3["z_Nd"])
    # std 0 -> the mean, exactly
    v0, _ = MC.draw_tables(synthetic_table(std=0.0), 3, seed=1, rho=0.0)
    for v in v0:
        np.testing.assert_array_equal(v, synthetic_table(std=0.0).mean)
    # a missing interval falls back to the untruncated Gaussian and is counted
    rows = table_rows(std=0.3)
    for r in rows:
        r["lower_95"] = np.nan
        r["upper_95"] = np.nan
    tab_nan = GA.PredictionTable.from_frame(pd.DataFrame(rows))
    _, dn = MC.draw_tables(tab_nan, 2, seed=1, rho=0.0)
    assert dn.attrs["n_untruncated_cells_total"] == 2 * 70
    with pytest.raises(ValueError):
        MC.exchangeable_correlation(2, 1.0)


def member_rows(*, std: float = 0.3, q95: float = 2.5, offsets=(-0.2, -0.1, 0.0, 0.1, 0.2), **kw) -> list[dict]:
    """``table_rows`` plus the M7 member means (``mean + offset_k``, averaging to the mean) and ``conformal_q95``."""
    rows = table_rows(std=std, **kw)
    for r in rows:
        for k, d in enumerate(offsets):
            r[f"{GA.MEMBER_COLUMN_PREFIX}{k}"] = r["mean_logD"] + d
        r[GA.CONFORMAL_Q95_COLUMN] = q95
    return rows


def test_registered_draw_is_one_member_plus_a_conformal_residual(tmp_path):
    """Task X finding V-07 (section 14): one ensemble member per draw, shared by Pr and Nd, plus one residual per metal
    at the calibrated conformal scale; the truncated Gaussian is a flagged fallback a registered run refuses."""
    tab = GA.PredictionTable.from_frame(pd.DataFrame(member_rows()))
    assert tab.has_members and tab.n_members == 5 == GA.N_MEMBERS_REGISTERED and tab.meta["has_members"]
    from gen19ct.models import ladder as LAD
    assert GA.N_MEMBERS_REGISTERED == LAD.N_MEMBERS
    values, draws = MC.draw_tables(tab, 300, seed=7, rho=0.9)
    assert draws.attrs["draw_mode"] == MC.DRAW_MODE_REGISTERED == "member_plus_conformal_residual"
    assert set(draws["member"].unique()) == set(range(5)) and draws["member"].value_counts().min() > 30
    scale = tab.std * tab.conformal_q95 / MC.Z_975
    for k in range(5):
        z = np.array([draws.loc[k, f"z_{m}"] for m in tab.metals]).reshape(-1, 1, 1)
        np.testing.assert_allclose(values[k], tab.members[int(draws.loc[k, "member"])] + scale * z)
    # one member for both metals of a draw; the residual is common over the grid and Pr / Nd correlated
    stacked = np.stack(values)                                          # (K, M, A, L)
    resid = stacked - np.stack([tab.members[int(k)] for k in draws["member"]])
    for m_i in range(len(tab.metals)):
        r = resid[:, m_i] / scale[m_i]
        assert np.allclose(r, r[:, :1, :1], atol=1e-9)                  # the same z over the grid
    assert np.corrcoef(draws["z_Nd"], draws["z_Pr"])[0, 1] > 0.8
    # the residual scale: its 95 % half-width is the calibrated conformal one (q95 * std)
    rr = resid[:, 0, 0, 0]
    assert abs(np.std(rr) - scale[0, 0, 0]) < 0.15 * scale[0, 0, 0]
    assert scale[0, 0, 0] == pytest.approx(0.3 * 2.5 / 1.959963984540054)
    # deterministic in the seed; the member stream differs from the residual stream
    _, d2 = MC.draw_tables(tab, 300, seed=7, rho=0.9)
    pd.testing.assert_frame_equal(draws, d2)
    assert not np.array_equal(MC.member_choices(300, 5, seed=7), MC.member_choices(300, 5, seed=8))
    # the long frame and a sub-grid keep the members; a member mean off the record mean is refused
    tab2 = GA.PredictionTable.from_frame(tab.to_frame())
    assert tab2.has_members and np.allclose(tab2.members, tab.members) and np.allclose(tab2.conformal_q95, tab.conformal_q95)
    sub = tab.restrict(log_acid=(-1.0, 0.7))
    assert sub.has_members and sub.members.shape[1:] == sub.mean.shape
    assert tab.with_values(tab.mean + 1.0).has_members
    assert tab.record("Nd", 0, 0)["member_logD"] == pytest.approx([tab.mean[0, 0, 0] + d for d in (-0.2, -0.1, 0.0, 0.1, 0.2)])
    bad = pd.DataFrame(member_rows())
    bad.loc[0, f"{GA.MEMBER_COLUMN_PREFIX}0"] += 0.5
    with pytest.raises(ValueError, match="member means average"):
        GA.PredictionTable.from_frame(bad)
    with pytest.raises(ValueError, match="conformal_q95 must be > 0"):
        GA.PredictionTable.from_frame(pd.DataFrame(member_rows(q95=0.0)))
    # a table without member columns: the fallback only (flagged), refused when the run is registered
    plain = synthetic_table(std=0.3)
    assert not plain.has_members and plain.n_members == 0
    _, dp = MC.draw_tables(plain, 3, seed=1, rho=0.0)
    assert dp.attrs["draw_mode"] == MC.DRAW_MODE_FALLBACK and "member" not in dp.columns
    with pytest.raises(ValueError, match="V-07"):
        MC.draw_tables(plain, 3, seed=1, rho=0.0, allow_fallback=False)
    with pytest.raises(ValueError, match="V-07"):
        MC.run_monte_carlo(plain, setup_for(ts.solvating_entry(), plain), n_draws=1, n_designs=2, allow_fallback=False)
    assert {"draw_residual", "member_choice", "fallback"} <= set(MC.REGISTRATION_READINGS)
    # the prediction request carries the member columns the deployment step must fill
    grid = GA.prediction_grid(acid_M={"feed": (3.0, 3.0)}, ligand_M=(0.05, 0.3), n_acid=2, n_ligand=2)
    req = pd.read_csv(GA.write_prediction_request(tmp_path / "req.csv", grid, metals=["Nd"], system_id="s"))
    assert set(GA.MEMBER_COLUMNS) | {GA.CONFORMAL_Q95_COLUMN} <= set(req.columns) and req[GA.CONFORMAL_Q95_COLUMN].isna().all()


def test_monte_carlo_is_paired_reproducible_and_mass_balanced(tmp_path):
    entry = ts.solvating_entry()
    tab = synthetic_table(std=0.3)
    setup = setup_for(entry, tab)
    res = MC.run_monte_carlo(tab, setup, n_draws=3, n_designs=4, seed=18, rho=0.5)
    assert res.process_table.shape[0] == 12 and set(res.process_table["draw"]) == {0, 1, 2}
    # paired: the same candidate has the same variables in every draw
    for cand, g in res.process_table.groupby("candidate"):
        for v in setup.space.variables:
            assert g[v].nunique() == 1
    # reproducible: the same seed gives a byte-identical process table
    res2 = MC.run_monte_carlo(tab, setup, n_draws=3, n_designs=4, seed=18, rho=0.5)
    pd.testing.assert_frame_equal(res.process_table, res2.process_table)
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    write_csv(res.process_table, p1, float_format="%.10g")
    write_csv(res2.process_table, p2, float_format="%.10g")
    assert p1.read_bytes() == p2.read_bytes()
    # a different seed changes the draws (and so the table)
    res3 = MC.run_monte_carlo(tab, setup, n_draws=3, n_designs=4, seed=19, rho=0.5)
    assert not res3.draws["z_Nd"].equals(res.draws["z_Nd"])
    # mass balance on every draw: gen18's check_balances on the reference 6/3/3 cascade with each drawn table
    values, _ = MC.draw_tables(tab, 3, seed=18, rho=0.5)
    lig = setup.ligand
    for v in values:
        system = GA.build_gen19_system(tab, entry, feed_anion="nitrate", temperature_C=25.0, n_prior=2.5, values=v)
        res_c = solve_cascade(setup.base_spec, system)
        assert res_c.status.startswith("converged")
        bal = check_balances(setup.base_spec, res_c, system)
        assert max(abs(x) for x in bal.values()) < 1e-8, bal
    # the process table's own balance column agrees
    conv = res.process_table.loc[res.process_table["status"].str.startswith("converged")]
    assert len(conv) and (conv["balance_rel_max"] < 1e-8).all()
    assert set(res.usage["candidate"]) == set(range(4)) and (res.usage["n_evaluations"] > 0).all()


# --------------------------------------------------------------------------------------------- #
# aggregation and the lexicographic objective
# --------------------------------------------------------------------------------------------- #

def hand_table() -> pd.DataFrame:
    """Two candidates x four draws.  c0: purities 0.96, 0.98, 0.99, failed; recoveries 0.90, 0.84, 0.95, NaN;
    draw 2 carries THIRD_PHASE_RISK.  c1: purities 0.951, 0.952, 0.953, 0.954 with recoveries 0.81-0.84, all clean."""
    rows = []
    c0 = [(0.96, 0.90, "converged_newton", "OA_ASSUMED", "IN_DOMAIN_WITH_CAVEATS"),
          (0.98, 0.84, "converged_newton", "OA_ASSUMED", "IN_DOMAIN_WITH_CAVEATS"),
          (0.99, 0.95, "converged_ss", "OA_ASSUMED|THIRD_PHASE_RISK", "INADMISSIBLE"),
          (np.nan, np.nan, "failed", "NOT_CONVERGED", "INADMISSIBLE")]
    c1 = [(0.951 + 0.001 * k, 0.81 + 0.01 * k, "converged_newton", "", "IN_DOMAIN") for k in range(4)]
    for cand, spec in ((0, c0), (1, c1)):
        for k, (p, r, st, fl, reg) in enumerate(spec):
            rows.append({"draw": k, "candidate": cand, "status": st, "regime_status": reg, "in_domain": reg != "INADMISSIBLE",
                         "on_spec": False, "purity_mol": p, "recovery_from_feed": r, "recovery_total": r,
                         "net_product_mol_h": 0.05, "n_stages_total": 8 if cand == 0 else 6, "n_ext": 4 if cand == 0 else 3,
                         "n_scr": 2, "n_str": 2 if cand == 0 else 1, "oa_ext": 1.0 if cand == 0 else 1.5, "s_over_a": 0.5,
                         "w_over_a": 1.0, "consumption_index": 10.0 + cand + k, "consumption_index_incomplete": False,
                         "acid_mol_per_kg_oxide": 5.0, "throughput_kg_oxide_per_h": 0.01 * (1 + cand),
                         "throughput_mol_T_per_h_per_L_org": 0.05, "max_loading": 0.2, "flags": fl, "n_flags": 1,
                         "ood_distance_max": 0.0, "balance_rel_max": 1e-13, "iterations": 5, "front": 1})
    return pd.DataFrame(rows)


def test_p_both_arithmetic_on_a_hand_built_table():
    df = hand_table()
    usage = pd.DataFrame({"candidate": [0, 1], "statuses_used": ["IN_DOMAIN", "IN_DOMAIN|CONDITION_EXTRAPOLATION"],
                          "n_evaluations": [10, 10]})
    ops = MC.aggregate_operating_points(df, (0.95, 0.97), (0.80, 0.85), usage).set_index("candidate")
    c0, c1 = ops.loc[0], ops.loc[1]
    assert c0["n_draws"] == 4 and c0["n_converged"] == 3 and c0["n_failed"] == 1
    assert c0["p_purity_ge_0.95"] == 0.75 and c0["p_purity_ge_0.97"] == 0.5        # the failed draw is a miss
    assert c0["p_recovery_ge_0.85"] == 0.5 and c0["p_recovery_ge_0.80"] == 0.75
    assert c0["p_both_0.95_0.85"] == 0.5 and c0["p_both_0.97_0.85"] == 0.25          # draws 0 and 2; draw 2 only
    assert c0["p_constraints"] == 0.5 and c0["p_feasible"] == 0.5                       # THIRD_PHASE_RISK draw excluded
    assert c0["p_both_feasible_0.97_0.85"] == 0.0 and c0["p_both_feasible_0.95_0.80"] == 0.5
    assert c0["p_third_phase_risk"] == 0.25
    assert c0["purity_p50"] == 0.98 and math.isclose(c0["purity_p5"], 0.962) and c0["recovery_p50"] == 0.90
    assert c1["p_both_0.95_0.80"] == 1.0 and c1["p_both_0.97_0.80"] == 0.0 and c1["p_feasible"] == 1.0
    assert c1["consumption_index_median"] == 12.5 and c1["n_stages_total"] == 6
    assert c1["statuses_used"] == "IN_DOMAIN|CONDITION_EXTRAPOLATION"
    # ranking of cell (0.95, 0.80): c1 has the higher P(both) but rank 1 (extrapolation) < c0's rank 2
    ranked = RO.rank_recipes(ops.reset_index(), (0.95, 0.80))
    assert list(ranked["candidate"]) == [0, 1] and list(ranked["support_rank"]) == [2, 1]
    # with the extrapolation status lifted the sealed key 1, P(both targets), decides: c1 (1.0) before c0 (0.75); the
    # joint P(both & feasible) is printed beside and does not rank (task X finding V-06)
    lifted = RO.rank_recipes(ops.reset_index(), (0.95, 0.80), extrapolation=())
    assert list(lifted["candidate"]) == [1, 0] and list(lifted["obj_p_both"]) == [1.0, 0.75]
    assert "p_both_<cell> (desc)" == RO.LEXICOGRAPHIC_KEYS[1] and RO.LEXICOGRAPHIC_KEYS[2] == "p_feasible (desc)"
    assert "V-06" in RO.OBJECTIVE_READING
    # P(both) ties -> P(feasible) decides (sealed key 2), never the joint value
    tie = ops.reset_index().copy()
    tie.loc[tie["candidate"] == 0, "p_both_0.95_0.80"] = 1.0
    tie.loc[tie["candidate"] == 0, "p_feasible"] = 0.2
    tie.loc[tie["candidate"] == 0, "p_both_feasible_0.95_0.80"] = 0.2
    tie.loc[tie["candidate"] == 1, "p_feasible"] = 0.9
    tie.loc[tie["candidate"] == 1, "p_both_feasible_0.95_0.80"] = 0.1
    assert list(RO.rank_recipes(tie, (0.95, 0.80), extrapolation=())["candidate"]) == [1, 0]
    # task X finding VL2-03: an untracked recipe is refused, never ranked as clean
    with pytest.raises(KeyError, match="statuses_used"):
        RO.rank_recipes(ops.reset_index().drop(columns=["statuses_used"]), (0.95, 0.80))
    with pytest.raises(ValueError, match="VL2-03"):
        RO.rank_recipes(ops.reset_index().assign(statuses_used=["", np.nan]), (0.95, 0.80))
    with pytest.raises(ValueError, match="no D-source usage record"):
        MC.aggregate_operating_points(df, (0.95,), (0.80,), usage.iloc[:1])          # c1 converged, not in usage
    with pytest.raises(ValueError, match="no D-source usage record"):
        MC.aggregate_operating_points(df, (0.95,), (0.80,), usage.assign(statuses_used=["IN_DOMAIN", ""]))
    untracked = MC.aggregate_operating_points(df, (0.95,), (0.80,), None)                # no usage at all
    assert (untracked["statuses_used"] == "").all()
    with pytest.raises(ValueError, match="VL2-03"):
        RO.rank_recipes(untracked, (0.95, 0.80))


def test_unsupported_never_wins_and_the_f5_i_assertion_fires():
    df = hand_table()
    usage = pd.DataFrame({"candidate": [0, 1], "statuses_used": ["IN_DOMAIN", "IN_DOMAIN|UNSUPPORTED"], "n_evaluations": [1, 1]})
    ops = MC.aggregate_operating_points(df, (0.95,), (0.80,), usage)
    ranked = RO.rank_recipes(ops, (0.95, 0.80))
    assert list(ranked["candidate"]) == [0] and ranked.attrs["n_ineligible"] == 1     # c1 (P = 1.0) is ineligible
    RO.assert_no_unsupported_winner(RO.winner(ranked))
    forged = ops.loc[ops["candidate"] == 1].iloc[0].copy()
    forged["support_rank"] = 2
    with pytest.raises(AssertionError, match="F5"):
        RO.assert_no_unsupported_winner(forged)
    # OUTSIDE_TABLE is ineligible too; every bar lifted lets c1 win
    usage2 = usage.assign(statuses_used=["IN_DOMAIN", "OUTSIDE_TABLE"])
    ops2 = MC.aggregate_operating_points(df, (0.95,), (0.80,), usage2)
    assert list(RO.rank_recipes(ops2, (0.95, 0.80))["candidate"]) == [0]
    assert list(RO.rank_recipes(ops2, (0.95, 0.80), barred=(), extrapolation=())["candidate"]) == [1, 0]
    # F5 audit: barred (UNSUPPORTED + CONDITION_EXTRAPOLATION) never reaches 0.5, allowed does -> F5(ii)
    f5 = RO.f5_audit(ops, [(0.95, 0.80)])
    cell = f5["cells"][0]
    assert cell["F5_i"] is False and cell["winner_registered"] == 0 and cell["winner_gate_lifted"] == 1
    assert cell["winner_changes_when_gate_lifted"] is True
    assert cell["barred_unsupported_and_condition_extrapolation"]["max_p_both"] == 0.75    # c0: 3 of 4 draws
    assert cell["allowed_all_statuses"]["max_p_both"] == 1.0 and cell["F5_ii"] is False
    df_low = df.copy()
    df_low.loc[(df_low["candidate"] == 0) & (df_low["draw"] <= 1), "recovery_from_feed"] = 0.5   # c0 drops to 0.25
    ops_low = MC.aggregate_operating_points(df_low, (0.95,), (0.80,), usage)
    assert ops_low.set_index("candidate").loc[0, "p_both_0.95_0.80"] == 0.25
    f5b = RO.f5_audit(ops_low, [(0.95, 0.80)])
    assert f5b["cells"][0]["F5_ii"] is True and f5b["F5"] is True


def test_s2d_stability_arithmetic():
    rng = np.random.default_rng(0)
    rows = []
    # c0 dominates every draw; c1 is a near-duplicate recipe (stage counts within 1, O/A within 10 %); c2 far away
    for k in range(16):
        for cand, (p, r, n_ext, oa) in enumerate(((0.99, 0.95, 5, 1.0), (0.98, 0.94, 6, 1.05), (0.97, 0.93, 12, 3.0))):
            rows.append({"draw": k, "candidate": cand, "status": "converged_newton", "regime_status": "IN_DOMAIN",
                         "in_domain": True, "on_spec": True, "purity_mol": p + 0.001 * rng.standard_normal(),
                         "recovery_from_feed": r, "recovery_total": r, "net_product_mol_h": 0.05, "n_stages_total": n_ext + 4,
                         "n_ext": n_ext, "n_scr": 2, "n_str": 2, "oa_ext": oa, "s_over_a": 0.5, "w_over_a": 1.0,
                         "consumption_index": 10.0, "consumption_index_incomplete": False, "acid_mol_per_kg_oxide": 5.0,
                         "throughput_kg_oxide_per_h": 0.01, "throughput_mol_T_per_h_per_L_org": 0.05, "max_loading": 0.2,
                         "flags": "", "n_flags": 0, "ood_distance_max": 0.0, "balance_rel_max": 1e-13, "iterations": 5,
                         "front": 1})
    df = pd.DataFrame(rows)
    usage = pd.DataFrame({"candidate": [0, 1, 2], "statuses_used": ["IN_DOMAIN"] * 3, "n_evaluations": [1, 1, 1]})
    s = RO.stability(df, usage, (0.95, 0.90), (0.95,), (0.90,), n_boot=10, seed=19)
    assert s["top_candidate"] == 0 and s["fraction_kept"] == 1.0 and s["passes"] is True and len(s["bootstrap_winners"]) == 10
    # same_recipe tolerances
    a = {"n_ext": 5, "n_scr": 2, "n_str": 2, "oa_ext": 1.0}
    assert RO.same_recipe(a, {"n_ext": 6, "n_scr": 2, "n_str": 1, "oa_ext": 1.09})
    assert not RO.same_recipe(a, {"n_ext": 7, "n_scr": 2, "n_str": 2, "oa_ext": 1.0})
    assert not RO.same_recipe(a, {"n_ext": 5, "n_scr": 2, "n_str": 2, "oa_ext": 1.11})
    # an alternating winner: c0 qualifies in even draws only, c2 in odd draws only, c1 never -> c0 and c2 tie at
    # P = 0.5 on the full draws (c0 wins the tie on stages) and a bootstrap with more odd draws flips to c2
    df2 = df.copy()
    odd = df2["draw"] % 2 == 1
    df2.loc[odd & (df2["candidate"] == 0), "purity_mol"] = 0.90
    df2.loc[odd & (df2["candidate"] == 2), "purity_mol"] = 0.999
    df2.loc[~odd & (df2["candidate"] == 2), "purity_mol"] = 0.90
    df2.loc[df2["candidate"] == 1, "purity_mol"] = 0.90
    s2 = RO.stability(df2, usage, (0.95, 0.90), (0.95,), (0.90,), n_boot=20, seed=19)
    assert s2["top_candidate"] == 0 and s2["top_recipe"]["p_both_feasible"] == 0.5
    assert 0.0 < s2["fraction_kept"] < 1.0
    assert {w["candidate"] for w in s2["bootstrap_winners"]} == {0, 2}
    # deterministic in the seed
    s3 = RO.stability(df2, usage, (0.95, 0.90), (0.95,), (0.90,), n_boot=20, seed=19)
    assert s3["fraction_kept"] == s2["fraction_kept"]
    # resampling with replacement keeps the draw count
    sub = RO.resample_draws(df, np.random.default_rng(1))
    assert sub["boot_copy"].nunique() == 16 and len(sub) == len(df)


# --------------------------------------------------------------------------------------------- #
# the runner's gate and its outputs
# --------------------------------------------------------------------------------------------- #

def _ledgers(out: Path, *, marker: bool = True, errors: bool = False, records_index: bool = True, final_stage: bool = True):
    prog = {"markers": [D.NOT_IMPLEMENTED] if marker else [], "jobs": {"j1": {"errors": ["boom"] if errors else []}}}
    write_json(D.discovery_root(out) / "run_info_progress.json", prog)
    if records_index:
        write_csv(pd.DataFrame({"a": [1]}), D.discovery_root(out) / "records_index.csv")
    stages = [D.STAGES["not_implemented"]] if final_stage else ["00_safeguard"]
    write_json(D.discovery_root(out) / "decisions" / "wall_clock.json", {"invocations": [{"stages_done": stages}]})


def _stop(out: Path, stop):
    write_json(D.discovery_root(out) / "decisions" / "stop_rule.json", {"stop": stop})


def _conf(out: Path, s1, deployed="M2", s2=None, v6=False):
    RP = _runner()
    write_json(RP.confirmation_path(out), {"schema": RP.CONFIRMATION_SCHEMA, "S1": {"passed": s1, "deployed_predictor": deployed},
                                           "S2": {"passed": s2}, "V6": {"run": v6}, "seeds": {"verified": True}})


def test_runner_refuses_without_the_decision_files(tmp_path):
    RP = _runner()
    out = tmp_path / "root"
    ok_seal = lambda check, digests, expect_addenda=None: {"footer": "f" * 64, "n_addenda": 1}   # noqa: E731
    heavy_ok = lambda: {"complete": True, "n_incomplete": 0}                                     # noqa: E731
    kw = dict(seal=ok_seal, heavy_discovery_check=heavy_ok)
    # (a) the seal gate is first: a failing --check refuses before anything is read
    def bad_seal(check, digests, expect_addenda=None):
        raise SystemExit("refused: scripts/g19_seal_prereg.py --check exited 3")
    with pytest.raises(SystemExit, match="refused: scripts/g19_seal_prereg.py"):
        RP.refuse_unless_ready(out, seal=bad_seal, heavy_discovery_check=heavy_ok)
    # (b) no ledgers -> not complete
    with pytest.raises(SystemExit, match="not complete"):
        RP.refuse_unless_ready(out, **kw)
    _ledgers(out, marker=False)
    with pytest.raises(SystemExit, match="final stage"):
        RP.refuse_unless_ready(out, **kw)
    _ledgers(out, errors=True)
    with pytest.raises(SystemExit, match="errors"):
        RP.refuse_unless_ready(out, **kw)
    _ledgers(out, final_stage=False)
    with pytest.raises(SystemExit, match="wall_clock"):
        RP.refuse_unless_ready(out, **kw)
    _ledgers(out)
    # the record check runs only after the cheap checks and can refuse
    with pytest.raises(SystemExit, match="record set"):
        RP.refuse_unless_ready(out, seal=ok_seal, heavy_discovery_check=lambda: {"complete": False, "n_incomplete": 3})
    # (c) stop rule missing / pending / fired
    with pytest.raises(SystemExit, match="stop"):
        RP.refuse_unless_ready(out, **kw)
    _stop(out, None)
    with pytest.raises(SystemExit, match="not decided"):
        RP.refuse_unless_ready(out, **kw)
    _stop(out, True)
    with pytest.raises(SystemExit, match="stop rule fired"):
        RP.refuse_unless_ready(out, **kw)
    _stop(out, False)
    # (d) confirmation decision missing -> refused unless --exploratory (then labelled transfer-unsupported)
    with pytest.raises(SystemExit, match="S1.passed is undecided"):
        RP.refuse_unless_ready(out, **kw)
    gate = RP.refuse_unless_ready(out, exploratory=True, **kw)
    assert gate["label"]["label"] == RP.TRANSFER_UNSUPPORTED and gate["label"]["headline_allowed"] is False
    assert gate["confirmation"]["present"] is False
    # S1 failed -> refused even with --exploratory
    _conf(out, False)
    with pytest.raises(SystemExit, match="S1 FAILED"):
        RP.refuse_unless_ready(out, exploratory=True, **kw)
    # S1 passed, S2 pending, V6 not run -> runs, labelled transfer-unsupported
    _conf(out, True, s2=None, v6=False)
    gate = RP.refuse_unless_ready(out, **kw)
    assert gate["label"]["label"] == RP.TRANSFER_UNSUPPORTED and "V6 has not run" in gate["label"]["reasons"]
    assert gate["confirmation"]["deployed_predictor"] == "M2" and gate["stop_rule"]["stop"] is False
    # S1, S2 and V6 -> registered
    _conf(out, True, s2=True, v6=True)
    gate = RP.refuse_unless_ready(out, **kw)
    assert gate["label"] == {"label": "registered", "headline_allowed": True, "reasons": []}
    # the stop rule fired but --exploratory: runs, labelled
    _stop(out, True)
    assert RP.refuse_unless_ready(out, exploratory=True, **kw)["label"]["label"] == RP.TRANSFER_UNSUPPORTED
    # a foreign schema is refused
    _stop(out, False)
    write_json(RP.confirmation_path(out), {"schema": "other", "S1": {"passed": True}})
    with pytest.raises(SystemExit, match="schema"):
        RP.refuse_unless_ready(out, **kw)


def test_runner_case_helpers_and_outputs_on_a_synthetic_case(tmp_path):
    """The output writer, F5 audit, figures and D06 on a synthetic table with gen18's test entry (no real case file)."""
    RP = _runner()
    out = tmp_path / "root"
    entry = ts.solvating_entry()
    tab = synthetic_table(std=0.3)
    setup = setup_for(entry, tab)
    res = MC.run_monte_carlo(tab, setup, n_draws=3, n_designs=4, seed=18, rho=0.4)
    label = {"label": RP.TRANSFER_UNSUPPORTED, "headline_allowed": False, "reasons": ["--exploratory run"]}
    written = RP.write_run_outputs(RP.process_root(out), res, setup, label=label, n_boot=3, boot_seed=19)
    root = RP.process_root(out)
    for f in ("process_table.csv", "candidates.csv", "draws.csv", "usage.csv", "operating_points.csv", "rankings.csv",
              "winners.csv", "stability.json", "f5.json"):
        assert (root / f).exists(), f
    win = pd.read_csv(root / "winners.csv")
    assert len(win) == 9 and set(win["label"]) == {RP.TRANSFER_UNSUPPORTED} and (win["headline_allowed"] == False).all()  # noqa: E712
    f5 = json.loads((root / "f5.json").read_text(encoding="utf-8"))
    assert f5["F5_i_any_cell"] is False and len(f5["cells"]) == 9
    figs = [RP.figure_pareto(written["ops"], written["winners"], setup, label["label"], out / "figures" / "F14.png"),
            RP.figure_probability_map(written["ops"], written["winners"], setup, label["label"], out / "figures" / "F15.png")]
    assert all(p.exists() and p.stat().st_size > 1000 for p in figs)
    # D06 from files: "not computed" without summary.json, then with it
    md = RP.d06_markdown(out)
    assert "not computed" in md and "## Verdict" in md and "null" in md
    write_json(root / "summary.json", {"schema": RP.SCHEMA, "label": label, "gate": {}, "inputs": {}, "case": {},
                                       "monte_carlo": res.attrs, "gen18_code_sha256": "x" * 64})
    p = RP.write_d06(out)
    text = p.read_text(encoding="utf-8")
    for section in ("## Question", "## Evidence", "## Metrics", "## Verdict", "## Decision", "## Next action"):
        assert section in text
    assert RP.TRANSFER_UNSUPPORTED in text
    # the prediction request lists every metal x grid point with empty prediction columns
    grid = GA.prediction_grid(acid_M={"feed": (3.0, 3.0), "strip": (0.01, 0.3)}, ligand_M=(0.05, 0.3), n_acid=4, n_ligand=3)
    req = GA.write_prediction_request(tmp_path / "req.csv", grid, metals=["Nd", "Pr"], system_id="sys_x")
    rq = pd.read_csv(req)
    assert len(rq) == 24 and rq["mean_logD"].isna().all() and set(rq["metal"]) == {"Nd", "Pr"}
    assert math.isclose(rq["log_acid"].min(), -2.0) and math.isclose(rq["log_acid"].max(), math.log10(3.0))
    # the design-space intersection with a table box narrows the acid / ligand windows and refuses a disjoint one
    space, changes = RP.intersect_space_with_table(setup.space, tab)
    assert space.bounds["strip_acid_M"][0] >= 10 ** -2.0 and "ligand_total_M" in changes
    far = synthetic_table(la=np.linspace(2.0, 3.0, 3))
    with pytest.raises(SystemExit, match="does not intersect"):
        RP.intersect_space_with_table(setup.space, far)


def test_registered_constants():
    assert MC.K_DRAWS_REGISTERED == 64 and RO.N_BOOTSTRAP_RERANKINGS == 20 and RO.BOOTSTRAP_SEED == 19
    assert RO.STABILITY_THRESHOLD == 0.80 and RO.F5_P_BOTH_THRESHOLD == 0.5 and MC.TRUNCATION_LEVEL == 0.95
    assert RO.INELIGIBLE_STATUSES == {"UNSUPPORTED", "OUTSIDE_TABLE"}
    assert RO.EXTRAPOLATION_STATUSES == {"CONDITION_EXTRAPOLATION", "FAMILY_EXTRAPOLATION"}
    assert set(MC.CONSTRAINT_FLAGS) == {"THIRD_PHASE_RISK", "LOADING_CAP_HIT", "HIGH_LOADING"}
    assert MC.DRAW_MODE_REGISTERED == "member_plus_conformal_residual" and MC.Z_975 == pytest.approx(1.959963984540054)
    assert GA.MEMBER_COLUMNS == tuple(f"member_logD_{k}" for k in range(5)) and GA.CONFORMAL_Q95_COLUMN == "conformal_q95"
