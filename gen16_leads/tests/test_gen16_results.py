"""Regression tests for the gen16 results: the numbers the report is written from.

Fast tests pin every headline value to the committed artefact it was quoted from, so a
regenerated CSV cannot silently diverge from `DECISION_REPORT.md`.  Slow tests (`-m slow`) re-run
the pipelines that produce them.

Run:  .venv/Scripts/python.exe -m pytest gen16_leads/tests -q -m "not slow"
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
R = HERE / "results"
PY = sys.executable


def _row(df: pd.DataFrame, **eq) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for k, v in eq.items():
        m &= df[k] == v
    sub = df[m]
    assert len(sub) == 1, f"expected exactly one row for {eq}, got {len(sub)}"
    return sub.iloc[0]


# --------------------------------------------------------------------------------------
# the confirmed claim
# --------------------------------------------------------------------------------------
def test_confirmation_replicates_in_all_five_designs():
    t = pd.read_csv(R / "confirmation" / "discovery_vs_confirmation.csv")
    assert len(t) == 5 and set(t.design) == {"B", "BR", "BQ", "A", "BP"}
    assert t.confirmation_passes_P1.all()
    assert (t.confirmation_ci_low > 0).all(), "every confirmation interval must exclude zero"
    assert (t.confirmation_seeds_positive == 5).all()
    bp = _row(t, design="BP")
    assert bp.discovery_point == pytest.approx(1.988220, abs=1e-5)
    assert bp.confirmation_point == pytest.approx(1.909, abs=1e-3)


def test_confirmation_ran_once_on_the_committed_seeds():
    """The seeds actually used must hash to the commitment sealed before any lead ran.

    The seeds are compared through the commitment rather than written out here, so that this
    file stays clean under ``scripts/g16_audit_seeds.py`` (no discovery file may contain a
    confirmation-seed literal) and the check remains the one that matters: the run used the
    seeds the pre-registration committed to.
    """
    import hashlib
    import json
    commitment = "5a30455bc67d364aa3e65f6d3d28bd4b6fcf3301fe979ca4a1be4a9547de76a9"
    mark = json.loads((R / "confirmation" / "CONFIRMATION_RUN_ONCE.json").read_text())
    seeds = sorted(int(s) for s in mark["seeds"])
    assert hashlib.sha256(json.dumps(seeds).encode()).hexdigest() == commitment
    assert commitment in (HERE / "PRE_REGISTRATION.md").read_text(encoding="utf-8")
    assert mark["claims"] == ["C1_L3A_measurements_saved"]
    assert mark["suffix"] == "", "a suffix means the confirmation was run more than once"


def test_l3a_discovery_rule_met_in_all_five_designs():
    c = pd.read_csv(R / "L3" / "l3a_contrasts.csv")
    reg = c[(c.family == "registered") & (c.comparison == "G14_saved_vs_0")]
    assert len(reg) == 5
    assert (reg.point > 0).all() and (reg.p_perm < 0.05).all() and (reg.ci95_low > 0).all()


def test_heavier_always_saves_exactly_zero():
    """The registered cheapest competitor is degenerate by construction, not by measurement."""
    c = pd.read_csv(R / "L3" / "l3a_contrasts.csv")
    h = c[c.comparison == "HEAVIER_ALWAYS_saved_vs_0"]
    assert len(h) == 5 and (h.point.abs() < 1e-12).all()


# --------------------------------------------------------------------------------------
# the closed leads
# --------------------------------------------------------------------------------------
def test_l2_gate_is_closed():
    c = pd.read_csv(R / "L2" / "gate_contrasts.csv")
    bp = _row(c[c.family == "registered"], design="BP", comparison="OCURVLPO_vs_G14")
    assert bp.point == pytest.approx(0.0287, abs=5e-4)
    assert bp.ci95_low < 0 < bp.ci95_high, "the gate closed because the interval contains zero"
    assert not bool(bp.passes_P1)


def test_l2_reproduces_the_gen15_anchors():
    b = pd.read_csv(R / "L2" / "gate_board.csv")
    w = b[b.design == "BP"].set_index("arm").macro_mae_extractant
    assert w["G14"] == pytest.approx(0.5000794414203691, abs=1e-9)
    assert w["O_CURV"] == pytest.approx(0.4272110620580191, abs=1e-9)
    assert w["FLAT"] == pytest.approx(0.5885062528901843, abs=1e-9)


def test_l1_dose_response_is_monotone_in_bookkeeping_exactness():
    s = pd.read_csv(R / "L1" / "stage1_stats.csv")
    col = "rho_a" if "rho_a" in s.columns else [c for c in s.columns if c.startswith("rho")][0]
    got = {}
    for model in ("NAIVE", "ELEM", "SPECIES_CONST", "SPECIES"):
        sub = s[(s.model == model) & (s.set == "S8")]
        if len(sub):
            got[model] = float(sub.iloc[0][col])
    assert got["NAIVE"] > got["ELEM"] > got["SPECIES_CONST"] > got["SPECIES"], got
    assert got["SPECIES"] < 0.25, "the registered row must sit below the closed threshold"


def test_l4_registered_contrast_fails_the_five_design_rule():
    c = pd.read_csv(R / "L4" / "contrasts_abc.csv")
    reg = c[(c.family == "registered") & (c.comparison == "AOPT_vs_RANDOM")]
    assert len(reg) == 5
    assert not reg.passes_P1.all(), "L4 must not read as supported"
    assert (reg.point > 0).sum() < 5, "the sign flips; that is why it fails"


def test_l6_no_relaxation_adds_a_chemotype():
    r = pd.read_csv(R / "L6_cohort_audit" / "relaxations.csv")
    col = "n_chemotypes" if "n_chemotypes" in r.columns else "chemotypes"
    assert set(r[col]) == {45}, "every relaxation must leave 45 chemotypes"


# --------------------------------------------------------------------------------------
# protocol artefacts
# --------------------------------------------------------------------------------------
def test_large_artefact_manifest_matches_disk():
    r = subprocess.run([PY, str(HERE / "scripts" / "g16_manifest.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_contrast_files_all_declare_a_family():
    """Every gen16 comparison must be counted in a family.

    ``anchors/`` is excluded on purpose: it reproduces gen15's own locked contrast as a bench
    check, so it is not a gen16 comparison and must not enter the multiplicity count.
    ``refutation/`` and ``confirmation/`` are excluded for the same reason — a refuter check
    cannot become a claim, and the confirmation run tested one pre-counted claim.
    """
    skip = {"refutation", "confirmation", "anchors"}
    files = [p for p in R.rglob("*contrast*.csv") if not skip & set(p.parts)]
    assert files
    for p in files:
        d = pd.read_csv(p)
        assert "family" in d.columns, f"{p} has no family column"
        assert set(d.family) <= {"registered", "exploratory"}, p


def test_comparison_accounting_matches_the_report():
    """1030 contrast rows, 128 registered and 902 exploratory (DECISION_REPORT.md section 1)."""
    skip = {"refutation", "confirmation", "anchors"}
    n = {"registered": 0, "exploratory": 0}
    for p in R.rglob("*contrast*.csv"):
        if skip & set(p.parts):
            continue
        for fam, k in pd.read_csv(p).family.value_counts().items():
            n[fam] += int(k)
    assert n == {"registered": 128, "exploratory": 902}, n


@pytest.mark.slow
def test_confirmation_claim_reproduces_the_discovery_numbers():
    """The claim runner, pointed at the discovery seeds, must reproduce the lead's own CSV."""
    sys.path.insert(0, str(HERE))
    from gen16 import claims
    assert claims.self_check() == 0


# --------------------------------------------------------------------------------------
# L1 Stage 2: the cycle was run for real
# --------------------------------------------------------------------------------------
def test_stage2_all_references_converged():
    r = pd.read_csv(R / "L1" / "reference_species" / "reference_energies.csv")
    assert len(r) == 361
    assert (r.converged == "ok").all()
    assert r.energy_eV.notna().all()
    assert set(r.xtb_version.unique()) == {"6.7.1pre"}


def test_stage2_cycle_leaves_binding_energies_not_free_species_totals():
    """The bookkeeping check the lead turns on: after an exact subtraction the fill-species
    coefficients must be mean binding energies, not free-species totals."""
    c = pd.read_csv(R / "L1" / "stage2_coefficients.csv")
    sp = c[c.model == "SPECIES"]
    no3 = sp[sp.term.str.contains("n_NO3")].iloc[0]["coef"]
    h2o = sp[sp.term.str.contains("n_H2O")].iloc[0]["coef"]
    assert -25 < no3 < -8, no3          # Stage 1 had -431.97 (a free-species total)
    assert -3 < h2o < 0, h2o            # Stage 1 had -138.88
    ref = pd.read_csv(R / "L1" / "reference_species" / "reference_energies.csv").set_index("species")
    # the Stage 1 coefficient equals the true free-species energy plus this binding energy
    assert abs((-431.970549915437 - float(ref.loc["nitrate", "energy_eV"])) - no3) < 0.5
    assert abs((-138.87532349058438 - float(ref.loc["water", "energy_eV"])) - h2o) < 0.5


def test_stage2_verdict_is_closed():
    import json
    d = json.loads((R / "L1" / "stage2_decision.json").read_text())
    assert d["registered"]["verdict"] == "closed"
    assert abs(d["registered"]["rho_a"] + 0.0837) < 5e-4
    assert d["registered"]["ci95_low"] < 0 < d["registered"]["ci95_high"]
    # the cycle also kills the uncorrected construction that produced gen15's +0.644
    assert abs(d["cycle_only_NAIVE_on_dE"]["rho_a"]) < 0.25


def test_stage2_null_has_power():
    """Unlike Stage 1, whose estimator had reliability 0.000 and could detect nothing."""
    rel = pd.read_csv(R / "L1" / "stage2_reliability.csv")
    s2 = rel[rel.quantity.str.startswith("Stage 2")].iloc[0]
    assert s2.split_half_spearman > 0.5, s2.split_half_spearman
    assert s2.reliability > 0.5, s2.reliability
    assert s2.max_attainable_abs_rho > 0.40, "must exceed the registered decision bar"


def test_stage2_permutation_null_fails_decisively():
    p = pd.read_csv(R / "L1" / "stage2_perm_null.csv")
    a = p[p.target == "a"].iloc[0]
    assert a.observed_max_abs_rho < a.null_p95_max_abs_rho
    assert a.familywise_p > 0.5, a.familywise_p
