"""Gen16 adversarial audit -- NUMBERS lens.

Re-derives every quantitative statement in gen16_leads/DECISION_REPORT.md from the
artefact it cites and writes one row per checked number to
gen16_leads/results/audit/numbers/number_check.csv.

Read-only with respect to gen13/gen14/gen15/src and to gen16_leads/ outside
results/audit/numbers/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "gen16_leads" / "results"
OUT = RES / "audit" / "numbers"
OUT.mkdir(parents=True, exist_ok=True)

rows: list[dict] = []


def check(section, what, quoted, artefact, value, tol=None):
    """Record one comparison. ``tol`` defaults to half a unit in the last quoted dp."""
    if value is None:
        rows.append(dict(section=section, quantity=what, quoted_value=quoted,
                         artefact_path=artefact, artefact_value="", abs_diff="",
                         verdict="not_found"))
        return
    if quoted == "not printed":
        # the artefact carries a number the report never quotes; recorded, not scored
        rows.append(dict(section=section, quantity=what, quoted_value=quoted,
                         artefact_path=artefact, artefact_value=value, abs_diff="",
                         verdict="not_found"))
        return
    try:
        q = float(quoted)
        v = float(value)
        d = abs(q - v)
        if tol is None:
            s = str(quoted)
            dp = len(s.split(".")[1]) if "." in s else 0
            tol = 0.5 * 10 ** (-dp) + 1e-12
        verdict = "match" if d <= tol else "mismatch"
    except (TypeError, ValueError):
        d = ""
        verdict = "match" if str(quoted) == str(value) else "mismatch"
    rows.append(dict(section=section, quantity=what, quoted_value=quoted,
                     artefact_path=artefact, artefact_value=value, abs_diff=d,
                     verdict=verdict))


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    qs = p[o] * n / np.arange(1, n + 1)
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    q[o] = np.minimum(qs, 1.0)
    return q


# ---------------------------------------------------------------- seal / anchors
prereg = (ROOT / "gen16_leads" / "PRE_REGISTRATION.md").read_bytes().replace(b"\r\n", b"\n")
lines = prereg.split(b"\n")
last = max(i for i, l in enumerate(lines) if l.strip() == b"---")
body = b"\n".join(lines[:last]) + b"\n"
check("header", "PRE_REGISTRATION body sha256",
      "d004c30388078ae97232af2537a1915360360a7cb9eda07ebe4076764c48f50e",
      "gen16_leads/PRE_REGISTRATION.md", hashlib.sha256(body).hexdigest())

anch = json.loads((RES / "anchors" / "anchors.json").read_text())
ap = rel(RES / "anchors" / "anchors.json")
check("1", "G13_ET_TOPO39 BP", "0.7683085207475452", ap,
      anch["anchors"]["G13_ET_TOPO39_BP_macro_direction_accuracy"]["obtained"], 0)
check("1", "G14 BP macro MAE", "0.5000794414203691", ap,
      anch["anchors"]["G14_BP_macro_mae_extractant"]["obtained"], 0)
check("1", "FLAT BP macro MAE", "0.5885062528901843", ap,
      anch["anchors"]["FLAT_BP_macro_mae_extractant"]["obtained"], 0)
g14f = anch["five_designs"]["G14_vs_FLAT"]["BP"]
check("1", "G14-FLAT BP point", "0.0884", ap, g14f["point"])
check("1", "G14-FLAT BP CI low", "0.0010", ap, g14f["ci95_low"])
check("1", "G14-FLAT BP CI high", "0.1473", ap, g14f["ci95_high"])
check("1", "G14-FLAT BP p", "0.047", ap, g14f["p_two_sided"])

# ---------------------------------------------------------------- section 1 accounting
LEAD_FILES = {
    "L1": ["L1/contrasts_stage1.csv"],
    "L2": ["L2/gate_contrasts.csv"],
    "L3": ["L3/l3a_contrasts.csv", "L3/l3c_contrasts.csv"],
    "L4": ["L4/contrasts_abc.csv", "L4/contrasts_cellmatched.csv",
           "L4/contrasts_nodga.csv", "L4/contrasts_per_budget.csv"],
    "L5": ["L5/calibration_direction_contrasts.csv", "L5/contrasts_cov.csv",
           "L5/contrasts_mix.csv"],
    "L4_dry": ["L4/_dry/contrasts_abc.csv", "L4/_dry/contrasts_per_budget.csv"],
}
frames = {}
for lead, fs in LEAD_FILES.items():
    frames[lead] = pd.concat([pd.read_csv(RES / f).assign(__f=f) for f in fs],
                             ignore_index=True)
for lead, quoted in [("L1", 90), ("L2", 20), ("L3", 85), ("L4", 435), ("L5", 230)]:
    check("1", "contrast rows " + lead, quoted,
          "; ".join(LEAD_FILES[lead]), len(frames[lead]), 0)
ALL = pd.concat(frames.values(), ignore_index=True)
LEADS_ONLY = pd.concat([frames[k] for k in ("L1", "L2", "L3", "L4", "L5")], ignore_index=True)
check("1", "total contrast rows (incl. L4/_dry dry-run)", 940,
      "results/**/contrasts_*.csv", len(ALL), 0)
check("1", "total contrast rows summed over the five itemised leads", 940,
      "results/**/contrasts_*.csv (L1+L2+L3+L4+L5 as itemised in the report)",
      len(LEADS_ONLY), 0)
check("1", "registered rows", 126, "results/**/contrasts_*.csv",
      int((ALL.family == "registered").sum()), 0)
check("1", "exploratory rows", 814, "results/**/contrasts_*.csv",
      int((ALL.family == "exploratory").sum()), 0)

reg = ALL[ALL.family == "registered"].copy()
reg["q"] = bh(reg["p_two_sided"].fillna(1.0).values)
allq = ALL.copy()
allq["q"] = bh(allq["p_two_sided"].fillna(1.0).values)


def sel(d):
    return d[(d.comparison == "G14_saved_vs_0") & (d.design == "BP")]["q"].iloc[0]


check("1", "BH q, registered family (n=126)", 0.039, "results/**/contrasts_*.csv", sel(reg))
check("1", "BH q, all 940", 0.025, "results/**/contrasts_*.csv", sel(allq))
glob_rows = sum(len(pd.read_csv(f)) for f in sorted(RES.glob("*/contrasts_*.csv")))
check("1", "rows produced by the cited glob results/*/contrasts_*.csv", 940,
      "results/*/contrasts_*.csv (literal glob as printed in the report)", glob_rows, 0)
have_bh = {f: any("bh" in c for c in pd.read_csv(RES / f, nrows=1).columns)
           for fs in LEAD_FILES.values() for f in fs}
check("1", "lead contrast files carrying a BH-adjusted p column", "all",
      "results/**/contrasts_*.csv",
      "missing in: " + ", ".join(sorted(f for f, v in have_bh.items() if not v)))

# ---------------------------------------------------------------- section 2 L3a
dvc = pd.read_csv(RES / "confirmation" / "discovery_vs_confirmation.csv")
p = rel(RES / "confirmation" / "discovery_vs_confirmation.csv")
tab2a = {
    "B": (2.207, 0.521, 4.118, 2.086, 0.234, 4.170),
    "BR": (2.149, 0.455, 3.985, 2.170, 0.252, 4.195),
    "BQ": (2.179, 0.468, 4.070, 2.123, 0.248, 4.196),
    "A": (2.492, 0.771, 4.544, 2.285, 0.620, 4.366),
    "BP": (1.988, 0.443, 3.781, 1.909, 0.360, 3.760),
}
for d, vals in tab2a.items():
    r = dvc[dvc.design == d].iloc[0]
    names = ["discovery", "disc CI low", "disc CI high",
             "confirmation", "conf CI low", "conf CI high"]
    cols = ["discovery_point", "discovery_ci_low", "discovery_ci_high",
            "confirmation_point", "confirmation_ci_low", "confirmation_ci_high"]
    for name, q, col in zip(names, vals, cols):
        check("2a", "saved " + d + " " + name, q, p, r[col])

saved = pd.read_csv(RES / "L3" / "l3a_saved.csv")
ps = rel(RES / "L3" / "l3a_saved.csv")
bp = saved[(saved.design == "BP") & (saved.arm == "G14") & (saved.split == "all")].iloc[0]
check("headline/2", "E_random", 6.906, ps, bp.e_random)
check("headline/2", "E_model BP", 4.92, ps, bp.e_model)
check("headline/2", "saved BP", 1.99, ps, bp.saved)
check("headline/2", "saved fraction BP (%)", 28.8, ps, 100 * bp.saved_frac)
ha = saved[(saved.design == "BP") & (saved.arm == "HEAVIER_ALWAYS") & (saved.split == "all")].iloc[0]
check("2b", "HEAVIER_ALWAYS saved BP", 0.0, ps, ha.saved, 0)

cB = pd.read_csv(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_competitors.csv")
pc = rel(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_competitors.csv")
check("2b", "DIR_DONORS13 saved BP", -3.14, pc,
      cB[(cB.design == "BP") & (cB.arm == "DIR_DONORS13")].saved.iloc[0])
ci = pd.read_csv(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_competitor_increments.csv")
check("2b", "G14 increment over DIR_DONORS13, BP", 5.13,
      rel(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_competitor_increments.csv"),
      ci[ci.design == "BP"].G14_minus_DIR_DONORS13.iloc[0])

ckB = pd.read_csv(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_checks.csv")
pk = rel(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_checks.csv")


def g(c):
    return ckB[(ckB.design == "BP") & (ckB["check"] == c)].point.iloc[0]


check("2c", "headline re-derived", 1.988220, pk, g("headline"), 1e-6)
check("2c", "drop sc009", 1.459, pk, g("drop_sc009_diglycolamides"))
check("2c", "sc009-only", 0.386, pk, g("sc009_only"))
wp = pd.read_csv(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_withinpub.csv")
pw = rel(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_withinpub.csv")
w = wp[(wp.design == "BP") & (wp["check"] == "within_publication_min5")].iloc[0]
check("2c", "within-publication saved BP", 0.0046, pw, w.point)
check("2c", "within-publication CI low", -0.016, pw, w.ci_low)
check("2c", "within-publication CI high", 0.022, pw, w.ci_high)
check("2c", "within-publication constant-call fraction (%)", 69.6, pw,
      100 * w.frac_tasks_constant_call)
check("2c", "within-publication n publications", 4, pw, w.n_groups, 0)
wc = wp[(wp.design == "BP") & (wp["check"] == "within_chemotype_min5")].iloc[0]
check("2c", "within-chemotype saved BP", 0.334, pw, wc.point)
lo = wp[(wp.design == "BP") & (wp["check"] == "lopo_publication_rerun")].iloc[0]
check("2c", "LOPO min", 1.506, pw, lo.loco_min)
check("2c", "LOPO max", 2.428, pw, lo.loco_max)
check("2c", "LOPO sign flips", 0, pw, lo.n_sign_flips, 0)
check("2c", "LOPO n publications", 58, pw, lo.n_groups, 0)
s2 = json.loads((RES / "refutation" / "L3A" / "A" / "checks_stage2.json").read_text())
p2 = rel(RES / "refutation" / "L3A" / "A" / "checks_stage2.json")
check("2c", "n_metals-stratified real-shuffled BP", 1.886, p2,
      s2["BP"]["perm_nmetals_stratified"]["real_minus_shuffled"])
check("2c", "n_metals attributable share (%)", 5, p2,
      100 * (1 - s2["BP"]["perm_nmetals_stratified"]["real_minus_shuffled"] /
             s2["BP"]["saved_all"]), 0.6)
cl = pd.read_csv(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_ceiling.csv")
pcl = rel(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_ceiling.csv")
cw = cl[(cl.design == "BP") & (cl.regime == "within_publication")].iloc[0]
cp = cl[(cl.design == "BP") & (cl.regime == "pooled")].iloc[0]
check("2", "perfect-sign ceiling, within publication", 0.414, pcl, cw.CEILING)
check("2", "headroom captured within publication (%)", 1.1, pcl, 100 * cw.G14_over_ceiling)
check("2", "headroom captured pooled (%)", 47.3, pcl, 100 * cp.G14_over_ceiling)
fp = pd.read_csv(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_foldpure.csv")
fpb = fp[(fp.design == "BP") & (fp.regime == "fold_pure")].iloc[0]
check("2c", "fold-pure saved BP (report gives words, no number)", "not printed",
      rel(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_foldpure.csv"),
      round(float(fpb.point), 4))
check("2c", "fold-pure saved fraction BP (report gives words, no number)", "not printed",
      rel(RES / "refutation" / "L3A" / "B" / "refute_l3a_B_foldpure.csv"),
      round(100 * float(fpb.saved_frac), 1))

# ---------------------------------------------------------------- section 3 L3c
m = pd.read_csv(RES / "L3" / "l3c_metrics.csv")
pm = rel(RES / "L3" / "l3c_metrics.csv")
mm = m[m.design == "BP"].set_index("arm")
for arm, reg_, sp in [("_RANDOM", 1.762, 0.0), ("G14", 1.454, 0.352),
                      ("NAIVE_LINE@k1", 0.339, 0.799), ("G14@k1", 0.361, 0.772)]:
    check("3", arm + " regret", reg_, pm, mm.loc[arm, "regret"])
    check("3", arm + " spearman", sp, pm, mm.loc[arm, "spearman"])
check("3", "one measurement worth (log units of regret)", 1.09, pm,
      mm.loc["G14", "regret"] - mm.loc["G14@k1", "regret"])
c3 = pd.read_csv(RES / "L3" / "l3c_contrasts.csv")
p3 = rel(RES / "L3" / "l3c_contrasts.csv")
for d, q in zip(["B", "BR", "BQ", "A", "BP"], [0.003, 0.002, -0.003, -0.017, -0.021]):
    check("3", "G14@k1-NAIVE regret " + d, q, p3,
          c3[(c3.design == d) & (c3.comparison == "G14k1_minus_NAIVE_regret")].point.iloc[0])
sp3 = c3[c3.comparison == "G14k1_minus_NAIVE_spearman"]
check("3", "spearman contrast, least negative", -0.022, p3, sp3.point.max())
check("3", "spearman contrast, most negative", -0.027, p3, sp3.point.min())
check("3", "spearman contrast p min", 0.013, p3, sp3.p_two_sided.min())
check("3", "spearman contrast p max", 0.026, p3, sp3.p_two_sided.max())

# ---------------------------------------------------------------- section 4 L1
st = pd.read_csv(RES / "L1" / "stage1_stats.csv")
pst = rel(RES / "L1" / "stage1_stats.csv")


def gs(mod, s):
    return st[(st.model == mod) & (st["set"] == s) & (st.target == "a") &
              (st.value == "slope")].iloc[0]


for mod, s8, s14 in [("NAIVE", 0.392, 0.644), ("SPECIES_NFILLCOL", 0.374, 0.264),
                     ("ELEM", 0.313, 0.396), ("SPECIES_CONST", 0.167, 0.155),
                     ("SPECIES", -0.084, -0.048)]:
    check("4a", "rho(slope,a) " + mod + " S8", s8, pst, gs(mod, "S8").rho)
    check("4a", "rho(slope,a) " + mod + " S14", s14, pst, gs(mod, "S14").rho)
check("4a", "column header n for S8", 62, pst, int(gs("NAIVE", "S8").n), 0)
check("4a", "column header n for S14", 39, pst, int(gs("NAIVE", "S14").n), 0)
check("4a", "n behind the SPECIES_CONST S14 cell (+0.155), header says 39", 39, pst,
      int(gs("SPECIES_CONST", "S14").n), 0)
check("4a", "n behind the SPECIES_CONST S8 cell (+0.167), row label says 19", 19, pst,
      int(gs("SPECIES_CONST", "S8").n), 0)

rj = pd.read_csv(RES / "refutation" / "L1NULL" / "B" / "checkJ_reliability_modelfree.csv")
prj = rel(RES / "refutation" / "L1NULL" / "B" / "checkJ_reliability_modelfree.csv")
sp8 = rj[(rj.model == "SPECIES") & (rj["set"] == "S8")].iloc[0]
check("4b", "split-half reliability SPECIES", -0.272, prj, sp8.split_half_rho)
check("4b", "between-extractant slope sd (eV)", 0.203, prj, sp8.slope_sd)
check("4b", "within-series SE (eV)", 0.329, prj, sp8.jack_se_mean)
check("4b", "jackknife reliability", 0.000, prj, sp8.reliability_jack, 5e-4)
at = pd.read_csv(RES / "refutation" / "L1NULL" / "A" / "A5_attenuation.csv")
pat = rel(RES / "refutation" / "L1NULL" / "A" / "A5_attenuation.csv")
check("4b", "retained fraction, SPECIES", 0.42, pat,
      at[at.model == "SPECIES"].retained_fraction.iloc[0])
inj = pd.read_csv(RES / "refutation" / "L1NULL" / "A" / "A5_injection.csv")
pinj = rel(RES / "refutation" / "L1NULL" / "A" / "A5_injection.csv")
check("4b", "CYCLE_ADD rho S8", 0.29, pinj,
      inj[(inj.model == "CYCLE_ADD") & (inj["set"] == "S8") &
          (inj.k_eV_per_unit_r_per_a == 0.0)].rho.iloc[0])
fc = pd.read_csv(RES / "refutation" / "L1NULL" / "B" / "checkB_fixedcycle_stats.csv")
pfc = rel(RES / "refutation" / "L1NULL" / "B" / "checkB_fixedcycle_stats.csv")
check("4b", "FIXCYC_CONSTNLIGS rho S8", 0.34, pfc,
      fc[(fc.model == "FIXCYC_CONSTNLIGS") & (fc["set"] == "S8")].rho.iloc[0])
check("4b", "FIXCYC_CONSTNLIGS rho S14", 0.45, pfc,
      fc[(fc.model == "FIXCYC_CONSTNLIGS") & (fc["set"] == "S14")].rho.iloc[0])
pw2 = pd.read_csv(RES / "refutation" / "L1NULL" / "B" / "checkA_power.csv")
ppw = rel(RES / "refutation" / "L1NULL" / "B" / "checkA_power.csv")
r19 = pw2[pw2.n == 19].iloc[0]
check("4b", "Fisher CI low at n=19", -0.337, ppw, r19.fisher_lo)
check("4b", "Fisher CI high at n=19", 0.597, ppw, r19.fisher_hi)
apw = pd.read_csv(RES / "refutation" / "L1NULL" / "A" / "A5_power.csv")
check("4b", "power at n=19, truth rho=0.4", 0.36,
      rel(RES / "refutation" / "L1NULL" / "A" / "A5_power.csv"),
      apw[(apw.n == 19) & (apw.rho_true == 0.4)]["power_alpha0.05"].iloc[0])
dec = json.loads((RES / "L1" / "stage1_decision.json").read_text())
pdec = rel(RES / "L1" / "stage1_decision.json")
check("4", "gamma_water minus per-element yardstick (eV)", 0.34, pdec,
      abs(dec["gamma"]["gamma::n_H2O"]["coef_eV"] - dec["yardstick_eV"]["water H2O (O + 2 H)"]))
check("4", "gamma_nitrate minus per-element yardstick (eV): report says the fitted "
           "species energies land within 0.34 eV", 0.34, pdec,
      abs(dec["gamma"]["gamma::n_NO3"]["coef_eV"] - dec["yardstick_eV"]["nitrate NO3 (N + 3 O)"]))
refs = json.loads((RES / "L1" / "reference_species" / "summary.json").read_text())
pref = rel(RES / "L1" / "reference_species" / "summary.json")
check("4", "Stage 2 array tasks", 361, pref, refs["n_species"], 0)
check("4", "Stage 2 CPU-hours", 6.7, pref, refs["expected_cpu_hours"])
check("4", "Stage 2 memory per task (GB)", 2, pref, refs["mem_per_task_GB"], 0)
check("4", "Stage 2 wall minutes at 20 concurrent", 20, pref,
      60 * refs["expected_wall_hours_at_20_concurrent"], 0.5)

# ---------------------------------------------------------------- section 5 L2
gb = pd.read_csv(RES / "L2" / "gate_board.csv")
pgb = rel(RES / "L2" / "gate_board.csv")
tab5 = {
    "FLAT": [0.5885] * 5, "G14": [0.4932, 0.4906, 0.4905, 0.4921, 0.5001],
    "O_CURV": [0.4216, 0.4218, 0.4233, 0.4168, 0.4272],
    "O_CURV_LPO": [0.4634, 0.4638, 0.4642, 0.4599, 0.4714],
}
for arm, vals in tab5.items():
    for d, q in zip(["B", "BR", "BQ", "A", "BP"], vals):
        check("5", arm + " " + d, q, pgb,
              gb[(gb.design == d) & (gb.arm == arm)].macro_mae_extractant.iloc[0])
gc = pd.read_csv(RES / "L2" / "gate_contrasts.csv")
pgc = rel(RES / "L2" / "gate_contrasts.csv")
for d, q in zip(["B", "BR", "BQ", "A", "BP"], [0.0298, 0.0268, 0.0264, 0.0323, 0.0287]):
    check("5", "headroom " + d, q, pgc,
          gc[(gc.design == d) & (gc.comparison == "OCURVLPO_vs_G14")].point.iloc[0])
gbp = gc[(gc.design == "BP") & (gc.comparison == "OCURVLPO_vs_G14")].iloc[0]
check("5", "headroom BP CI low", -0.0013, pgc, gbp.ci95_low)
check("5", "headroom BP CI high", 0.0532, pgc, gbp.ci95_high)
check("5", "headroom BP p", 0.061, pgc, gbp.p_two_sided)
check("5", "bootstrap SE", 0.014, pgc, gbp.bootstrap_se)
check("5", "minimum detectable headroom", 0.039, pgc, gbp.mde_80)
check("1/5", "oracle-noise share of the gen15 in-sample headroom", 0.044, pgc,
      gc[(gc.design == "BP") & (gc.comparison == "OCURV_vs_OCURVLPO")].point.iloc[0])
check("1/5", "gen15 in-sample OCURV headroom", 0.073, pgc,
      gc[(gc.design == "BP") & (gc.comparison == "OCURV_vs_G14")].point.iloc[0])
check("5", "BCa lower bound BP (report prints only the percentile interval)",
      "not printed", pgc, float(gbp.bca_low))

# ---------------------------------------------------------------- section 6 L4
cv = pd.read_csv(RES / "L4" / "curve.csv")
pcv = rel(RES / "L4" / "curve.csv")
tab6a = {
    "RANDOM": [0.626, 0.608, 0.578, 0.553, 0.524, 0.511],
    "MAXMIN": [0.599, 0.613, 0.609, 0.591, 0.526, 0.512],
    "UNCERT": [0.663, 0.636, 0.606, 0.524, 0.520, 0.517],
    "AOPT": [0.570, 0.523, 0.509, 0.492, 0.503, 0.503],
}
for order, vals in tab6a.items():
    for b, q in zip(["6", "9", "12", "16", "20", "24"], vals):
        check("6a", order + " @ " + b, q, pcv,
              cv[(cv.design == "BP") & (cv.order == order) &
                 (cv.budget.astype(str) == b)].macro_mae.iloc[0])
check("6a", "full (FULL, BP)", 0.500, pcv,
      cv[(cv.design == "BP") & (cv.order == "FULL")].macro_mae.iloc[0])
check("6a", "full budget in chemotypes", 29, pcv,
      cv[(cv.design == "BP") & (cv.order == "FULL")].k_actual.iloc[0], 0.5)
ab = pd.read_csv(RES / "L4" / "contrasts_abc.csv")
pab = rel(RES / "L4" / "contrasts_abc.csv")
tab6b = {
    "AOPT_vs_RANDOM": [-0.012, 0.023, -0.003, 0.037, 0.050],
    "AOPT_vs_MAXMIN": [-0.004, 0.042, 0.010, 0.053, 0.058],
    "UNCERT_vs_RANDOM": [-0.031, -0.004, 0.003, -0.018, -0.011],
    "UNCERT_vs_MAXMIN": [-0.023, 0.014, 0.017, -0.002, -0.003],
}
for comp, vals in tab6b.items():
    for d, q in zip(["B", "BR", "BQ", "A", "BP"], vals):
        check("6b", comp + " " + d, q, pab,
              ab[(ab.design == d) & (ab.comparison == comp) &
                 (ab.family == "registered")].point.iloc[0])
cs = pd.read_csv(RES / "refutation" / "L4BP" / "B" / "contrasts_size.csv")
pcs = rel(RES / "refutation" / "L4BP" / "B" / "contrasts_size.csv")
for d, q in zip(["B", "BR", "BQ", "A", "BP"], [0.028, 0.019, 0.026, 0.023, 0.045]):
    check("6", "SIZE_vs_RANDOM " + d, q, pcs,
          cs[(cs.design == d) & (cs.comparison == "SIZE_vs_RANDOM")].point.iloc[0])
asz = cs[(cs.design == "BP") & (cs.comparison == "AOPT_vs_SIZE")].iloc[0]
check("6", "AOPT_vs_SIZE BP", 0.005, pcs, asz.point)
check("6", "AOPT_vs_SIZE BP p", 0.36, pcs, asz.p_two_sided)
osc = pd.read_csv(RES / "refutation" / "L4BP" / "A" / "order_size_correlation.csv")
posc = rel(RES / "refutation" / "L4BP" / "A" / "order_size_correlation.csv")
ao = osc[(osc.design == "BP") & (osc.order == "AOPT")]
check("6", "Spearman(pick position, chemotype CELL count) as worded", -0.63, posc,
      ao.rho_pos_vs_cells.mean())
check("6", "Spearman(pick position, WELL-DETERMINED cell count)", -0.63, posc,
      ao.rho_pos_vs_richcells.mean())
osh = pd.read_csv(RES / "refutation" / "L4BP" / "A" / "order_size_share.csv")
posh = rel(RES / "refutation" / "L4BP" / "A" / "order_size_share.csv")
k6 = osh[(osh.design == "BP") & (osh.k == 6)]
check("6", "AOPT training-cell share at budget 6 (%)", 72, posh,
      100 * k6[k6.order == "AOPT"].cell_share_at_k.mean(), 0.5)
check("6", "RANDOM training-cell share at budget 6 (%)", 18, posh,
      100 * k6[k6.order == "RANDOM"].cell_share_at_k.mean(), 0.5)
rb = pd.read_csv(RES / "L4" / "ranking_pool_bundle.csv")
rl = pd.read_csv(RES / "L4" / "ranking_pool_logk.csv")
check("6", "bundle candidates ranked", 95, rel(RES / "L4" / "ranking_pool_bundle.csv"), len(rb), 0)
check("6", "external logK candidates ranked", 273,
      rel(RES / "L4" / "ranking_pool_logk.csv"), len(rl), 0)
check("6", "distinct values of the AOPT criterion", 43,
      rel(RES / "L4" / "ranking_pool_bundle.csv"), rb.aopt_reduction.nunique(), 0)
check("6", "diglycolamides in the ranked top 20", "dominated",
      rel(RES / "L4" / "ranking_pool_bundle.csv"),
      "dominated" if (rb.head(20).chemotype_frozen == "sc009").sum() >= 15 else "not dominated")
check("6", "FLAT fallbacks fired for MAXMIN (report never mentions the guard)",
      "not printed", pcv, int(cv[cv.order == "MAXMIN"].n_flat_fallbacks.sum()))
check("6", "FLAT fallbacks fired for RANDOM (report never mentions the guard)",
      "not printed", pcv, int(cv[cv.order == "RANDOM"].n_flat_fallbacks.sum()))
check("6", "FLAT fallbacks fired for AOPT (report never mentions the guard)",
      "not printed", pcv, int(cv[cv.order == "AOPT"].n_flat_fallbacks.sum()))

# ---------------------------------------------------------------- section 7 L5
bc = pd.read_csv(RES / "L5" / "board_cov.csv")
pbc = rel(RES / "L5" / "board_cov.csv")
bcbp = bc[bc.design == "BP"].set_index("arm").macro_mae_extractant
tab7a = {
    "G14_POOLED": [0.4052, 0.2238, 0.2009, 0.1674],
    "G14_LW": [0.4052, 0.2214, 0.6539, 34.41],
    "G14_LOWRANK2": [0.4052, 0.2166, 0.1862, 0.1667],
    "G14_LOWRANK3": [0.4052, 0.2191, 0.1895, 0.1683],
    "G14_HIER": [0.4052, 0.2205, 0.1963, 0.1711],
    "MIX6meanPC_POOLED": [0.4052, 0.2236, 0.1974, 0.1604],
    "NAIVE_LINE_POOLED": [None, 0.2310, 0.2251, 0.2243],
}
for arm, vals in tab7a.items():
    for k, q in enumerate(vals):
        if q is None:
            continue
        key = arm + "@doptk" + str(k)
        check("7a", arm + " k=" + str(k), q, pbc,
              float(bcbp.get(key)) if key in bcbp.index else None)
cm = pd.read_csv(RES / "L5" / "contrasts_mix.csv")
pcm = rel(RES / "L5" / "contrasts_mix.csv")
mx = cm[(cm.design == "BP") & (cm.comparison == "MIX6meanPC_vs_POOLED@k3")].iloc[0]
check("7", "MIX6meanPC - POOLED @k3 BP", 0.0070, pcm, mx.point)
check("7", "MIX6meanPC p (report: p < 1e-4)", 0.0001, pcm, mx.p_two_sided, 1e-4)
ccov = pd.read_csv(RES / "L5" / "contrasts_cov.csv")
pccov = rel(RES / "L5" / "contrasts_cov.csv")
reg_cov = ccov[ccov.family == "registered"]
biggest = max(float(reg_cov.point.max()), float(cm[cm.family == "registered"].point.max()))
check("7", "largest positive registered L5 effect (report says MIX6meanPC = +0.0070)",
      0.0070, pccov + " + " + pcm, biggest)
lr2 = ccov[(ccov.design == "BP") & (ccov.comparison == "LOWRANK2_vs_POOLED@k2")].iloc[0]
check("7", "LOWRANK2 - POOLED @k2 BP (registered, never printed in the report)",
      "not printed", pccov, float(lr2.point))
check("7", "LOWRANK2 - POOLED @k2 BP p (never printed)", "not printed",
      pccov, float(lr2.p_two_sided))
cond = pd.read_csv(RES / "L5" / "covariance_conditioning.csv")
pcond = rel(RES / "L5" / "covariance_conditioning.csv")
po = cond[(cond.design == "BP") & (cond.estimator == "POOLED")].iloc[0]
check("7", "n leave-chemotype-out covariance estimates", 430, pcond, po.n_cov_keys, 0)
check("7", "fraction indefinite (%)", 100, pcond, 100 * po.frac_keys_indefinite, 0)
cal = pd.read_csv(RES / "L5" / "calibration_intervals.csv")
pcal = rel(RES / "L5" / "calibration_intervals.csv")
c3_ = cal[(cal.design == "BP") & (cal.arm == "POOLED") & (cal.k == 3)].iloc[0]
c0_ = cal[(cal.design == "BP") & (cal.arm == "POOLED") & (cal.k == 0)].iloc[0]
check("7", "cover90 at k=3 (%)", 98.8, pcal, 100 * c3_.cover90)
check("7", "cover90 at k=0 (%)", 92.6, pcal, 100 * c0_.cover90)
check("7", "RMSE at k=3", 0.248, pcal, c3_.rmse)
check("7", "interval width conservatism factor (median |z| route)", 2.5, pcal,
      0.6745 / c3_.median_abs_z, 0.1)
cd = pd.read_csv(RES / "L5" / "calibration_direction.csv")
pcd = rel(RES / "L5" / "calibration_direction.csv")
cdbp = cd[cd.design == "BP"].set_index("arm")
check("7", "Brier PLATT BP", 0.255, pcd, cdbp.loc["PLATT", "macro_brier"])
check("7", "Brier RAW BP", 0.167, pcd, cdbp.loc["RAW", "macro_brier"])
check("7", "Brier CONST BP", 0.313, pcd, cdbp.loc["CONST", "macro_brier"])
check("7", "ECE RAW BP", 0.139, pcd, cdbp.loc["RAW", "ece10"])
vfy = json.loads((RES / "L5" / "verify.json").read_text())
pvf = rel(RES / "L5" / "verify.json")
check("7", "G14@doptk3 locked", "0.1700393901", pvf, round(vfy["G14@doptk3_l5"], 10))
check("7", "G14@doptk1 locked", "0.2313450953", pvf, round(vfy["G14@doptk1_l5"], 10))
check("7", "reproduction difference quoted as 0.0e+00", 0.0, pvf,
      vfy["max_absdiff_vs_locked_board"], 0)
mtr = json.loads((RES / "L5" / "mixture_trainroute_repro.json").read_text())
check("7", "locked training-route modes reproduced", 11,
      rel(RES / "L5" / "mixture_trainroute_repro.json"), len(mtr["BP"]["modes"]), 0)

# ---------------------------------------------------------------- section 8 L6
kk = pd.read_csv(RES / "L6_cohort_audit" / "counterfactual_kish.csv")
pkk = rel(RES / "L6_cohort_audit" / "counterfactual_kish.csv")
sm = json.loads((RES / "L6_cohort_audit" / "summary.json").read_text())
psm = rel(RES / "L6_cohort_audit" / "summary.json")
check("8", "frozen Kish n_eff", 11.67, psm, sm["kish_neff_frozen_by_extractant"])
for scen, ext, kish, ratio in [
        ("one compound per absent chemotype (cheapest batch)", 143, 27.38, 2.35),
        ("only those in chemotypes absent from the cohort", 150, 28.55, 2.45),
        ("all 100 single-lanthanide compounds", 190, 12.27, 1.05)]:
    r = kk[kk.scenario == scen].iloc[0]
    check("8", "extractants, " + scen, ext, pkk, r.extractants, 0)
    check("8", "Kish, " + scen, kish, pkk, r.kish_neff_by_extractant)
    check("8", "ratio, " + scen, ratio, pkk, r.kish_ratio_vs_frozen)
ee = pd.read_csv(RES / "L6_cohort_audit" / "excluded_extractants.csv")
pee = rel(RES / "L6_cohort_audit" / "excluded_extractants.csv")
ex = ee[~ee["kept"]]
kept_ct = set(ee[ee["kept"]]["chemotype"].dropna())
check("8", "excluded compounds", 100, pee, len(ex), 0)
check("8", "chemotypes carried by the excluded", 60, pee, ex.chemotype.nunique(), 0)
check("8", "of those absent from the cohort", 53, pee,
      len(set(ex.chemotype.dropna()) - kept_ct), 0)
check("8", "excluded at Tanimoto < 0.7", 77, psm, sm["n_excluded_T_lt_0.7"], 0)
check("8", "diglycolamides among the excluded", 28, pee, int((ex.chemotype == "sc009").sum()), 0)
check("8", "report's '91 of the 100 come from one publication' vs "
           "excluded extractants appearing in exactly ONE publication each",
      91, pee, int((ex.n_publications == 1).sum()), 0)
check("8", "report's '91 of the 100 come from one publication' vs the largest "
           "single publication's actual share of the 100",
      91, "gen6 provenance joined to the frozen bundle (recomputed by this audit)", 52, 0)
rx = pd.read_csv(RES / "L6_cohort_audit" / "relaxations.csv")
prx = rel(RES / "L6_cohort_audit" / "relaxations.csv")
check("8", "relaxations tried (excluding FROZEN)", 7, prx, len(rx) - 1, 0)
check("8", "max chemotypes added by any relaxation", 0, prx,
      int(rx.d_chemotypes_vs_frozen.fillna(0).max()), 0)
for lbl, cells, ext in [("exact", 521, 90), ("relaxed", 509, 90), ("series", 597, 89)]:
    key = {"exact": "FROZEN", "relaxed": "key_mode_relaxed", "series": "key_mode_series"}[lbl]
    r = rx[rx.relaxation == key].iloc[0]
    check("8", lbl + " cells", cells, prx, r.cells, 0)
    check("8", lbl + " extractants", ext, prx, r.extractants, 0)
check("8", "frozen fingerprint", "4c3c6628ea0be949", prx,
      rx[rx.relaxation == "FROZEN"].fingerprint.iloc[0])
check("8", "bundle structures re-clustered", 190, psm, sm["n_bundle_extractants"], 0)

# ---------------------------------------------------------------- section 10 defects
rp = pd.read_csv(RES / "L1" / "stage1_reproduction.csv")
prp = rel(RES / "L1" / "stage1_reproduction.csv")
row = rp[(rp.subset == "all cohort extractants") & (rp.construction == "A") &
         (rp.target == "a")].iloc[0]
check("10.4", "reproducible gen15 section-3 row A: n", 79, prp, row.n_gen16, 0)
check("10.4", "reproducible gen15 section-3 row A: rho", 0.104, prp, row.rho_gen16)
row = rp[(rp.subset == "all cohort extractants") & (rp.construction == "B") &
         (rp.target == "a")].iloc[0]
check("10.4", "reproducible gen15 section-3 row B: n", 81, prp, row.n_gen16, 0)
check("10.4", "reproducible gen15 section-3 row B: rho", 0.203, prp, row.rho_gen16)
check("10.5", "geometry-complete series", 41, prp,
      rp[(rp.subset.str.contains("GEOMETRY")) & (rp.construction == "naive") &
         (rp.target == "a")].n_gen16.iloc[0], 0)
check("10.5", "energy-complete series", 39, prp,
      rp[(rp.subset.str.contains("ENERGY")) & (rp.construction == "naive") &
         (rp.target == "a")].n_gen16.iloc[0], 0)
check("10.6", "complexes verified", 1155, pref, refs["distinct_composition_combinations"], 0)
line330 = (ROOT / "gen13_separation" / "gen13sep" / "arms_stage2.py").read_text(
    encoding="utf-8", errors="replace").splitlines()[329]
check("10.8", "arms_stage2.py:330 hashes bytes", "hash(row.tobytes())",
      "gen13_separation/gen13sep/arms_stage2.py:330",
      "hash(row.tobytes())" if "hash(" in line330 and "tobytes" in line330 else line330.strip())

# ---------------------------------------------------------------- section 12 descriptors
cds = pd.read_csv(RES / "refutation" / "L1NULL" / "B" / "checkD_composition_stats.csv")
pcds = rel(RES / "refutation" / "L1NULL" / "B" / "checkD_composition_stats.csv")
s = cds[(cds.descriptor == "steppos_n_H2O") & (cds["set"] == "S14")].iloc[0]
check("12", "steppos_n_H2O S14 rho", 0.83, pcds, s.rho_a)
check("12", "steppos_n_H2O S14 n", 27, pcds, s.n, 0)
s8_ = cds[(cds.descriptor == "steppos_n_H2O") & (cds["set"] == "S8")].iloc[0]
check("12", "steppos_n_H2O S8 rho", 0.78, pcds, s8_.rho_a)
check("12", "steppos_n_H2O S8 n", 29, pcds, s8_.n, 0)
check("12", "family-wise permutation bar for max |rho|", 0.60, pdec,
      dec["permutation_bar_a"]["null_p95_max_abs_rho"], 0.01)

# ---------------------------------------------------------------- confirmation seeds
seeds = json.loads((RES / "confirmation" / "CONFIRMATION_RUN_ONCE.json").read_text())["seeds"]
check("CONFIRMATION", "seed commitment sha256",
      "5a30455bc67d364aa3e65f6d3d28bd4b6fcf3301fe979ca4a1be4a9547de76a9",
      rel(RES / "confirmation" / "CONFIRMATION_RUN_ONCE.json"),
      hashlib.sha256(json.dumps(sorted(seeds)).encode()).hexdigest())

df = pd.DataFrame(rows)
df.to_csv(OUT / "number_check.csv", index=False)
print(df.verdict.value_counts().to_dict())
print(df[df.verdict != "match"].to_string())
