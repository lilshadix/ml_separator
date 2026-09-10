"""Verification of the Phase-6 corrections to gen16_leads/DECISION_REPORT.md.

Read-only over the committed artefacts; writes results/audit/verify/verification.csv.
Run:  .venv/Scripts/python.exe gen16_leads/scripts/audit_verify.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "audit", "verify")
os.makedirs(OUT, exist_ok=True)

rows: list[dict] = []


def rec(item, quantity, report_value, artefact, artefact_value, tol=None):
    verdict = "info"
    if report_value is not None and artefact_value is not None and tol is not None:
        try:
            d = abs(float(report_value) - float(artefact_value))
            verdict = "match" if d <= tol else "MISMATCH"
        except (TypeError, ValueError):
            verdict = "match" if str(report_value) == str(artefact_value) else "MISMATCH"
    elif report_value is not None and artefact_value is not None:
        verdict = "match" if str(report_value) == str(artefact_value) else "MISMATCH"
    rows.append(dict(item=item, quantity=quantity, report_value=report_value,
                     artefact=artefact, artefact_value=artefact_value, verdict=verdict))
    return verdict


def p(*parts):
    return os.path.join(RES, *parts)


# ---------------------------------------------------------------- items 1, 3
bar = pd.read_csv(p("refutation", "L1NULL", "A", "A6_descriptor_bar.csv"))
chk = pd.read_csv(p("refutation", "L1NULL", "A", "A4_checks.csv"))

g = bar[(bar.descriptor == "slope_CYCLE_ADD")].set_index("set")
rec(1, "CYCLE_ADD rho S8 (n=62)", 0.2906, "A6_descriptor_bar", g.loc["S8", "rho_vs_a"], 5e-5)
rec(1, "CYCLE_ADD n S8", 62, "A6_descriptor_bar", int(g.loc["S8", "n"]), 0)
rec(1, "CYCLE_ADD rho S8_no_sc009", -0.0258, "A6_descriptor_bar",
    g.loc["S8_no_sc009", "rho_vs_a"], 5e-5)
rec(1, "CYCLE_ADD n S8_no_sc009", 43, "A6_descriptor_bar", int(g.loc["S8_no_sc009", "n"]), 0)
rec(1, "CYCLE_ADD p S8_no_sc009", 0.8694, "A6_descriptor_bar", g.loc["S8_no_sc009", "p"], 5e-5)

c = chk[chk.model == "CYCLE_ADD"].set_index("check")
rec(1, "CYCLE_ADD const-composition rho (n=19)", 0.0491, "A4_checks", c.loc["4_const19", "rho"], 5e-5)
rec(1, "CYCLE_ADD const-composition n", 19, "A4_checks", int(c.loc["4_const19", "n"]), 0)
rec(1, "CYCLE_ADD const-composition p", 0.8417, "A4_checks", c.loc["4_const19", "p"], 5e-5)
rec(1, "CYCLE_ADD varying-composition rho (n=43)", 0.4761, "A4_checks",
    c.loc["4_varying43", "rho"], 5e-5)
rec(1, "CYCLE_ADD varying-composition n", 43, "A4_checks", int(c.loc["4_varying43", "n"]), 0)
rec(1, "CYCLE_ADD varying-composition p", 0.0012, "A4_checks", c.loc["4_varying43", "p"], 5e-5)

fx = pd.read_csv(p("refutation", "L1NULL", "B", "checkB_fixedcycle_stats.csv"))
f1 = fx[(fx.model == "FIXCYC_CONSTNLIGS") & (fx.set == "S8")].iloc[0]
rec(1, "FIXCYC_CONSTNLIGS S8 rho (Table 4c)", 0.339, "checkB_fixedcycle_stats", f1.rho, 5e-4)
rec(1, "FIXCYC_CONSTNLIGS S8 p", 0.007, "checkB_fixedcycle_stats", f1.p, 5e-4)
rec(1, "FIXCYC_CONSTNLIGS S8 loco_min (report 'LOCO min +0.037')", 0.037,
    "checkB_fixedcycle_stats", f1.loco_min, 5e-4)
f2 = fx[(fx.model == "FIXCYC_ALL") & (fx.set == "S8")].iloc[0]
rec(1, "FIXCYC_ALL S8 rho (Table 4c)", 0.290, "checkB_fixedcycle_stats", f2.rho, 5e-4)
rec(1, "FIXCYC_ALL S8 p", 0.022, "checkB_fixedcycle_stats", f2.p, 5e-4)
rec(1, "FIXCYC_ALL S8 loco_min (report 'LOCO min -0.026')", -0.026,
    "checkB_fixedcycle_stats", f2.loco_min, 5e-4)

comp = bar[bar.descriptor == "competitor"].set_index("set")
for s, n, r in [("S8", 61, -0.4158), ("S14", 39, -0.4571),
                ("all82", 80, -0.5300), ("S8_no_sc009", 42, -0.2800)]:
    rec(3, f"competitor rho {s}", r, "A6_descriptor_bar", comp.loc[s, "rho_vs_a"], 5e-5)
    rec(3, f"competitor n {s}", n, "A6_descriptor_bar", int(comp.loc[s, "n"]), 0)
# does the competitor really beat every xTB estimator in Table 4a?
t4a = {"NAIVE_S8": 0.392, "NAIVE_S14": 0.644, "SPECIES_NFILLCOL_S8": 0.374,
       "SPECIES_NFILLCOL_S14": 0.264, "ELEM_S8": 0.313, "ELEM_S14": 0.396,
       "SPECIES_CONST_S8": 0.167, "SPECIES_CONST_S14": 0.155,
       "SPECIES_S8": -0.084, "SPECIES_S14": -0.048}
maxx = max(abs(v) for v in t4a.values())
rec(3, "max |rho| among Table 4a xTB estimators", None, "DECISION_REPORT Table 4a", maxx)
rec(3, "max |rho| of the competitor over the sets quoted", None, "A6_descriptor_bar",
    max(abs(comp.loc[s, "rho_vs_a"]) for s in ["S8", "S14", "all82"]))
rec(3, "claim: competitor larger in magnitude than EVERY xTB estimator in Table 4a",
    "True", "computed", str(max(abs(comp.loc[s, 'rho_vs_a']) for s in ['S8', 'S14', 'all82']) > maxx))

# ---------------------------------------------------------------- item 2
pn = pd.read_csv(p("L1", "stage1_perm_null.csv")).set_index("target")
rec(2, "perm null p95 max|rho| target a", 0.608791, "stage1_perm_null",
    pn.loc["a", "null_p95_max_abs_rho"], 5e-6)
rec(2, "observed max|rho| target a", 0.601651, "stage1_perm_null",
    pn.loc["a", "observed_max_abs_rho"], 5e-6)
rec(2, "observed argmax target a", "slope__NAIVE__S14", "stage1_perm_null",
    pn.loc["a", "observed_argmax"])
rec(2, "familywise p target a", 0.0565, "stage1_perm_null", pn.loc["a", "familywise_p"], 1e-9)
rec(2, "family size", 12, "stage1_perm_null", int(pn.loc["a", "family_size"]), 0)
rec(2, "permutation unit count (chemotype-level)", None, "stage1_perm_null",
    int(pn.loc["a", "n_chemotype_units"]))
rec(2, "same family, target b: observed vs bar", None, "stage1_perm_null",
    f"obs {pn.loc['b','observed_max_abs_rho']:.4f} > bar "
    f"{pn.loc['b','null_p95_max_abs_rho']:.4f}, familywise p {pn.loc['b','familywise_p']}")
c1 = pd.read_csv(p("L1", "contrasts_stage1.csv"))
if "rho_chemotype_means" in c1.columns:
    m = c1[(c1.model == "NAIVE") & (c1["set"] == "S14") & (c1.target == "a")]
    if len(m):
        rec(2, "extractant-level rho NAIVE/S14/a", 0.644, "contrasts_stage1",
            float(m.iloc[0]["point"]), 5e-4)
        rec(2, "chemotype-level rho NAIVE/S14/a (the perm statistic)", 0.6017,
            "contrasts_stage1", float(m.iloc[0]["rho_chemotype_means"]), 5e-4)

# ---------------------------------------------------------------- item 4
dec = json.load(open(p("L1", "stage1_decision.json")))
gh2o = dec["gamma"]["gamma::n_H2O"]["coef_eV"]
gno3 = dec["gamma"]["gamma::n_NO3"]["coef_eV"]
yh2o = dec["yardstick_eV"]["water H2O (O + 2 H)"]
yno3 = dec["yardstick_eV"]["nitrate NO3 (N + 3 O)"]
rec(4, "gamma n_H2O", -138.88, "stage1_decision.json", gh2o, 5e-3)
rec(4, "yardstick H2O", -139.22, "stage1_decision.json", yh2o, 5e-3)
rec(4, "|gamma_H2O - yardstick|", 0.34, "computed", abs(gh2o - yh2o), 5e-3)
rec(4, "gamma n_NO3", -431.97, "stage1_decision.json", gno3, 5e-3)
rec(4, "yardstick NO3", -414.21, "stage1_decision.json", yno3, 5e-3)
rec(4, "|gamma_NO3 - yardstick|", 17.8, "computed", abs(gno3 - yno3), 5e-2)

# ---------------------------------------------------------------- item 5
fp = pd.read_csv(p("refutation", "L3A", "B", "refute_l3a_B_foldpure.csv"))
b = fp[(fp.design == "BP") & (fp.regime == "fold_pure")].iloc[0]
bp_pool = fp[(fp.design == "BP") & (fp.regime == "pooled_rebuilt")].iloc[0]
rec(5, "fold-pure saved BP", 0.906, "foldpure.csv", b.point, 5e-4)
rec(5, "fold-pure E_random BP", 5.52, "foldpure.csv", b.e_random, 5e-3)
rec(5, "fold-pure saved fraction BP (%)", 16.4, "foldpure.csv", 100 * b.saved_frac, 5e-2)
rec(5, "fold-pure share of perfect-sign ceiling (%)", 49.8, "foldpure.csv",
    100 * b.g14_over_ceiling, 5e-2)
cl = pd.read_csv(p("refutation", "L3A", "B", "refute_l3a_B_ceiling.csv"))
cp = cl[(cl.design == "BP") & (cl.regime == "pooled")].iloc[0]
rec(5, "pooled share of perfect-sign ceiling (%)", 47.3, "ceiling.csv",
    100 * cp.G14_over_ceiling, 5e-2)
rec(5, "fold-pure as % of headline magnitude", 46, "computed",
    100 * b.point / bp_pool.point, 0.5)
rec(5, "fold-pure n_tasks (report/CONFIRMATION say 455)", 455, "foldpure.csv",
    int(b.n_tasks), 0)
rec(5, "foldpure_notes.json content", None, "refute_l3a_B_foldpure_notes.json",
    open(p("refutation", "L3A", "B", "refute_l3a_B_foldpure_notes.json")).read().strip())
cw = cl[(cl.design == "BP") & (cl.regime == "within_publication")].iloc[0]
rec(5, "within-publication ceiling share (%)", 1.1, "ceiling.csv",
    100 * cw.G14_over_ceiling, 5e-2)
rec(5, "within-publication perfect-sign ceiling", 0.414, "ceiling.csv", cw.CEILING, 5e-4)

# ---------------------------------------------------------------- item 6
cov = pd.read_csv(p("L5", "contrasts_cov.csv"))
lr = cov[cov.comparison == "LOWRANK2_vs_POOLED@k2"].set_index("design")
rec(6, "LOWRANK2@k2 BP point", 0.0151, "contrasts_cov", lr.loc["BP", "point"], 5e-5)
rec(6, "LOWRANK2@k2 BP ci low", 0.0067, "contrasts_cov", lr.loc["BP", "ci95_low"], 5e-5)
rec(6, "LOWRANK2@k2 BP ci high", 0.0232, "contrasts_cov", lr.loc["BP", "ci95_high"], 5e-5)
rec(6, "LOWRANK2@k2 BP p", 0.0020, "contrasts_cov", lr.loc["BP", "p_two_sided"], 1e-9)
rec(6, "LOWRANK2@k2 BP seeds", 5, "contrasts_cov", int(lr.loc["BP", "seeds_positive"]), 0)
rec(6, "LOWRANK2@k2 BP LOCO stable", "True", "contrasts_cov", str(lr.loc["BP", "loco_sign_stable"]))
rec(6, "LOWRANK2@k2 design A p", 0.3362, "contrasts_cov", lr.loc["A", "p_two_sided"], 1e-9)
rec(6, "LOWRANK2@k2 same sign in all five", "True", "contrasts_cov",
    str(bool((lr["point"] > 0).all())))
rec(6, "LOWRANK2@k2 significant (p<0.05) in all five", "n/a", "contrasts_cov",
    str(bool((lr["p_two_sided"] < 0.05).all())))
lr3 = cov[cov.comparison == "LOWRANK2_vs_POOLED@k3"].set_index("design")
rec(6, "LOWRANK2@k3 BP point", 0.0003, "contrasts_cov", lr3.loc["BP", "point"], 5e-5)
rec(6, "LOWRANK2@k3 BP seeds", 4, "contrasts_cov", int(lr3.loc["BP", "seeds_positive"]), 0)
rec(6, "LOWRANK2@k3 BP LOCO stable", "False", "contrasts_cov",
    str(lr3.loc["BP", "loco_sign_stable"]))
mix = pd.read_csv(p("L5", "contrasts_mix.csv"))
mx = mix[mix.comparison == "MIX6meanPC_vs_POOLED@k3"].set_index("design")
rec(6, "MIX6meanPC@k3 BP point", 0.0070, "contrasts_mix", mx.loc["BP", "point"], 5e-5)
rec(6, "MIX6meanPC@k3 BP p", 0.0, "contrasts_mix", mx.loc["BP", "p_two_sided"], 1e-9)

# ---------------------------------------------------------------- item 7
files = sorted(glob.glob(os.path.join(RES, "**", "*contrast*.csv"), recursive=True))
keep, dry = [], []
for f in files:
    rel = os.path.relpath(f, RES).replace("\\", "/")
    if rel.split("/")[0] in ("refutation", "confirmation", "anchors"):
        continue
    (dry if "/_dry/" in rel else keep).append(f)


def load(fs):
    out = []
    for f in fs:
        d = pd.read_csv(f)
        d["_src"] = os.path.relpath(f, RES).replace("\\", "/")
        out.append(d)
    return pd.concat(out, ignore_index=True)


allc = load(keep + dry)
dryc = load(dry)
nonc = load(keep)
rec(7, "total contrast rows", 940, "results/**/*contrast*.csv", len(allc), 0)
rec(7, "registered rows", 126, "computed", int((allc.family == "registered").sum()), 0)
rec(7, "exploratory rows", 814, "computed", int((allc.family == "exploratory").sum()), 0)
rec(7, "_dry rows", 80, "results/L4/_dry", len(dryc), 0)
rec(7, "_dry registered rows", 4, "results/L4/_dry", int((dryc.family == "registered").sum()), 0)
rec(7, "rows excluding _dry", 860, "computed", len(nonc), 0)
rec(7, "registered excluding _dry", 122, "computed", int((nonc.family == "registered").sum()), 0)
rec(7, "exploratory excluding _dry", 738, "computed", int((nonc.family == "exploratory").sum()), 0)
by_lead = {}
for f in keep + dry:
    rel = os.path.relpath(f, RES).replace("\\", "/")
    lead = rel.split("/")[0]
    by_lead[lead] = by_lead.get(lead, 0) + len(pd.read_csv(f))
for k, v in sorted(by_lead.items()):
    rec(7, f"rows in {k}", None, "computed", v)

# BH within the registered family of 126
reg = allc[allc.family == "registered"].copy()
# p_two_sided is the chemotype-blocked bootstrap p for L3 and the two-sided bootstrap p
# elsewhere; it is the column whose values the report's BH table quotes as "raw p".
reg["praw"] = reg["p_two_sided"].astype(float)
reg = reg.reset_index(drop=True)
pv = reg["praw"].astype(float).values
n = len(pv)
order = np.argsort(pv, kind="mergesort")
q = np.empty(n)
prev = 1.0
for rank in range(n - 1, -1, -1):
    i = order[rank]
    val = pv[i] * n / (rank + 1)
    prev = min(prev, val)
    q[i] = prev
reg["q_bh126"] = q
reg.to_csv(os.path.join(OUT, "bh_registered_126.csv"), index=False)


def qrange(mask, label, lo, hi):
    s = reg[mask]
    if not len(s):
        rec(7, f"BH q {label}", f"{lo}-{hi}", "no rows matched", None)
        return
    rec(7, f"BH q {label} min", lo, "recomputed BH over 126", s.q_bh126.min(), 6e-3)
    rec(7, f"BH q {label} max", hi, "recomputed BH over 126", s.q_bh126.max(), 6e-3)
    rec(7, f"raw p {label}", None, "computed",
        f"{s.praw.min():.4g}-{s.praw.max():.4g}")


qrange(reg.comparison.astype(str).str.contains("G14_saved_vs_0"), "L3a G14_saved_vs_0", 0.020, 0.042)
qrange(reg.comparison.astype(str) == "MIX6meanPC_vs_POOLED@k3", "MIX6meanPC@k3", 0.0, 0.001)
qrange(reg.comparison.astype(str) == "LOWRANK2_vs_POOLED@k2", "LOWRANK2@k2", 0.011, 0.547)
qrange(reg.comparison.astype(str) == "OCURVLPO_vs_G14", "L2 gate", 0.080, 0.246)
qrange(reg.comparison.astype(str).str.contains("AOPT_vs_RANDOM"), "L4 AOPT", 0.0, 0.993)
l3c = reg[reg["_src"].str.contains("l3c")]
for v in sorted(set(l3c["value"].astype(str))):
    s = l3c[l3c["value"].astype(str) == v]
    rec(7, f"BH q L3c value={v}", None, "recomputed BH over 126",
        f"{s.q_bh126.min():.4f}-{s.q_bh126.max():.4f} (raw {s.praw.min():.4g}-{s.praw.max():.4g})")

# BH over all 940
allc2 = allc.copy()
allc2["praw"] = allc2["p_two_sided"].astype(float)
pv = allc2["praw"].astype(float).fillna(1.0).values
n = len(pv)
order = np.argsort(pv, kind="mergesort")
q = np.empty(n)
prev = 1.0
for rank in range(n - 1, -1, -1):
    i = order[rank]
    prev = min(prev, pv[i] * n / (rank + 1))
    q[i] = prev
allc2["q_bh940"] = q
sel = allc2[(allc2.comparison.astype(str).str.contains("G14_saved_vs_0")) & (allc2.design == "BP")]
rec(7, "BH q over all 940, L3a BP", 0.025, "recomputed BH over 940",
    float(sel.q_bh940.iloc[0]) if len(sel) else None, 6e-3)
sel126 = reg[(reg.comparison.astype(str).str.contains("G14_saved_vs_0")) & (reg.design == "BP")]
rec(7, "BH q over 126, L3a BP (CONFIRMATION says 0.039)", 0.039, "recomputed BH over 126",
    float(sel126.q_bh126.iloc[0]) if len(sel126) else None, 6e-3)
l1reg = reg[reg["_src"].str.startswith("L1/")]
rec(7, "L1 registered rows", 2, "computed", len(l1reg), 0)
if len(l1reg):
    rec(7, "L1 registered q within its own family (report says 0.518)", 0.518,
        "contrasts_stage1 p_bh_within_family",
        float(pd.read_csv(p("L1", "contrasts_stage1.csv")).query("family=='registered'")
              ["p_bh_within_family"].max()), 6e-3)
allc2.to_csv(os.path.join(OUT, "bh_all_940.csv"), index=False)

# ---------------------------------------------------------------- item 8
ex = pd.read_csv(p("L6_cohort_audit", "excluded_extractants.csv"))
exc = ex[~ex.kept.astype(bool)]
rec(8, "excluded compounds", 100, "excluded_extractants", len(exc), 0)
rec(8, "excluded with n_publications == 1", 91, "excluded_extractants",
    int((exc.n_publications == 1).sum()), 0)
pub_ok = False
try:
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.dirname(ROOT))
    from gen16_leads.gen16 import cohort_audit as ca  # type: ignore
    raw = ca.load_bundle_raw()
    sub = raw[raw["canonical_smiles"].isin(set(exc.extractant))]
    npub = sub["publication_id"].nunique()
    top = sub.groupby("publication_id")["canonical_smiles"].nunique().sort_values()
    rec(8, "distinct publications spanned by the 100", 23, "bundle+gen6 provenance", int(npub), 0)
    rec(8, "largest publication's share of the 100", 52, "bundle+gen6 provenance",
        int(top.iloc[-1]), 0)
    pub_ok = True
except Exception as e:  # pragma: no cover
    rec(8, "distinct publications / largest share", "23 / 52",
        "NOT REPRODUCIBLE from committed artefacts", f"error: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- item 9
gt = pd.read_csv(p("L2", "gate_contrasts.csv"))
g2 = gt[(gt.design == "BP") & (gt.comparison == "OCURVLPO_vs_G14")].iloc[0]
rec(9, "L2 BP bca_low", 5.7e-06, "gate_contrasts", g2.bca_low, 5e-8)
rec(9, "L2 BP bca_high (report +0.0544)", 0.0544, "gate_contrasts", g2.bca_high, 5e-5)
rec(9, "L2 BP percentile ci low", -0.0013, "gate_contrasts", g2.ci95_low, 5e-5)
rec(9, "L2 BP percentile ci high", 0.0532, "gate_contrasts", g2.ci95_high, 5e-5)
rec(9, "L2 BP point", 0.0287, "gate_contrasts", g2.point, 5e-5)
rec(9, "L2 BP p", 0.061, "gate_contrasts", g2.p_two_sided, 5e-4)
rec(9, "L2 BP mde_80", 0.0386, "gate_contrasts", g2.mde_80, 5e-5)
rec(9, "L2 BP units improved", 61, "gate_contrasts", int(g2.units_improved), 0)
reg2 = gt[gt.comparison == "OCURVLPO_vs_G14"].set_index("design")
rec(9, "designs whose percentile CI excludes zero", "A", "gate_contrasts",
    ",".join(sorted(reg2.index[reg2.ci95_low > 0])))
rec(9, "designs whose BCa CI excludes zero", None, "gate_contrasts",
    ",".join(sorted(reg2.index[reg2.bca_low > 0])))

# ---------------------------------------------------------------- item 10
cal = pd.read_csv(p("L5", "calibration_direction.csv"))
cb = cal[cal.design == "BP"].set_index("arm")
rec(10, "BP RAW brier", 0.1666, "calibration_direction", cb.loc["RAW", "macro_brier"], 5e-4)
rec(10, "BP RAW ece", 0.1393, "calibration_direction", cb.loc["RAW", "ece10"], 5e-4)
rec(10, "BP RAW macro accuracy", 0.821, "calibration_direction",
    cb.loc["RAW", "macro_accuracy"], 5e-4)
rec(10, "BP PLATT brier", 0.2550, "calibration_direction", cb.loc["PLATT", "macro_brier"], 5e-4)
rec(10, "BP PLATT ece", 0.3012, "calibration_direction", cb.loc["PLATT", "ece10"], 5e-4)
rec(10, "BP CONST brier", 0.3128, "calibration_direction", cb.loc["CONST", "macro_brier"], 5e-4)
w = cal.pivot(index="design", columns="arm", values="ece10")
rec(10, "PLATT ece worse than RAW in all five designs", "True", "calibration_direction",
    str(bool((w["PLATT"] > w["RAW"]).all())))
wb = cal.pivot(index="design", columns="arm", values="macro_brier")
rec(10, "RAW brier beats CONST in all five designs", "True", "calibration_direction",
    str(bool((wb["RAW"] < wb["CONST"]).all())))

# ------------------------------------------------------- items 11, 12 (text)
DOC = os.path.dirname(RES)


def txt(name):
    return open(os.path.join(DOC, name), encoding="utf-8").read()


dr, cf, rl, st = txt("DECISION_REPORT.md"), txt("CONFIRMATION.md"), \
    txt("REFUTATION_LOG.md"), txt("STATUS.md")
head = dr[dr.index("## Headline"):dr.index("## 1. What was run")]
rec(11, "headline says 'Every other lead closed' while L1 is undecided", "absent",
    "DECISION_REPORT headline", "present" if "Every other lead closed" in head else "absent")
rec(11, "'L1 having closed' still asserted in body", "absent", "DECISION_REPORT L228",
    "present" if "L1 having closed" in dr else "absent")
rec(11, "'not closed, not positive' still used as a verdict", "absent",
    "STATUS Phase-4 paragraph",
    "present" if "not\nclosed, not positive" in st or "not closed, not positive" in st
    else "absent")
rec(11, "fold-purity dent carried in the headline (STATUS claims it is)", "present",
    "DECISION_REPORT headline block",
    "present" if ("16.4" in head or "fold-pure" in head or "5.52" in head) else "absent")
rec(11, "guardrail 1 arithmetic: +1.91 of 6.91 as a percentage", 28.8, "computed",
    round(100 * 1.909 / 6.906, 1), 0.05)
rec(11, "L2 'underpowered by design' present in headline/§5/§9", "3", "DECISION_REPORT",
    str(dr.count("underpowered by design") + dr.count("underpowered to detect")))
rec(12, "REFUTATION_LOG §2 item 4 -> '§4 of the lead's own report'", "decision rule",
    "results/L3/L3_REPORT.md §4", "'4. Loop-validity checks' (no decision-rule text)")
rec(12, "CONFIRMATION §2 paraphrase of PRE_REGISTRATION §4",
    "passed its registered decision rule", "PRE_REGISTRATION.md §4", "passed P1")
rec(2, "rows in L1's registered family clearing the bar, target b",
    "0 (report: 'nothing clears')", "contrasts_stage1 clears_familywise_bar",
    "1 (NAIVE slope S14 vs b, chemotype rho 0.7503 > bar 0.6044, familywise p 0.003)")
rec(7, "report itemisation 'L1 90 rows (4 models x 3 sets x 3 targets)'", 90, "computed",
    "4*3*3 = 36; the file is 5 models x 3 sets x 3 targets x 2 coefficients = 90")

# ---------------------------------------------------------------- write
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "verification.csv"), index=False)
print(out.to_string())
print()
print("MISMATCH rows:", int((out.verdict == "MISMATCH").sum()))
