#!/usr/bin/env python
"""Block-A decision evaluator (PROTOCOL sections 6 and 10; sealed with amendment A1).  Reads only stored row-level
predictions (predictions/<ID>_<design>.parquet); computes nothing new from a model.

For every block-A slot column that exists (S01, S02, S03 from their own runs; S04 draws and S21 draws from theirs):
  * dR2 and dbalanced-MAE vs B1 and vs B2 on identical rows, per design (seed means);
  * the harm rule vs B1 (and vs B2 for criterion 2) on BPG: a declared subgroup's row MAE worse by > 0.010 in the seed
    mean AND by > 0.010 in >= 4 of 5 seeds.  Subgroups: the 12 pair types, sc009 / non-sc009, donor-defined DGA /
    non-DGA, one end read / both hidden;
  * criterion 1 = dR2 >= 0.01 and dbalMAE >= 0.005 vs B1 on BPG AND on BP, and no harm vs B1;
  * criterion 2 vs B2 likewise;
  * routing: S03 / S04 run iff S02 passes criterion 1; S21's route = among S01 / S02 passing criterion 1, the larger
    BPG dR2 vs B1, ties (< 0.005) to fewer parameters (S01) then the lower slot number;
  * control rules, each required on BPG and on BP:  route - max(S21 draws) > 0 and mean(S21 draws) < 0.5 * route
    (dR2 vs B1);  S02 - S03 >= 0.005 and S02 - max(S04 draws) > 0;
  * labels: a slot passing criterion 1 = 'exploratory gain'; + criterion 2 + its S21 rule = 'donor-conditioned
    (exploratory)'; S02 + electronic rules = 'electronic-specific (exploratory)'; failing a control downgrades the
    attribution, not the criterion-1 eligibility; S03 / S04 / S21 are controls and never finalist-eligible.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/report/r4_decide.py
writes results/blockA_decision.json and results/blockA_decision.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R4))
KEY = ["split_seed", "fold", "cell_id", "A", "B"]
DESIGNS = ("BPG", "BP")
N_PARAMS = {"S01": 7, "S02": 15}
PAIRS = ["La/Ce", "Ce/Pr", "Pr/Nd", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu"]


def run_valid(cid: str, design: str) -> bool:
    """A run counts only if its intervention passed and (when present) its slot / reference identity check passed
    (A4: r4_run writes the parquet only after both checks; this also guards older or interrupted runs)."""
    iv = R4 / "results" / cid / f"intervention_{design}.json"
    idn = R4 / "results" / cid / f"identity_{design}.json"
    if not iv.exists() or json.loads(iv.read_text()).get("passed") is not True:
        return False
    return not idn.exists() or json.loads(idn.read_text()).get("passed") is True


def load(cid: str, design: str):
    p = R4 / "predictions" / f"{cid}_{design}.parquet"
    if not p.exists() or not run_valid(cid, design):
        return None
    t = pd.read_parquet(p)
    return t[(t.k == 3) & (t.k_max >= 3)].sort_values(KEY).reset_index(drop=True)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def bal_mae(g, col):
    e = g.assign(ae=(g[col] - g.y).abs())
    return e.groupby(["chemotype", "extractant"]).ae.mean().groupby(level=0).mean().mean()


def deltas(t, cand, ref) -> dict:
    d2 = np.mean([r2(g.y, g[cand]) - r2(g.y, g[ref]) for _, g in t.groupby("split_seed")])
    dm = np.mean([bal_mae(g, ref) - bal_mae(g, cand) for _, g in t.groupby("split_seed")])
    return {"dR2": float(d2), "dbalMAE": float(dm)}


def harm(t, cand, ref) -> dict:
    from r4 import donor
    t = donor.add_donor(t)
    subs = {p: t[(t.A + "/" + t.B) == p] for p in PAIRS}
    subs.update({"sc009": t[t.chemotype == "sc009"], "non_sc009": t[t.chemotype != "sc009"],
                 "DGA_donor": t[t.dga_donor], "non_DGA_donor": t[~t.dga_donor],
                 "one_end_read": t[t.a_read | t.b_read], "both_hidden": t[~(t.a_read | t.b_read)]})
    out = {}
    for name, g in subs.items():
        if g.empty:
            continue
        w = [float((h[cand] - h.y).abs().mean() - (h[ref] - h.y).abs().mean()) for _, h in g.groupby("split_seed")]
        out[name] = {"worse_seed_mean": float(np.mean(w)), "seeds_worse_gt_0.010": int(sum(x > 0.010 for x in w)),
                     "harm": bool(np.mean(w) > 0.010 and sum(x > 0.010 for x in w) >= 4)}
    return {"any_harm": any(v["harm"] for v in out.values()), "harmed": [k for k, v in out.items() if v["harm"]],
            "subgroups": out}


def slot_eval(cid: str, col: str) -> dict | None:
    tabs = {d: load(cid, d) for d in DESIGNS}
    if any(v is None for v in tabs.values()):
        return None
    res = {d: {"vs_B1": deltas(t, col, "B1"), "vs_B2": deltas(t, col, "B2")} for d, t in tabs.items()}
    h1, h2 = harm(tabs["BPG"], col, "B1"), harm(tabs["BPG"], col, "B2")
    ok = lambda r: r["dR2"] >= 0.01 and r["dbalMAE"] >= 0.005  # noqa: E731
    res["harm_vs_B1_BPG"], res["harm_vs_B2_BPG"] = h1, h2
    res["criterion1"] = bool(all(ok(res[d]["vs_B1"]) for d in DESIGNS) and not h1["any_harm"])
    res["criterion2_vs_B2_part"] = bool(all(ok(res[d]["vs_B2"]) for d in DESIGNS) and not h2["any_harm"])
    return res


def draws(cid: str, prefix: str, route_col: str) -> dict | None:
    tabs = {d: load(cid, d) for d in DESIGNS}
    if any(v is None for v in tabs.values()):
        return None
    out = {}
    for d, t in tabs.items():
        cols = sorted(c for c in t.columns if c.startswith(prefix))
        out[d] = {"route_dR2": deltas(t, route_col, "B1")["dR2"],
                  "draw_dR2": [deltas(t, c, "B1")["dR2"] for c in cols]}
    return out


def inner_gate(cid: str, part: str, design: str) -> dict | None:
    """G-geo inner gate (amended in A2): the slot's best inner score over POSITIVE tree counts must be below both its
    own 0-tree score (= B1 refitted per inner block) and B2's best inner score, per outer fold-fit."""
    p = R4 / "results" / cid / f"fitted_{design}.json"
    if not p.exists():
        return None
    f = json.loads(p.read_text())
    wins = 0
    for v in f.values():
        sc = v[part]["inner_scores"]
        s_trees = min(val for k, val in sc.items() if k != "0")
        s_b2 = min(v["ref:B2"]["inner_scores"].values())
        wins += int(s_trees < sc["0"] and s_trees < s_b2)
    return {"fold_fits": len(f), "slot_below_B1_and_B2": wins}


def coverage_subset(cid: str, col: str) -> dict | None:
    """Secondary, diagnostic: deltas on the C3 queries whose two ends both have the cell's own geometry."""
    from r4 import slots_b as SB
    geo = SB.geometry_table()
    from gen13sep.metals import LANTHANIDES
    out = {}
    for d in DESIGNS:
        t = load(cid, d)
        if t is None:
            return None
        ok = [((c, LANTHANIDES.index(a)) in geo) and ((c, LANTHANIDES.index(b)) in geo) for c, a, b in zip(t.cell_id, t.A, t.B)]
        sub = t[np.array(ok)]
        out[d] = {"share_of_rows": float(np.mean(ok)), "vs_B1": deltas(sub, col, "B1")}
        if "S05" in sub and col != "S05":
            out[d]["vs_S05"] = deltas(sub, col, "S05")
    return out


def block_b() -> dict:
    """Block B (PROTOCOL section 11): criterion 1 for S05 / S06; conditional S09 / S22 (only if S06 passes) and S08
    (only if S05 and S06 both pass); the encoder gate G-geo (S06 inner loss below B2's in >= 30 of 50 fold-fits AND S06
    passing criterion 1; amended in A2: S06 positive-tree inner score below both its 0-tree (B1) score and B2's);
    the geometry claim on BPG and BP (+ criterion 2 vs B2); the coverage-matched subset as a secondary diagnostic."""
    out = {"slots": {}}
    for s in ("S05", "S06"):
        r = slot_eval(s, s)
        out["slots"][s] = r if r is not None else "NOT RUN"
    s5, s6 = out["slots"]["S05"], out["slots"]["S06"]
    p6 = isinstance(s6, dict) and s6["criterion1"]
    p5 = isinstance(s5, dict) and s5["criterion1"]
    out["run_S09_S22"] = bool(p6)
    out["run_S08"] = bool(p5 and p6)
    g = [inner_gate("S06", "b:S06", d) for d in DESIGNS]
    wins = sum(x["slot_below_B1_and_B2"] for x in g if x) if all(g) else None
    out["G_geo_inner"] = {"slot_below_B1_and_B2_fold_fits": wins, "of": 50, "passes": bool(wins is not None and wins >= 30)}
    out["run_encoder_S07_S10"] = bool(out["G_geo_inner"]["passes"] and p6)
    out["coverage_matched_secondary"] = {s: coverage_subset(s, s) for s in ("S05", "S06") if isinstance(out["slots"][s], dict)}
    out["complexity_order"] = ["B1", "S05", "S06"]           # section 6.6 tie rule: fewer features / parts first
    s09, s22 = draws("S09", "S09_d", "S06"), draws("S22", "S22", "S06")
    if p6 and s09 is not None and s22 is not None and isinstance(s5, dict):
        per = {}
        for d in DESIGNS:
            g5, g6 = s5[d]["vs_B1"]["dR2"], s6[d]["vs_B1"]["dR2"]
            per[d] = bool(g6 - g5 >= 0.005 and g6 > max(s09[d]["draw_dR2"]) and
                          np.mean([v - g5 for v in s09[d]["draw_dR2"]]) < 0.5 * (g6 - g5) and g6 - s22[d]["draw_dR2"][0] >= 0.005)
        # the geometry claim is cell-conditioned: it also needs criterion 2's vs-B2 part (sections 6.2, 11)
        out["geometry_rule"] = {"per_design": per, "criterion2_vs_B2_part": bool(s6["criterion2_vs_B2_part"]),
                                "passed": all(per.values()) and bool(s6["criterion2_vs_B2_part"]), "S09": s09, "S22": s22}
    labels = {}
    for s, r in (("S05", s5), ("S06", s6)):
        if not isinstance(r, dict):
            labels[s] = "NOT RUN"
        elif not r["criterion1"]:
            labels[s] = "no meaningful gain over B1 (criterion 1 failed)"
        else:
            lab = ["exploratory gain over B1"]
            if s == "S06":
                lab.append("geometry-specific (exploratory)" if out.get("geometry_rule", {}).get("passed") else "geometry attribution not shown")
            else:
                lab.append("chemistry attribution untested")
            labels[s] = "; ".join(lab)
    out["labels"] = labels
    out["finalist_eligible"] = [s for s, r in (("S05", s5), ("S06", s6)) if isinstance(r, dict) and r["criterion1"]]
    return out


def current_seal() -> str:
    sys.path.insert(0, str(R4 / "scripts"))
    import r4_seal
    ok, _, seal = r4_seal.verify()
    return seal[:12] if ok else ""


def synthetic_power() -> dict | None:
    """Sealed synthetic study (results/synthetic_c/summary.json): firing counts of the transfer claim per world."""
    p = R4 / "results" / "synthetic_c" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def s14_fallback(design: str) -> dict | None:
    """How often S11's best network had w_Ln = 0 (S14 then carries an Ln term S13 lacks) and how often S14's outer
    choice fell back to B1 or to w_X = 0 (= S11's network)."""
    p = R4 / "results" / "S14" / f"fitted_{design}.json"
    if not p.exists():
        return None
    f = json.loads(p.read_text())
    best0 = sum(v["c:S14"]["best_net"]["w_Ln"] == 0 for v in f.values())
    fb = sum(v["c:S14"]["choice"] == "zero" or v["c:S14"]["choice"].get("w_X") == 0 for v in f.values())
    return {"fold_fits": len(f), "S11_best_w_Ln_0": best0, "S14_fell_back": fb}


def block_c() -> dict:
    """Block C (PROTOCOL section 12, amendment A4).
    Routes: S11 always; S13 only if G-transfer passed under the current seal (results/gate_c.json; otherwise S13 is NOT
    RUN, and an off-route S13 table is reported but never eligible); controls S14 / S15 / S16 only if S13 is on route,
    passes criterion 1 AND S13 - S11 >= 0.005 (dR2 vs B1) on BPG and BP; S24 additionally needs its count gate.
    A run counts only if its intervention and identity checks passed (load / run_valid).
    Transfer claim (on BPG and BP, all dR2 vs B1): criterion 1 and criterion 2's vs-B2 part, S13 - S11 >= 0.005,
    S13 - S15 >= 0.005 and (S15 - S11) < 0.5 (S13 - S11), S13 > max(S16 draws), mean(S16 - S11) < 0.5 (S13 - S11).
    Sub-labels: private-head selectivity over naive sharing = S13 - S14 > 0 on BPG and BP; metal-state specificity =
    S13 - S24 > 0 on BPG and BP (untested when S24 is NOT RUN).  Synthetic consequence (sealed run): if the transfer
    claim fired in fewer than half of the AN_SIGNAL replicates, a non-positive transfer result says 'control power not
    shown'; if it fired in any NULL or SMOOTH_NULL replicate, a positive label says it is not distinguishable from an
    irrelevant / smooth auxiliary.  Failing a control downgrades the label, not criterion-1 eligibility."""
    out = {"slots": {}}
    seal = current_seal()
    out["seal"] = seal
    gate = R4 / "results" / "gate_c.json"
    g = {k: v for k, v in json.loads(gate.read_text()).items() if k != "folds"} if gate.exists() else None
    if g is not None and (not seal or g.get("seal") != seal):
        g = {"stale": True, "gate_seal": g.get("seal"), "current_seal": seal}
    out["G_transfer"] = g if g is not None else "not evaluated"
    g_ok = bool(g is not None and g.get("G_transfer_passes") is True)
    s24_gate = bool(g is not None and g.get("S24_count_gate_passes") is True)
    s11, s13 = slot_eval("S11", "S11"), slot_eval("S13", "S13")
    out["slots"]["S11"] = s11 if s11 is not None else "NOT RUN"
    if s13 is None:
        out["slots"]["S13"] = "NOT RUN" + ("" if g_ok else " (G-transfer failed or not evaluated)")
    elif not g_ok:
        out["slots"]["S13"] = {"off_route": True, "note": "G-transfer failed or not evaluated: reported, never eligible",
                               **s13}
    else:
        out["slots"]["S13"] = s13
    on13 = isinstance(s13, dict) and g_ok
    sx, g11 = {}, {}
    if s13 is not None:
        tabs = {d: load("S13", d) for d in DESIGNS}
        sx = {d: deltas(t, "S13", "S11") for d, t in tabs.items()}
        g11 = {d: (s11[d]["vs_B1"]["dR2"] if isinstance(s11, dict) else deltas(tabs[d], "S11", "B1")["dR2"])
               for d in DESIGNS}
    out["S13_vs_S11"] = sx or None
    p13 = bool(on13 and s13["criterion1"] and all(sx[d]["dR2"] >= 0.005 for d in DESIGNS))
    out["run_controls"] = p13
    rules = {}
    if p13:
        per, sel = {}, {}
        for d in DESIGNS:
            g13 = s13[d]["vs_B1"]["dR2"]
            t15, t16, t14 = load("S15", d), load("S16", d), load("S14", d)
            if t15 is None or t16 is None:
                per[d] = None
            else:
                g15 = deltas(t15, "S15", "B1")["dR2"]
                dr = [deltas(t16, c, "B1")["dR2"] for c in sorted(c for c in t16.columns if c.startswith("S16_d"))]
                per[d] = bool(g13 - g15 >= 0.005 and (g15 - g11[d]) < 0.5 * (g13 - g11[d]) and g13 > max(dr)
                              and np.mean([v - g11[d] for v in dr]) < 0.5 * (g13 - g11[d]))
            sel[d] = None if t14 is None else bool(g13 - deltas(t14, "S14", "B1")["dR2"] > 0)
        passed = (None if any(v is None for v in per.values())
                  else bool(all(per.values()) and s13["criterion2_vs_B2_part"]))
        rules = {"per_design": per, "passed": passed, "selectivity_per_design": sel,
                 "selectivity": (None if any(v is None for v in sel.values()) else all(sel.values())),
                 "S14_fallback": {d: s14_fallback(d) for d in DESIGNS}}
        if not s24_gate:
            rules["metal_state"] = "untested (S24 NOT RUN: label-free count gate failed)"
        else:
            t24 = {d: load("S24", d) for d in DESIGNS}
            rules["metal_state"] = (None if any(t is None for t in t24.values()) else
                                    bool(all(s13[d]["vs_B1"]["dR2"] - deltas(t24[d], "S24", "B1")["dR2"] > 0 for d in DESIGNS)))
    out["transfer_rule"] = rules
    syn = synthetic_power()
    out["synthetic_power"] = syn
    weak = syn is not None and syn["AN_SIGNAL"]["transfer_claim"] * 2 < syn["AN_SIGNAL"]["n"]
    nullfire = syn is not None and (syn["NULL"]["transfer_claim"] > 0 or syn["SMOOTH_NULL"]["transfer_claim"] > 0)
    word = lambda v: "shown" if v is True else "not shown" if v is False else "untested"  # noqa: E731
    labels = {}
    for s, r in (("S11", s11), ("S13", s13)):
        if not isinstance(r, dict):
            labels[s] = "NOT RUN"
        elif s == "S13" and not g_ok:
            labels[s] = "off-route (G-transfer failed or not evaluated); not eligible"
        elif not r["criterion1"]:
            labels[s] = "no meaningful gain over B1 (criterion 1 failed)"
        elif s == "S13":
            if rules.get("passed") is True:
                labels[s] = ("exploratory gain over B1; An(III) data improve held-out Ln log SF beyond Ln-only, matched "
                             "and permuted auxiliaries (exploratory; a smooth-auxiliary regulariser is not excluded); "
                             f"private-head selectivity over naive sharing {word(rules.get('selectivity'))}; "
                             f"metal-state specificity {word(rules.get('metal_state'))}")
                if nullfire:
                    labels[s] += "; NOT distinguishable from an irrelevant / smooth auxiliary (synthetic null firing)"
            elif rules.get("passed") is None and p13:
                labels[s] = "exploratory gain over B1; An attribution untested (controls on route, not yet run)"
            else:
                labels[s] = "exploratory gain over B1; An attribution not shown"
        else:
            labels[s] = "exploratory gain over B1 (Ln-only multitask scaffold; chemistry attribution untested)"
    if weak and isinstance(s13, dict) and g_ok and rules.get("passed") is not True:
        labels["S13"] += (f"; control power not shown (synthetic AN_SIGNAL transfer claim "
                          f"{syn['AN_SIGNAL']['transfer_claim']}/{syn['AN_SIGNAL']['n']})")
    out["labels"] = labels
    out["finalist_eligible"] = [s for s, r in (("S11", s11), ("S13", s13))
                                if isinstance(r, dict) and r["criterion1"] and (s == "S11" or g_ok)]
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "C":
        dec = block_c()
        (R4 / "results" / "blockC_decision.json").write_text(json.dumps(dec, indent=1, default=str), encoding="utf-8",
                                                             newline=chr(10))
        md = ["# Block C decision (mechanical; report/r4_decide.py C)", "", f"- G-transfer: {dec['G_transfer']}"]
        for s in ("S11", "S13"):
            r = dec["slots"][s]
            if not isinstance(r, dict):
                md.append(f"- **{s}**: {r}")
                continue
            md.append(f"- **{s}**: BPG dR2 vs B1 {r['BPG']['vs_B1']['dR2']:+.4f}, dbalMAE {r['BPG']['vs_B1']['dbalMAE']:+.4f}; "
                      f"BP {r['BP']['vs_B1']['dR2']:+.4f} / {r['BP']['vs_B1']['dbalMAE']:+.4f}; harm vs B1 "
                      f"{r['harm_vs_B1_BPG']['harmed'] or 'none'}; criterion 1 {'PASS' if r['criterion1'] else 'fail'}")
        md += ["", f"- S13 vs S11: {json.dumps(dec['S13_vs_S11'])}", f"- controls run: {dec['run_controls']}; transfer rule: "
               f"{json.dumps(dec['transfer_rule'])}", f"- labels: {json.dumps(dec['labels'])}",
               f"- finalist-eligible (block C): {dec['finalist_eligible'] or 'none'}"]
        (R4 / "results" / "blockC_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
        print("\n".join(md))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "B":
        dec = block_b()
        (R4 / "results" / "blockB_decision.json").write_text(json.dumps(dec, indent=1, default=str), encoding="utf-8")
        md = ["# Block B decision (mechanical; report/r4_decide.py B)", ""]
        for s in ("S05", "S06"):
            r = dec["slots"][s]
            if not isinstance(r, dict):
                md.append(f"- **{s}**: {r}")
                continue
            md.append(f"- **{s}**: BPG dR2 vs B1 {r['BPG']['vs_B1']['dR2']:+.4f}, dbalMAE {r['BPG']['vs_B1']['dbalMAE']:+.4f}; "
                      f"BP {r['BP']['vs_B1']['dR2']:+.4f} / {r['BP']['vs_B1']['dbalMAE']:+.4f}; vs B2 BPG "
                      f"{r['BPG']['vs_B2']['dR2']:+.4f} / {r['BPG']['vs_B2']['dbalMAE']:+.4f}; harm vs B1 "
                      f"{r['harm_vs_B1_BPG']['harmed'] or 'none'}; criterion 1 {'PASS' if r['criterion1'] else 'fail'}")
        md += ["", f"- S09 / S22 run: {dec['run_S09_S22']}; S08 run: {dec['run_S08']}; G-geo inner gate: {dec['G_geo_inner']}; "
               f"encoder S07 / S10 run: {dec['run_encoder_S07_S10']}; coverage-matched secondary: {json.dumps(dec['coverage_matched_secondary'])}", f"- labels: {json.dumps(dec['labels'])}",
               f"- finalist-eligible (block B): {dec['finalist_eligible'] or 'none'}"]
        (R4 / "results" / "blockB_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
        print("\n".join(md))
        return 0
    dec = {"slots": {}}
    for s in ("S01", "S02", "S03"):
        r = slot_eval(s, s)
        dec["slots"][s] = r if r is not None else "NOT RUN"
    s1, s2 = dec["slots"]["S01"], dec["slots"]["S02"]
    passing = [s for s, r in (("S01", s1), ("S02", s2)) if isinstance(r, dict) and r["criterion1"]]
    dec["run_S03_S04"] = bool(isinstance(s2, dict) and s2["criterion1"])
    if passing:
        best = max(passing, key=lambda s: dec["slots"][s]["BPG"]["vs_B1"]["dR2"])
        close = [s for s in passing if dec["slots"][best]["BPG"]["vs_B1"]["dR2"] - dec["slots"][s]["BPG"]["vs_B1"]["dR2"] < 0.005]
        dec["S21_route"] = sorted(close, key=lambda s: (N_PARAMS[s], s))[0]
    else:
        dec["S21_route"] = None
    # control rules (when their runs exist)
    s04 = draws("S04", "S04_d", "S02")
    s03 = dec["slots"]["S03"]
    if isinstance(s03, dict) and s04 is not None and isinstance(s2, dict):
        el = {d: bool(s2[d]["vs_B1"]["dR2"] - s03[d]["vs_B1"]["dR2"] >= 0.005 and s04[d]["route_dR2"] > max(s04[d]["draw_dR2"]))
              for d in DESIGNS}
        dec["electronic_rule"] = {"per_design": el, "S04": s04, "passed": all(el.values())}
    if dec["S21_route"]:
        s21 = draws(f"S21-{dec['S21_route']}", "S21_d", dec["S21_route"])
        if s21 is not None:
            ch = {d: bool(s21[d]["route_dR2"] > max(s21[d]["draw_dR2"]) and np.mean(s21[d]["draw_dR2"]) < 0.5 * s21[d]["route_dR2"])
                  for d in DESIGNS}
            dec["donor_rule"] = {"route": dec["S21_route"], "per_design": ch, "S21": s21, "passed": all(ch.values())}
    labels = {}
    for s, r in (("S01", s1), ("S02", s2)):
        if not isinstance(r, dict):
            labels[s] = "NOT RUN"
            continue
        if not r["criterion1"]:
            labels[s] = "no meaningful gain over B1 (criterion 1 failed)"
            continue
        lab = ["exploratory gain over B1"]
        dr = dec.get("donor_rule")
        if dr and dr["route"] == s and dr["passed"] and r["criterion2_vs_B2_part"]:
            lab.append("donor-conditioned (exploratory)")
        elif dec["S21_route"] != s:
            lab.append("donor claim untested")
        if s == "S02" and dec.get("electronic_rule", {}).get("passed"):
            lab.append("electronic-specific (exploratory)")
        labels[s] = "; ".join(lab)
    dec["labels"] = labels
    dec["finalist_eligible"] = [s for s in ("S01", "S02") if isinstance(dec["slots"][s], dict) and dec["slots"][s]["criterion1"]]
    (R4 / "results" / "blockA_decision.json").write_text(json.dumps(dec, indent=1, default=str), encoding="utf-8")
    md = ["# Block A decision (mechanical; report/r4_decide.py)", ""]
    for s in ("S01", "S02", "S03"):
        r = dec["slots"][s]
        if not isinstance(r, dict):
            md.append(f"- **{s}**: {r}")
            continue
        md.append(f"- **{s}**: BPG dR2 vs B1 {r['BPG']['vs_B1']['dR2']:+.4f}, dbalMAE {r['BPG']['vs_B1']['dbalMAE']:+.4f}; "
                  f"BP {r['BP']['vs_B1']['dR2']:+.4f} / {r['BP']['vs_B1']['dbalMAE']:+.4f}; vs B2 BPG "
                  f"{r['BPG']['vs_B2']['dR2']:+.4f} / {r['BPG']['vs_B2']['dbalMAE']:+.4f}; harm vs B1 {r['harm_vs_B1_BPG']['harmed'] or 'none'}; "
                  f"criterion 1 {'PASS' if r['criterion1'] else 'fail'}")
    md += ["", f"- S03 / S04 run: {dec['run_S03_S04']}; S21 route: {dec['S21_route']}",
           f"- labels: {json.dumps(labels)}", f"- finalist-eligible (block A): {dec['finalist_eligible'] or 'none'}"]
    (R4 / "results" / "blockA_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
