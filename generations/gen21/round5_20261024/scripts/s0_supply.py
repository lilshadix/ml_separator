#!/usr/bin/env python
"""Stage-0 label-blind supply ledger, PROVISIONAL laboratory key, mechanism-class map and supply-audit kill
(CLAUDE_CODE_PROMPT.md sections 4.3 items 1 and 7, 4.6, 5.5, 7.5).  METADATA ONLY: no D, log D or SF column is
loaded (asserted by r5.supply.read_meta); no bench value, prediction or score is touched.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s0_supply.py
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s0_supply.py --out results/supply_audit/<UTC> --request U5

Writes results/stage0/lab_key.csv (corpus references and literature candidates, conservative and liberal laboratory
ids, frozen flags, classes), lab_key_links.csv (corpus links, for the user's U6 review), mechanism_class_map.csv (the
frozen rule table), supply_audit.json (lab-by-lab claim supply, the audit under D4, the U6 review list) and
<round5>/SUPPLY_LEDGER.csv.  These are sealed Stage-0 inputs: once round 5 is sealed the script refuses unless --out
is given (a post-seal rerun after an answered request, brief 4.6), which writes the same files under that folder only
and appends a one-line outcome below CONFIRMATION_PACKAGE.md's append-only marker; --amend overrides the guard (an
AMENDMENTS.md entry and a reseal follow).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402

import pandas as pd  # noqa: E402

from r5 import supply as SU  # noqa: E402

OUT = paths.RES / "stage0"
LEDGER = paths.R5 / "SUPPLY_LEDGER.csv"
SCENARIOS_CONFIRM1 = [  # ideas2/confirm.md "Power by supply scenario" (EXPLORATORY planning, quoted, not recomputed)
    {"scenario": "verified only (7 E1 labs)", "G_HS": "3", "G_Gd": "4-5", "G_ECT": "2-3",
     "C-GD": "below minimum (0.44-0.56 at G = 5)", "A1": "0.50 at G = 5", "C-B1 chain": "below minimum",
     "C-ECT": "below minimum"},
    {"scenario": "expected (about 13 labs)", "G_HS": "6", "G_Gd": "8", "G_ECT": "5-6", "C-GD": "0.62-0.74", "A1": "0.83",
     "C-B1 chain": "0.36 / 0.23", "C-ECT": "0.63-0.67 at 6"},
    {"scenario": "all E1 + E2 convert (16-19)", "G_HS": "8", "G_Gd": "10-12", "G_ECT": "7-9", "C-GD": "0.88-0.94",
     "A1": "0.93", "C-B1 chain": "0.52-0.65 / 0.33-0.45", "C-ECT": "0.89-0.99"},
    {"scenario": "extended targeted search (20-30 labs)", "G_HS": "10-13", "G_Gd": "13-16", "G_ECT": "7-9",
     "C-GD": "0.97-1.00", "A1": "0.99", "C-B1 chain": "0.70-0.83 / 0.50-0.62", "C-ECT": "0.89-0.99"}]


def main(argv=None) -> int:
    global OUT, LEDGER
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="post-seal rerun: write every output under this folder only")
    ap.add_argument("--request", default="", help="the answered request that triggered a post-seal rerun (e.g. U5)")
    ap.add_argument("--amend", action="store_true", help="rewrite the sealed results/stage0 files (amendment + reseal)")
    a = ap.parse_args(argv)
    from r5 import sealcheck
    if a.out:
        OUT = (paths.R5 / a.out) if not Path(a.out).is_absolute() else Path(a.out)
        if OUT.resolve() == (paths.RES / "stage0").resolve():
            raise SystemExit("--out must not be results/stage0 (the sealed inputs)")
        LEDGER = OUT / "SUPPLY_LEDGER.csv"
    else:
        sealcheck.refuse_if_sealed("scripts/s0_supply.py (rewrites results/stage0 and SUPPLY_LEDGER.csv)", a.amend)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cand = SU.read_meta(SU.CAND, dtype=str).fillna("")
    reg = SU.read_meta(SU.REG)
    surn = SU.read_meta(SU.SURN)
    src = SU.read_meta(SU.SRC_DOI)
    loaded = {"P4_candidates.csv": list(cand.columns), "P4_corpus_registry.csv": list(reg.columns),
              "P4_corpus_surnames.csv": list(surn.columns), "sources_by_doi.csv": list(src.columns)}

    # ---- corpus laboratory key ------------------------------------------------------------------------
    ck, links, parsed = SU.corpus_key(reg)
    ck_s, links_s, _ = SU.corpus_key(reg, SU.DROP)       # sensitivity: only the four named surnames dropped
    doi2pub = {str(d).lower(): str(p) for d, p in zip(src.doi, src.publication_ids)}
    ck["publication_ids"] = ck.key.str.lower().map(doi2pub).fillna("")
    pub2key = {p: k for k, ps in zip(ck.key, ck.publication_ids) for p in ps.split(";") if p}
    mine = set().union(*(set(P["families"]) for P in parsed.values()))
    p4s = set(surn.surname.astype(str))

    def lab_counts(t):
        return {"conservative": int(t.lab_conservative.nunique()), "liberal": int(t.lab_liberal.nunique()),
                "references": int(len(t))}
    sub = {"all_registry_keys": ck, "v1_cohort_dois": ck[ck.in_v1_cohort], "c3_dois": ck[ck.c3_cells.fillna(0) > 0]}
    counts = {k: lab_counts(v) for k, v in sub.items()}
    counts_s = {k: lab_counts(ck_s[ck_s.key.isin(v.key)]) for k, v in sub.items()}
    ect = {}
    pc = paths.RES / "stage0" / "ect_population_cells.csv"
    if pc.exists():
        cells = pd.read_csv(pc)
        for nm, m in (("V1_GE1_publications", cells.n_v1 >= 1), ("PRIMARY_publications", cells.in_PRIMARY)):
            pubs = sorted(cells[m].publication.unique())
            keys = {p: pub2key.get(p, "") for p in pubs}
            labs = ck.set_index("key").reindex([k for k in keys.values() if k])
            ect[nm] = {"n_publications": len(pubs), "unmapped": sorted(p for p, k in keys.items() if not k),
                       "labs_conservative": int(labs.lab_conservative.nunique()),
                       "labs_liberal": int(labs.lab_liberal.nunique()),
                       "by_publication": {p: {"doi": k, "lab_conservative": ck.set_index("key").lab_conservative.get(k, ""),
                                              "lab_liberal": ck.set_index("key").lab_liberal.get(k, ""),
                                              "senior": ck.set_index("key").senior.get(k, "")} for p, k in keys.items()}}
    else:
        ect["note"] = "results/stage0/ect_population_cells.csv not found; run scripts/s0_ect_population.py first"

    # ---- literature candidates -------------------------------------------------------------------------
    cf = SU.candidates_frozen(cand, parsed)
    lab_of_ref = dict(zip(ck.key, ck.lab_conservative))
    cf["auto_best_ref_lab"] = cf.auto_best_ref.map(lab_of_ref).fillna("")
    cs = SU.claim_supply(cf)
    fcs = {"P4_rates_pE2_0.5_pU_0.25": SU.expected_claims(cs, 0.5, 0.25),
           "PC_P4_rates_pE2_2of6_pU_0": SU.expected_claims(cs, 2 / 6, 0.0)}
    audit = SU.supply_audit(cs, fcs, cand=cf)

    # ---- outputs ------------------------------------------------------------------------------------------
    ccols = ["unit_type", "key", "year", "publication_ids", "in_v1_cohort", "c3_cells", "senior", "pi_positions",
             "families", "family_given_decisions", "truncated", "lab_conservative", "lab_liberal"]
    kc = ck[ccols].copy()
    kc["status"] = "PROVISIONAL (U6 review pending)"
    kd = cf.rename(columns={"cand_id": "key"}).assign(unit_type="candidate")
    dcols = ["unit_type", "key", "doi", "year", "first_author", "last_author", "lab_id", "lab_conservative", "lab_liberal",
             "lab_in_corpus_P4", "lab_in_corpus_frozen", "lab_in_corpus_auto", "auto_best_ref", "auto_best_ref_lab",
             "auto_evidence", "second_order_link", "frozen_fix_notes", "eligibility_P4", "eligibility_frozen",
             "screen_level", "ext_class", "chemotype_label", "chemotype_label_merged", "chemotype_in_corpus",
             "mech_rule", "mechanism_class", "cgd_stratum", "ect_report_class", "il_diluent_flag", "aq_complexant_flag",
             "synergist_flag", "acid_medium", "medium_rule", "medium_class", "gd_medium_eligible", "n_ln_reported",
             "m22_rule_8Ln_EuGdTb", "full_LaLu_series", "sibling_conditions", "sibling_desc", "t_classes",
             "non_t0_possible", "t0_only"]
    kd = kd[dcols].copy()
    kd["status"] = "PROVISIONAL (U6 review pending)"
    lab_key = pd.concat([kc, kd], ignore_index=True, sort=False)
    lab_key.to_csv(OUT / "lab_key.csv", index=False, lineterminator="\n")
    links.to_csv(OUT / "lab_key_links.csv", index=False, lineterminator="\n")
    SU.mech_rules_table().to_csv(OUT / "mechanism_class_map.csv", index=False, lineterminator="\n")
    led = pd.DataFrame(SU.ledger_rows(cf))
    crow = [{"section": "claim", "quantity": f"{r['claim']} supply ({r['unit']})", "tier": f"minimum G {r['min_G']}",
             "reported_P4": "", "corrected_PC_P4": f"realistic {r['realistic'][0]}-{r['realistic'][1]} (brief)",
             "recomputed_r5": f"verified {r['supply_verified_literature']}; in hand {r['supply_in_hand']}; "
                              f"decision (D4) {r['decision_supply']}; bracket high {r['bracket_high']}; role {r['role']}; "
                              f"below minimum: {r['below_min']}; forecasts "
                              + "; ".join(f"{n} {v['expected']}" for n, v in r["forecasts"].items())
                              + (f" ({r['forecast_unit']})" if r.get("forecast_unit") else "")
                              + f"; all-convert {r['supply_all_convert']}"
                              + (f" (laboratories in hand {r['supply_in_hand_laboratories']}, all-convert "
                                 f"{r['supply_all_convert_laboratories']})" if "supply_in_hand_laboratories" in r else ""),
             "source": r["source"], "note": "forecasts, all-convert and the bracket high never decide (D4)"}
            for r in audit["claims"]]
    crow.append({"section": "audit", "quantity": "supply-audit kill (section 4.3 item 7)", "tier": "",
                 "reported_P4": "", "corrected_PC_P4": "expected: every claim below its minimum G (brief 4.6)",
                 "recomputed_r5": f"kill fires: {audit['kill_fires']} (test claims only); under forecasts: "
                                  f"{audit['kill_under_forecast']}; under the bracket high: {audit['kill_under_bracket_high']}; "
                                  f"Gd-dip descriptive-run trigger: {audit['descriptive_run_trigger_gd_dip']}",
                 "source": "brief 4.3 item 7, 4.6", "note": audit["outcome"]})
    led = pd.concat([led, pd.DataFrame(crow)], ignore_index=True)
    led["status"] = "PROVISIONAL (U5 / U6 / U2 unanswered at Stage 0)"
    led.to_csv(LEDGER, index=False, lineterminator="\n")

    dis = cf[cf.lab_in_corpus_frozen != cf.lab_in_corpus_auto][["cand_id", "lab_id", "lab_in_corpus_frozen",
                                                                 "lab_in_corpus_auto", "auto_evidence"]]
    out = {
        "label": "STAGE-0 SUPPLY AUDIT (metadata only; no D, log D or SF column loaded; no score)",
        "status": "PROVISIONAL: laboratory key pending U6 review, planning amplitudes pending U5, SAFE pending U2",
        "columns_loaded": loaded,
        "inputs_lf_sha256": {p.name: paths.sha256_text_lf(p) for p in (SU.CAND, SU.REG, SU.SURN, SU.SRC_DOI)},
        "lab_key_rules": {"drop_common_surnames": sorted(SU.DROP), "link_L2": ">= 2 shared distinctive family names",
                          "link_LPI": "same person (family + first initial) in a PI position (first or last author of "
                                      "the first ordered representation) of both references",
                          "conservative": "connected components of L2 or LPI", "liberal": "senior (last-listed) person",
                          "family_given": "Chinese family-name list + cross-reference attestation + representation "
                                          "majority", "institution_fix": SU.INSTITUTION_FIX, "candidate_fix": SU.CAND_FIX,
                          "second_order_links_sensitivity": SU.SECOND_ORDER},
        "corpus_lab_counts": counts, "corpus_lab_counts_sensitivity_four_named_only": counts_s,
        "corpus_links": {"n": int(len(links)), "L2": int(links.L2.sum()), "LPI": int(links.LPI.sum()),
                         "single_pi_only": int(links.single_pi_only.sum()),
                         "n_sensitivity_four_named_only": int(len(links_s))},
        "family_given_repair": {"references_with_FG_decisions": int((ck.family_given_decisions.str.contains("FG")).sum()),
                                "families_added_vs_P4_surnames": sorted(mine - p4s)[:80],
                                "P4_surnames_not_in_repaired_parse": sorted(p4s - mine)[:80]},
        "ect_laboratory_annotation": ect,
        "claim_supply_lab_by_lab": cs,
        "expected_supply_forecasts": fcs,
        "audit": audit,
        "minimum_G_source": {"C-ECT": 7, "C-GD": 6, "Gd-dip transport": 10, "SAFE route": 10},
        "supply_scenarios_confirm1_quoted": SCENARIOS_CONFIRM1,
        "U6_review_list": {
            "single_pi_only_corpus_links": links[links.single_pi_only][["key_a", "key_b", "pi_shared"]].to_dict("records"),
            "family_given_references": ck[ck.family_given_decisions.str.contains("FG")][["key", "family_given_decisions"]]
            .to_dict("records"),
            "institution_link_fixes": SU.INSTITUTION_FIX, "second_order_links": SU.SECOND_ORDER,
            "hand_vs_auto_disagreements": dis.to_dict("records"),
            "truncated_author_lists": ck[ck.truncated].key.tolist()},
    }
    if a.out:
        out["post_seal_rerun"] = {"request": a.request, "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    paths.write_text_lf(OUT / "supply_audit.json", json.dumps(out, indent=1, default=str) + "\n")
    print("corpus labs (conservative / liberal):", {k: (v["conservative"], v["liberal"]) for k, v in counts.items()})
    print("  sensitivity (four named surnames only):", {k: (v["conservative"], v["liberal"]) for k, v in counts_s.items()})
    print("corpus links:", out["corpus_links"])
    if "V1_GE1_publications" in ect:
        print("E-CT publications:", {k: (v["n_publications"], v["labs_conservative"], v["labs_liberal"], v["unmapped"])
                                     for k, v in ect.items()})
    for r in led[led.section == "supply"].itertuples(index=False):
        print(f"  [{r.tier}] {r.quantity}: P4 {r.reported_P4} | r5 {r.recomputed_r5}")
    for r in audit["claims"]:
        print(f"  {r['claim']} ({r['role']}): min {r['min_G']}; verified {r['supply_verified_literature']}; in hand "
              f"{r['supply_in_hand']}; decision (D4) {r['decision_supply']}; bracket high {r['bracket_high']}; below_min "
              f"{r['below_min']}; forecasts { {n: v['expected'] for n, v in r['forecasts'].items()} }; all-convert "
              f"{r['supply_all_convert']}")
    print("supply-audit kill fires:", audit["kill_fires"], "| under forecasts:", audit["kill_under_forecast"])
    if a.out:                                   # brief 4.6: the rerun's outcome, appended below CP's marker
        line = (f"- {out['post_seal_rerun']['utc']}, rerun after request {a.request or '(none named)'} ({a.out}): "
                + ", ".join(f"{r['claim']} {r['decision_supply']} / {r['min_G']}" for r in audit["claims"])
                + f"; kill fires: {audit['kill_fires']}.\n")
        with open(paths.R5 / "CONFIRMATION_PACKAGE.md", "ab") as fh:
            fh.write(line.encode("utf-8"))
        print("appended the outcome below the marker of CONFIRMATION_PACKAGE.md")
    print(f"hand vs auto lab flag disagreements: {len(dis)}")
    for p in (OUT / "lab_key.csv", OUT / "lab_key_links.csv", OUT / "mechanism_class_map.csv", OUT / "supply_audit.json",
              LEDGER):
        print("wrote", p)
    print(f"done {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
