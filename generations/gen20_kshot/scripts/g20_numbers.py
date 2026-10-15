#!/usr/bin/env python
"""Collect every number quoted in GEN20_REPORT.md into tables/report_numbers.csv with its source
file and row, so a reader can trace each one (gen19 convention).

    .venv/bin/python generations/gen20_kshot/scripts/g20_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

G20 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G20))
from gen20k import bootstrap  # noqa: E402

R = bootstrap.RESULTS
rows: list[dict] = []


def add(name, value, source, where=""):
    rows.append({"name": name, "value": value, "source": str(Path(source).relative_to(G20)) if str(source).startswith(str(G20)) else str(source), "where": where})


def levels(label, design, model, K, pol, arm_k, cols):
    f = R / label / f"levels_{design}_{model}_C{K}_{pol}.csv"
    lv = pd.read_csv(f)
    r = lv[lv.arm == arm_k]
    if r.empty:
        return
    for c in cols:
        add(f"{design}/{model}/C{K}/{pol}/{arm_k}/{c}", float(r[c].iat[0]), f, f"arm == {arm_k}")


def main() -> int:
    # anchors and replication
    rep = json.loads((R / "gen15_replication" / "manifest.json").read_text())["records"]
    for k, v in rep["anchors"].items():
        add(f"gen15_replication/{k}", v, R / "gen15_replication" / "board.csv", f"arm == {k}, macro_mae_extractant")
    cl = pd.read_csv(R / "gen15_replication" / "closure_summary.csv")
    for _, r in cl.iterrows():
        add(f"gen15_replication/closed_fraction/{r['how']}", float(r["closed_fraction"]), R / "gen15_replication" / "closure_summary.csv", r["how"])
        add(f"gen15_replication/adjacent_closed_fraction/{r['how']}", float(r["adjacent_closed_fraction"]), R / "gen15_replication" / "closure_summary.csv", r["how"])
    # cohort counts
    for model in ("metal", "pair"):
        f = R / "discovery" / f"cohort_counts_BP_{model}.csv"
        cc = pd.read_csv(f)
        for _, r in cc.iterrows():
            for c in ("cells", "extractants", "chemotypes", "publications", "query_pairs", "kish_chemotypes_pairs", "largest_chemotype_share"):
                if c in cc.columns and pd.notna(r.get(c)):
                    add(f"cohort/BP/{model}/k{int(r['k'])}/{c}", float(r[c]), f, f"k == {int(r['k'])}")
    # registered contrasts
    f = R / "discovery" / "registered_contrasts_BP.csv"
    rc = pd.read_csv(f)
    for _, r in rc.iterrows():
        tag = f"registered/{r['name']}/{r['statistic']}"
        for c in ("point", "ci95_low", "ci95_high", "ci9875_low", "ci9875_high", "bca9875_low", "bca9875_high", "p_two_sided",
                  "wcr_wcr_p_wcr", "mde_80", "seeds_positive", "loco_all_positive", "n_units", "n_clusters", "kish_clusters", "macro_reference", "macro_candidate"):
            if c in rc.columns and pd.notna(r.get(c)):
                add(f"{tag}/{c}", r[c], f, f"name == {r['name']}, statistic == {r['statistic']}")
    f = R / "discovery" / "registered_style_contrasts_other_arms_BP.csv"
    ex = pd.read_csv(f)
    for _, r in ex[ex.statistic.isin(["MAE_macro_extractant", "R2"])].iterrows():
        tag = f"registered_style/{r['mstar']}/{r['name']}/{r['statistic']}"
        for c in ("point", "ci9875_low", "ci9875_high", "p_two_sided", "wcr_wcr_p_wcr", "mde_80", "seeds_positive", "loco_all_positive"):
            if c in ex.columns and pd.notna(r.get(c)):
                add(f"{tag}/{c}", r[c], f, f"mstar == {r['mstar']}, name == {r['name']}, statistic == {r['statistic']}")
    f = R / "discovery" / "extra_macro_contrasts_BP_metal_DOPT_Q.csv"
    if f.exists():
        ex = pd.read_csv(f)
        for _, r in ex.iterrows():
            for c in ("point", "ci95_low", "ci95_high", "ci9875_low", "ci9875_high", "p_two_sided", "wcr_wcr_p_wcr", "seeds_positive", "loco_all_positive"):
                if c in ex.columns and pd.notna(r.get(c)):
                    add(f"extra/{r['name']}/{c}", r[c], f, f"name == {r['name']}")
    # ladders
    cols = ["R2", "R2_ci_low", "R2_ci_high", "MAE", "MAE_ci_low", "MAE_ci_high", "RMSE", "MAE_macro_extractant", "MAE_macro_chemotype",
            "dir_acc_0", "dir_acc_0.1", "dir_acc_0.3", "n_rows", "n_cells", "n_extractants", "kish_chemotypes_pairs", "MAE_FLAT", "y_sd"]
    arms = ["FLAT@ZERO", "FLAT@BLUP20", "G14@ZERO", "G14@BLUP20", "G14@BLUP_INDEP", "NNC@ZERO", "NNC@BLUP20", "NNC@HEAD_FT", "NNC@SELECTED",
            "LINE", "PROJ", "RIDGE2", "NN_TANIMOTO@RESCALE", "MEAN_CURVE@BLUP20"]
    for design in ("BP", "B"):
        for a in arms:
            for k in (0, 1, 2, 3):
                levels("discovery", design, "metal", 3, "DOPT_Q", f"{a}|k{k}", cols)
            for k in (0, 3, 6):
                levels("discovery", design, "metal", 6, "DOPT_Q", f"{a}|k{k}", cols)
            for k in (0, 1, 3, 5, 8, 10):
                levels("discovery", design, "pair", 10, "DOPT_Q", f"{a}|k{k}", cols)
    for pol in ("RANDOM", "UNIFORM", "DOPT_CURVE", "DOPT_Q", "UNC_SEQ"):
        for K in (3, 6):
            for a in ("G14@BLUP20", "FLAT@BLUP20", "LINE", "RIDGE2", "NNC@SELECTED"):
                levels("discovery", "BP", "metal", K, pol, f"{a}|k{K}", ["R2", "MAE", "MAE_macro_extractant"])
    for design in ("A", "BR", "BQ", "BPG"):
        for a in ("G14@ZERO", "G14@BLUP20", "FLAT@BLUP20", "LINE", "RIDGE2", "NNC@SELECTED"):
            for k in (0, 3):
                levels("discovery_light", design, "metal", 3, "DOPT_Q", f"{a}|k{k}", ["R2", "MAE", "MAE_macro_extractant"])
    for name in ("sc009_only", "without_sc009"):
        f = R / "discovery" / f"levels_BP_metal_C3_DOPT_Q_{name}.csv"
        lv = pd.read_csv(f)
        for _, r in lv[lv.arm_base.isin(["G14@BLUP20", "LINE", "RIDGE2", "NNC@SELECTED", "FLAT@BLUP20"])].iterrows():
            for c in ("R2", "MAE", "MAE_macro_extractant", "n_cells", "n_extractants"):
                add(f"BP/metal/C3/DOPT_Q/{name}/{r['arm']}/{c}", float(r[c]), f, f"arm == {r['arm']}")
    f = R / "discovery" / "levels_BP_pair_C10_DOPT_Q_not_lab_known.csv"
    lv = pd.read_csv(f)
    for _, r in lv[lv.arm_base.isin(["G14@BLUP20", "LINE", "RIDGE2", "PROJ"])].iterrows():
        for c in ("R2", "MAE", "n_rows", "n_cells"):
            add(f"BP/pair/C10/DOPT_Q/not_lab_known/{r['arm']}/{c}", float(r[c]), f, f"arm == {r['arm']}")
    # per pair
    for a in ("G14-BLUP20_k0", "G14-BLUP20_k3", "LINE_k3", "NNC-SELECTED_k3", "RIDGE2_k3"):
        f = R / "discovery" / f"per_pair_BP_metal_C3_DOPT_Q_{a}.csv"
        if not f.exists():
            continue
        pp = pd.read_csv(f)
        for _, r in pp.iterrows():
            for c in ("n_rows", "n_extractants", "R2", "MAE", "y_sd", "MAE_FLAT", "dir_acc_0"):
                add(f"per_pair/BP/metal/C3/DOPT_Q/{a}/{r['pair']}/{c}", float(r[c]), f, f"pair == {r['pair']}")
    # exploratory family
    f = R / "discovery" / "exploratory_contrasts_BP.csv"
    ec = pd.read_csv(f)
    add("exploratory/BP/family_size", int(ec.family_size.iat[0]), f)
    add("exploratory/BP/n_bh_significant_0.05", int((ec.p_bh < 0.05).sum()), f)
    keep = ec[(ec.model == "metal") & (ec.statistic == "MAE") & (
        ec.name.str.contains("_vs_RANDOM@") | ec.name.str.contains("G14@BLUP20_vs_FLAT@BLUP20") | ec.name.str.contains("NNC@SELECTED_vs_G14@BLUP20")
        | ec.name.str.contains("G14@BLUP20_vs_LINE") | ec.name.str.contains("G14@BLUP20_vs_PROJ") | ec.name.str.contains("G14@BLUP_INDEP_vs_LINE"))]
    for _, r in keep.iterrows():
        for c in ("point", "ci95_low", "ci95_high", "p_two_sided", "p_bh", "seeds_positive", "loco_all_positive"):
            add(f"exploratory/BP/{r['name']}/{c}", r[c], f, f"name == {r['name']}")
    # setting B
    for name in ("settingB_levels_BP.csv", "settingB_contrasts_macro_BP.csv"):
        f = R / "discovery" / name
        t = pd.read_csv(f)
        key = "arm" if "arm" in t.columns else "name"
        for _, r in t.iterrows():
            for c in ("R2", "MAE", "MAE_macro_extractant", "dir_acc_0", "n_units", "n_extractants", "n_chemotypes", "point", "ci95_low", "ci95_high", "p_two_sided", "units_improved"):
                if c in t.columns and pd.notna(r.get(c)):
                    add(f"settingB/BP/{r[key]}/{c}", r[c], f, f"{key} == {r[key]}")
    # confirmation
    f = R / "confirmation" / "registered_contrasts_BP.csv"
    if f.exists():
        rc = pd.read_csv(f)
        for _, r in rc.iterrows():
            tag = f"confirmation/{r['name']}/{r['statistic']}"
            for c in ("point", "ci9875_low", "ci9875_high", "bca9875_low", "bca9875_high", "p_two_sided", "wcr_wcr_p_wcr", "mde_80", "seeds_positive", "loco_all_positive", "n_units", "n_clusters"):
                if c in rc.columns and pd.notna(r.get(c)):
                    add(f"{tag}/{c}", r[c], f, f"name == {r['name']}, statistic == {r['statistic']}")
        for a in ("G14@BLUP20", "RIDGE2", "LINE", "FLAT@BLUP20", "NNC@SELECTED", "G14@ZERO"):
            for k in (0, 3):
                levels("confirmation", "BP", "metal", 3, "DOPT_Q", f"{a}|k{k}", ["R2", "MAE", "MAE_macro_extractant"])
    out = G20 / "tables"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "report_numbers.csv", index=False)
    print(f"{len(rows)} numbers -> {out / 'report_numbers.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
