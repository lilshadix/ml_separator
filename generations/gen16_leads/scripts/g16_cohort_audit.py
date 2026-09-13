"""L6 — cohort audit.  Runs everything and writes gen16_leads/results/L6_cohort_audit/.

    python generations/gen16_leads/scripts/g16_cohort_audit.py

Never writes the frozen cohort.  Every relaxed cohort is described, never persisted over
``gen13_separation/manifests/``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GEN16_ROOT = Path(__file__).resolve().parents[1]
if str(GEN16_ROOT) not in sys.path:
    sys.path.insert(0, str(GEN16_ROOT))

from gen16 import cohort_audit as ca                                  # noqa: E402
from gen13sep import paths                                            # noqa: E402
from gen13sep.metrics import effective_sample_size                    # noqa: E402

OUT = GEN16_ROOT / "results" / "L6_cohort_audit"
OUT.mkdir(parents=True, exist_ok=True)

FROZEN_MODES = {"exact": "4c3c6628ea0be949", "relaxed": "97fb945371e30ac5",
                "series": "f0270097d692aaf3"}


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """The frozen recipe, after dropping the one diagnostic column gen16 adds."""
    fr = frame.drop(columns=[c for c in ("n_publications_in_cell",) if c in frame.columns])
    cols = sorted(c for c in fr.columns if c != "safe_exp_ids")
    payload = pd.util.hash_pandas_object(fr[cols], index=False).to_numpy().tobytes()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def main() -> None:
    log: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        log.append(s)

    raw = ca.load_bundle_raw()
    chem = pd.read_parquet(paths.CHEMISTRY_MAP_PARQUET).set_index("extractant")
    frozen = pd.read_parquet(paths.MANIFEST_DIR / "cohort_exact.parquet")
    kept_smiles = sorted(set(frozen["extractant"]))

    # ---- 0. reproduce the frozen builds -------------------------------------------
    say("== 0. frozen reproduction ==")
    repro = []
    for mode, fp in FROZEN_MODES.items():
        c = ca.build_cohort_switched(ca.Switches(key_mode=mode), bundle=raw)
        fz = pd.read_parquet(paths.MANIFEST_DIR / f"cohort_{mode}.parquet")
        got = frame_fingerprint(c.frame)
        repro.append({"key_mode": mode, "cells": c.audit["n_cells"],
                      "frozen_cells": int(len(fz)),
                      "extractants": c.audit["n_extractants"],
                      "frozen_extractants": int(fz["extractant"].nunique()),
                      "fingerprint": got, "frozen_fingerprint": fp, "matches": got == fp})
        say(f"  {mode:8s} cells={c.audit['n_cells']:4d} (frozen {len(fz)})  "
            f"extractants={c.audit['n_extractants']} (frozen {fz['extractant'].nunique()})  "
            f"fp={got} {'OK' if got == fp else 'MISMATCH'}")
    if not all(r["matches"] for r in repro):
        raise SystemExit("frozen reproduction failed - stop, do not trust any number below")

    kish_ext = ca.kish_by_extractant(frozen)
    kish_cell = ca.kish_by_cell(frozen)
    say(f"  Kish n_eff over chemotypes, by DISTINCT EXTRACTANTS per chemotype = {kish_ext:.4f}"
        "   <- this is gen13's published 11.7 (scripts/g13_analysis.py)")
    say(f"  Kish n_eff over chemotypes, by CELL counts                        = {kish_cell:.4f}")
    say()

    # ---- 1. trace every bundle extractant through the rules -------------------------
    say("== 1. exclusion trace over all 190 bundle extractants ==")
    trace = ca.trace_exclusions(bundle=raw)
    trace["n_cells_frozen_gt0"] = trace["n_cells_frozen"] > 0
    # cross-check the trace against the frozen cohort itself
    assert set(trace.loc[trace["kept"], "extractant"]) == set(kept_smiles), \
        "trace 'kept' set disagrees with the frozen cohort"
    say(f"  bundle extractants                                        {len(trace)}")
    say(f"  with >= 2 distinct lanthanides measured ANYWHERE           "
        f"{int((trace['n_lanthanides_bundle'] >= 2).sum())}")
    say(f"  in the frozen cohort                                       {int(trace['kept'].sum())}")
    ge2_excluded = trace[(trace["n_lanthanides_bundle"] >= 2) & (~trace["kept"])]
    say(f"  >= 2 lanthanides but NOT in the frozen cohort              {len(ge2_excluded)}")
    say()
    say(trace["exclusion_rule"].value_counts().to_string())
    say()

    excluded = trace[~trace["kept"]].copy()

    # ---- 2. chemotype and ECFP4 Tanimoto of the excluded set -------------------------
    say("== 2. what the excluded compounds are ==")
    kept_ct = set(chem.loc[kept_smiles, "chem__supercluster"])
    nn = ca.nn_tanimoto(list(excluded["extractant"]), kept_smiles)
    nn = nn.merge(excluded[["extractant", "extractant_name", "n_rows_bundle",
                            "n_lanthanides_bundle", "chemotype", "chem_family",
                            "exclusion_rule"]], on="extractant", how="left")
    nn["chemotype_new"] = ~nn["chemotype"].isin(kept_ct)
    nn["nn_kept_name"] = nn["nn_kept_extractant"].map(chem["extractant_name"])
    nn["would_seed_new_chemotype_T_lt_0.7"] = nn["max_tanimoto_to_kept"] < 0.7
    nn = nn.sort_values("max_tanimoto_to_kept").reset_index(drop=True)

    # independent check that the frozen supercluster map IS single-linkage Tanimoto 0.7
    clus = ca.single_linkage_clusters(list(chem.index), 0.7)
    cl = pd.DataFrame({"extractant": list(clus), "recluster": [clus[s] for s in clus]})
    cl["frozen"] = cl["extractant"].map(chem["chem__supercluster"])
    agree = cl.groupby("recluster")["frozen"].nunique().eq(1).all() and \
        cl.groupby("frozen")["recluster"].nunique().eq(1).all()
    say(f"  frozen chem__supercluster == my single-linkage ECFP4 Tanimoto-0.7 clustering: {agree} "
        f"({cl['recluster'].nunique()} clusters vs {cl['frozen'].nunique()} frozen)")
    say(f"  excluded compounds                                         {len(nn)}")
    say(f"  chemotypes represented among the excluded                  "
        f"{nn['chemotype'].nunique()}")
    say(f"  of those, chemotypes ABSENT from the kept cohort           "
        f"{nn.loc[nn['chemotype_new'], 'chemotype'].nunique()}")
    say(f"  excluded compounds with max Tanimoto to kept < 0.7         "
        f"{int(nn['would_seed_new_chemotype_T_lt_0.7'].sum())}")
    say(f"  excluded compounds joining an existing chemotype (T>=0.7)  "
        f"{int((~nn['would_seed_new_chemotype_T_lt_0.7']).sum())}")
    say(f"  max Tanimoto to kept: median {nn['max_tanimoto_to_kept'].median():.3f}  "
        f"min {nn['max_tanimoto_to_kept'].min():.3f}  max {nn['max_tanimoto_to_kept'].max():.3f}")
    dup = nn[nn["max_tanimoto_to_kept"] >= 0.999]
    if len(dup):
        say(f"  ECFP4-identical to a kept extractant (T = 1.00): {len(dup)}")
        for _, r in dup.iterrows():
            say(f"    {r['extractant_name']}  ==  {r['nn_kept_name']}  (different canonical SMILES)")
    say()

    # ---- 3. relaxations, one rule at a time ------------------------------------------
    say("== 3. relaxations, ONE RULE AT A TIME ==")
    S = ca.Switches
    runs = [
        ("FROZEN", S(), "no relaxation"),
        ("relax_todga_quarantine", S(todga_quarantine=False),
         "keep the 129 rows whose structure is TODGA but whose name is not"),
        ("relax_sentinel", S(sentinel_drop=False), "keep the 3 rows at log D <= -6"),
        ("relax_publication_in_key", S(publication_in_key=False),
         "drop publication_id from the cell key (cells may mix publications)"),
        ("key_mode_relaxed", S(key_mode="relaxed"),
         "gen13's documented 'relaxed' key: drop contact time and metal concentration"),
        ("key_mode_series", S(key_mode="series"),
         "gen13's documented 'series' key: relaxed columns + experiment_series_id"),
        ("key_mode_none", S(key_mode="none"),
         "MAXIMAL: no condition columns in the key at all (chemically wrong; a bound)"),
        ("relax_require_chemotype", S(require_chemotype=False),
         "allow extractants missing from the frozen gen6 chemistry map"),
    ]
    rel_rows = []
    for name, sw, what in runs:
        c = ca.build_cohort_switched(sw, bundle=raw)
        a = c.audit
        rel_rows.append({
            "relaxation": name, "what_it_relaxes": what,
            "cells": a["n_cells"], "extractants": a["n_extractants"],
            "chemotypes": a["n_chemotypes"], "ecfp_clusters": a["n_ecfp_clusters"],
            "publications": a["n_publications"], "rows_in_cells": a["rows_in_cells"],
            "cells_with_all_14": a["cells_with_all_14"],
            "cells_mixing_publications": a["cells_mixing_publications"],
            "kish_neff_by_extractant": round(a["kish_n_eff_chemotypes_by_extractant"], 4),
            "kish_neff_by_cell": round(a["kish_n_eff_chemotypes_by_cell"], 4),
            "d_extractants_vs_frozen": a["n_extractants"] - 90,
            "d_chemotypes_vs_frozen": a["n_chemotypes"] - 45,
            "new_extractants": ";".join(sorted(set(c.frame["extractant"]) - set(kept_smiles))),
            "lost_extractants": ";".join(sorted(set(kept_smiles) - set(c.frame["extractant"]))),
            "fingerprint": frame_fingerprint(c.frame),
        })
        say(f"  {name:28s} cells={a['n_cells']:4d}  ext={a['n_extractants']:3d}  "
            f"chemotypes={a['n_chemotypes']:3d}  Kish={a['kish_n_eff_chemotypes_by_extractant']:.3f}  "
            f"(d_ext {a['n_extractants'] - 90:+d}, d_chemo {a['n_chemotypes'] - 45:+d})")
    relax = pd.DataFrame(rel_rows)
    say()
    say("  MIN_METALS_PER_CELL = 2 is not relaxable: a separation factor needs two metals "
        "in one cell.")
    say()

    # ---- 4. the counterfactual that the audit actually uncovers -----------------------
    say("== 4. counterfactual: what a SECOND lanthanide on the excluded compounds would buy ==")
    ext_per_ct = frozen.groupby("chemotype")["extractant"].nunique()

    def cf(subset: pd.DataFrame, name: str) -> dict:
        add = subset.groupby("chemotype")["extractant"].nunique()
        comb = ext_per_ct.add(add, fill_value=0)
        k = float(effective_sample_size(comb.to_numpy()))
        say(f"  {name:52s} +{len(subset):3d} ext -> {int(comb.sum()):3d} extractants / "
            f"{len(comb):3d} chemotypes / Kish {k:6.3f}  ({k / kish_ext:.2f}x)")
        return {"scenario": name, "n_added": int(len(subset)),
                "extractants": int(comb.sum()), "chemotypes": int(len(comb)),
                "kish_neff_by_extractant": round(k, 4),
                "kish_ratio_vs_frozen": round(k / kish_ext, 3)}

    nn_idx = nn.set_index("extractant")
    excl_new_ct = excluded[~excluded["chemotype"].isin(kept_ct)]
    excl_far = excluded[excluded["extractant"].map(
        nn_idx["would_seed_new_chemotype_T_lt_0.7"]).fillna(False)]
    one_per_new = excl_new_ct.sort_values("n_rows_bundle", ascending=False) \
        .drop_duplicates("chemotype")
    say(f"  {'FROZEN':52s}  {90:3d} extractants / {45:3d} chemotypes / "
        f"Kish {kish_ext:6.3f}  (1.00x)")
    cfs = [cf(excluded, "all 100 single-lanthanide compounds"),
           cf(excl_new_ct, "only those in chemotypes absent from the cohort"),
           cf(excl_far, "only those with max ECFP4 Tanimoto to kept < 0.7"),
           cf(one_per_new, "one compound per absent chemotype (cheapest batch)")]
    kish_cf = cfs[0]["kish_neff_by_extractant"]
    say("  HYPOTHETICAL: these need new laboratory measurements, not a filter change.")
    say(f"  Kish barely moves for 'all 100' because {int((excluded['chemotype'] == 'sc009').sum())}"
        f" of the 100 are themselves in sc009, the dominant diglycolamide chemotype "
        f"({int(ext_per_ct['sc009'])} of the cohort's 90 extractants).")
    pd.DataFrame(cfs).to_csv(OUT / "counterfactual_kish.csv", index=False)
    say()

    # ---- write ------------------------------------------------------------------------
    # one row per candidate rule: how many extractants it excludes, and what relaxing it
    # alone does to the cohort.  Rules that exclude nothing are reported with 0, not omitted.
    rel_idx = relax.set_index("relaxation")
    rule_to_relaxation = {
        "todga_name_quarantine": "relax_todga_quarantine",
        "sentinel_log_D_le_-6": "relax_sentinel",
        "missing_publication_id": None,               # 0 rows lack a publication_id
        "publication_in_key_splits_metals": "relax_publication_in_key",
        "exact_condition_key_splits_metals": "key_mode_none",
        "missing_chemotype_in_gen6_map": "relax_require_chemotype",
        "single_metal_in_bundle": None,               # not relaxable: a SF needs two metals
    }
    counts = trace["exclusion_rule"].value_counts()
    new_ct_by_rule = (trace[~trace["kept"]].groupby("exclusion_rule")["chemotype"]
                      .agg(lambda s: len(set(s) - kept_ct)))
    br_rows = []
    for rule, relname in rule_to_relaxation.items():
        row = {"rule": rule,
               "n_extractants": int(counts.get(rule, 0)),
               "n_new_chemotypes": int(new_ct_by_rule.get(rule, 0)),
               "relaxation_arm": relname if relname else "not relaxable",
               "cells_if_relaxed": np.nan, "extractants_if_relaxed": np.nan,
               "chemotypes_if_relaxed": np.nan, "kish_neff_if_relaxed": np.nan}
        if relname:
            r = rel_idx.loc[relname]
            row.update(cells_if_relaxed=int(r["cells"]),
                       extractants_if_relaxed=int(r["extractants"]),
                       chemotypes_if_relaxed=int(r["chemotypes"]),
                       kish_neff_if_relaxed=float(r["kish_neff_by_extractant"]))
        br_rows.append(row)
    br_rows.append({"rule": "kept", "n_extractants": int(counts.get("kept", 0)),
                    "n_new_chemotypes": 0, "relaxation_arm": "-",
                    "cells_if_relaxed": 521, "extractants_if_relaxed": 90,
                    "chemotypes_if_relaxed": 45, "kish_neff_if_relaxed": kish_ext})
    by_rule = pd.DataFrame(br_rows).sort_values("n_extractants", ascending=False)
    say("== exclusions by rule ==")
    say(by_rule.to_string(index=False))
    say()

    trace.to_csv(OUT / "excluded_extractants.csv", index=False)
    by_rule.to_csv(OUT / "exclusions_by_rule.csv", index=False)
    relax.to_csv(OUT / "relaxations.csv", index=False)
    nn.to_csv(OUT / "nn_tanimoto.csv", index=False)

    summary = {
        "n_bundle_extractants": int(len(trace)),
        "n_bundle_rows": int(len(raw)),
        "n_cohort_extractants": int(trace["kept"].sum()),
        "n_cohort_cells": 521,
        "cohort_fingerprint": frame_fingerprint(
            ca.build_cohort_switched(ca.Switches(), bundle=raw).frame),
        "n_with_ge2_lanthanides_anywhere": int((trace["n_lanthanides_bundle"] >= 2).sum()),
        "n_excluded_with_ge2_lanthanides": int(len(ge2_excluded)),
        "kish_neff_frozen_by_extractant": kish_ext,
        "kish_neff_frozen_by_cell": kish_cell,
        "max_new_independent_chemotypes_over_all_relaxations":
            int(max(0, relax["chemotypes"].max() - 45)),
        "max_new_extractants_over_all_relaxations":
            int(max(0, relax["extractants"].max() - 90)),
        "n_excluded_single_metal": int((trace["exclusion_rule"] ==
                                        "single_metal_in_bundle").sum()),
        "n_excluded_chemotypes_absent_from_cohort":
            int(nn.loc[nn["chemotype_new"], "chemotype"].nunique()),
        "n_excluded_T_lt_0.7": int(nn["would_seed_new_chemotype_T_lt_0.7"].sum()),
        "kish_counterfactual_if_2nd_metal_measured": kish_cf,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "run.log").write_text("\n".join(log), encoding="utf-8")
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
