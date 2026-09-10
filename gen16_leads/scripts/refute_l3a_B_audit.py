"""Third stage of the lens-B refutation of CLAIM L3A: subset intervals and a bookkeeping audit.

Checks every number the claim quotes against the lead's own CSVs, recomputes the registered
decision rule from those CSVs, and puts a chemotype-blocked interval on the two subsets that
moved the point estimate most (diglycolamides removed; candidates with 5-8 metals).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen16 import bootstrap                      # noqa: E402
from gen16 import l3_decision as L               # noqa: E402
from refute_l3a_B import Board, OUT, ext_meta, block_boot   # noqa: E402

LEAD = bootstrap.RESULTS / "L3"


def main() -> int:
    t0 = time.time()
    em = ext_meta()
    rows, notes = [], {}

    for d in L.DESIGNS:
        bd = Board(d, em)
        T = bd.T
        for lab, keep in (("drop_sc009", bd.chem != "sc009"),
                          ("nmet_5_8", (bd.nmet >= 5) & (bd.nmet <= 8)),
                          ("nmet_9_14", (bd.nmet >= 9) & (bd.nmet <= 14))):
            W = block_boot(bd.chem, keep)
            r = L.l3a_saved_weighted(W, bd.mats)
            sv = L.seed_macro(r["saved"], T.seeds)
            er = L.seed_macro(r["e_random"], T.seeds)
            p0 = L.l3a_saved_weighted(keep.astype(float)[None, :], bd.mats)
            pt = float(np.atleast_1d(L.seed_macro(p0["saved"], T.seeds))[0])
            per = float(np.atleast_1d(L.seed_macro(p0["e_random"], T.seeds))[0])
            with np.errstate(invalid="ignore", divide="ignore"):
                fr = sv / er
            rows.append(dict(design=d, subset=lab, point=pt, saved_frac=pt / per,
                             ci_low=L.pct(sv, 0.025), ci_high=L.pct(sv, 0.975),
                             p_two_sided=L.two_sided_p(sv),
                             frac_ci_low=L.pct(fr, 0.025), frac_ci_high=L.pct(fr, 0.975),
                             n_ext=int(keep.sum()), n_blocks=len(set(bd.chem[keep])),
                             n_tasks=int(np.isfinite(p0["saved"][0]).sum())))
        print(f"[{d}] subset intervals done t={time.time()-t0:.0f}s", flush=True)

    S = pd.DataFrame(rows)
    S.to_csv(OUT / "refute_l3a_B_subset_ci.csv", index=False)

    # ---------------- bookkeeping audit against the lead's own CSVs ----------------
    saved = pd.read_csv(LEAD / "l3a_saved.csv")
    con = pd.read_csv(LEAD / "l3a_contrasts.csv")
    tasks = pd.read_csv(LEAD / "tasks_summary.csv")
    reg = con[(con.family == "registered")].copy()
    notes["registered_rows_l3a"] = int(len(reg))
    notes["registered_comparisons"] = sorted(reg.comparison.unique().tolist())

    # the decision rule recomputed from the CSV, not from the report prose
    rule = []
    for r in reg.itertuples():
        rule.append(dict(design=r.design, point=r.point, ci_low=r.ci95_low, ci_high=r.ci95_high,
                         p_perm=r.p_perm, p_block=r.p_block_boot,
                         rule_met=bool(r.point > 0 and r.p_perm < 0.05 and r.ci95_low > 0),
                         csv_says=bool(r.passes_registered),
                         seeds_positive=int(r.seeds_positive),
                         loco_min=r.loco_min, loco_max=r.loco_max,
                         loco_stable=bool(r.loco_sign_stable), n_units=int(r.n_units),
                         n_ext=int(r.n_extractants), n_blocks=int(r.n_blocks)))
    notes["registered_rule_recomputed"] = rule
    notes["registered_rule_agrees_with_csv"] = bool(all(x["rule_met"] == x["csv_says"] for x in rule))

    # headline numbers quoted in the claim vs the CSV board
    bp = saved[(saved.design == "BP") & (saved.arm == "G14") & (saved.split == "all")].iloc[0]
    notes["claim_check_BP"] = {
        "e_random_csv": float(bp.e_random), "e_model_csv": float(bp.e_model),
        "saved_csv": float(bp.saved), "saved_frac_csv": float(bp.saved_frac),
        "claim_says": {"e_random": 6.91, "e_model": 4.92, "saved": 1.99, "frac": 0.288},
        "rounding_ok": bool(abs(round(float(bp.e_random), 2) - 6.91) < 1e-9
                            and abs(round(float(bp.e_model), 2) - 4.92) < 1e-9
                            and abs(round(float(bp.saved), 2) - 1.99) < 1e-9
                            and abs(round(float(bp.saved_frac), 3) - 0.288) < 1e-9),
        "e_random_minus_e_model": float(bp.e_random - bp.e_model),
        "matches_saved": bool(abs(float(bp.e_random - bp.e_model) - float(bp.saved)) < 1e-12)}
    hv = saved[(saved.arm == "HEAVIER_ALWAYS") & (saved.split == "all")]
    notes["heavier_always_all_zero"] = bool((hv.saved.abs() < 1e-15).all())
    notes["heavier_always_values"] = hv.saved.tolist()

    # per-seed signs straight from the board
    ps = {}
    for d in L.DESIGNS:
        r = saved[(saved.design == d) & (saved.arm == "G14") & (saved.split == "all")].iloc[0]
        cols = [c for c in saved.columns if c.startswith("saved_seed_")]
        ps[d] = {c.replace("saved_seed_", ""): float(r[c]) for c in cols}
    notes["per_seed_saved"] = ps
    notes["per_seed_min"] = {d: min(v.values()) for d, v in ps.items()}

    # exclusions consistent across designs and arms?
    notes["task_counts_identical_across_designs"] = bool(
        tasks.l3a_n_tasks.nunique() == 1 and tasks.l3a_n_candidate_slots.nunique() == 1)
    notes["l3a_min_candidates"] = tasks.l3a_min_candidates.unique().tolist()
    notes["l3a_n_dropped_lt5"] = tasks.l3a_n_dropped_lt5.unique().tolist()
    notes["frac_tasks_all_same_call"] = dict(zip(tasks.design, tasks.l3a_frac_tasks_all_same_call))
    notes["n_candidate_slots_no_call"] = dict(zip(tasks.design, tasks.l3a_n_candidate_slots_no_call))

    (OUT / "refute_l3a_B_audit_notes.json").write_text(json.dumps(notes, indent=2, default=str),
                                                       encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(S.to_string())
    print(json.dumps(notes, indent=2, default=str))
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
