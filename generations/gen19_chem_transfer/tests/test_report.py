"""The report, figure and decision generators on synthetic inputs (brief sections 20, 22, 25, 29, 34; pre-registration
sections 9, 10, 15-17, 19).

Nothing here reads a real table, a real fold or a discovery record: every input file is a tiny synthetic file written
to ``tmp_path`` with the column names of the real writers (``metrics.SUMMARY_COLUMNS``, ``discovery.contrast_rows`` +
``transfer.BootstrapResult.record``, ``calibration.coverage_by_category``, the scorer's ``decisions.json``).  The tests
assert that every number printed in the generated markdown is traceable to a cited file (or a literal of the sealed
text), that a removed input becomes ``not computed (input missing: ...)``, that no number is invented when every input
is absent, and that the figure functions write their PNG + data CSV and report their inputs.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.evaluation import calibration as EC
from gen19ct.evaluation import figures as FG
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import report as R
from gen19ct.manifest import write_csv, write_json, write_text

SCRIPTS = paths.G19_ROOT / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------- #
# synthetic inputs shaped like the real writers' outputs
# --------------------------------------------------------------------------------------------- #

def _summary_row(arm, design, metric, value, *, variant="primary", half="selection", seed="104729", seed_set="discovery",
                 status="registered", stratum="all", aggregation="unit_macro", n_units=20, n_rows=300, note="",
                 unit_reading="registered: hidden cell", scoring_filter="none", job=""):
    return {"job": job or f"{design}__{variant}", "design": design, "variant": variant, "half": half, "arm": arm, "seed": seed,
            "seed_set": seed_set, "status": status, "averaging_unit": "unit", "n_units": n_units, "n_rows": n_rows,
            "stratum": stratum, "metric": metric, "aggregation": aggregation, "role": "primary", "value": value,
            "note": note, "scoring_filter": scoring_filter, "unit_reading": unit_reading, "threshold_setting": "k10_p1_m3",
            "cluster_unit_registered": "system", "label": "discovery, optimistically biased (selection half)"}


def _contrast_rows(key, family, cand, comp, design, point, lo, hi, p, verdict, cluster_units, *, margin=0.1057,
                   learned=True):
    rows = []
    for i, cu in enumerate(cluster_units):
        rows.append({"family": family, "contrast": f"{cand} vs {comp}", "design": design, "candidate": cand, "comparator": comp,
                     "half": "selection", "seed_set": "discovery", "decision_seed": 104729, "margin": margin,
                     "r19_verdict_full": "UNDECIDED" if learned else verdict, "verdict_stop_rule": verdict,
                     "verdict_ladder": verdict, "verdict_freezing_screen": verdict, "verdict_items_1_5": verdict,
                     "verdict_full": "UNDECIDED" if learned else verdict, "tost_verdict_eps0.05": "PASS",
                     "tost_low_90": lo + 0.01, "tost_high_90": hi - 0.01, "label": "discovery, optimistically biased (selection half)",
                     "batching_label": "batched" if learned else "", "r19_item4": "NOT_EVALUATED" if learned else "VACUOUS",
                     "seeds_evaluated": "104729" if learned else "104729, 130363, 155921, 196613, 262147",
                     "sensitivity_set": R.ADDENDUM_LABEL if learned else "registered (full)",
                     "sensitivities_not_run": "loose_setting, V5-P" if learned else "", "reported_verdict": verdict,
                     "seed_deltas": "{}", "sensitivities": "{}", "cluster_unit": cu, "clustered": True, "decides": True,
                     "point": point, "percentile_low": lo - 0.002 * i, "percentile_high": hi + 0.002 * i, "bca_low": lo - 0.001,
                     "bca_high": hi + 0.001, "p_two_sided": p, "bootstrap_sd": 0.03, "mde_80": 0.084, "bias_z0": 0.0,
                     "acceleration": 0.0, "loco_min": point - 0.05, "loco_max": point + 0.05, "n_units": 18, "n_clusters": 12 - i,
                     "n_resamples": 10000, "bootstrap_seed": 19, "n_nan_draws": 0, "higher_is_better": False,
                     "primary_cluster_unit": i == 0, "key": key, "bh_family": "registered" if family != "exploratory" else "exploratory",
                     "p_bh": min(1.0, p * 3), "p_bh_full_family": min(1.0, p * 6), "bh_m": 6})
    return rows


PREREG_TEXT = """# Gen19 -- pre-registration (synthetic copy for tests)

## 8. Statistical procedure and decision rule R19

Margin epsilon = 0.05 log D. Bootstrap 10,000 resamples, seed 19. Floor 0.3. Seeds 104729, 130363.

## 9. Success thresholds

rho5 = 0.20; delta5 = max(0.05, 0.20 x (L5 - N0)); gamma5 = 0.05; eta5 = 0.02; min_Y Delta_Y >= −0.02;
Spearman rho <= −0.10; coverage 50 / 80 / 95 in [0.40, 0.60], [0.70, 0.90], [0.88, 0.99]; every category with
>= 20 scored cells in [0.65, 0.92]; S2(d) >= 0.80. S2(a) sign in >= 11 of the 13 systems.

## 10. Failure conditions

F3: 80 % coverage outside [0.60, 0.95], or 95 % coverage < 0.85.

## 17. Deviations from the brief forced by the data

1. **PC88A/P507, Cyanex 272, D2EHPA/P204 absent.** None is present.
2. **No pH column.** Not identifiable.
3. **Mechanism experts (section 5).** Not testable.

## 18. What is not claimed

Nothing.

Sealed SHA-256 of everything above this line: `abc`

## POST-HOC addendum 1 (2026-09-15, orchestrator; results seen: no)

The compute-driven reduction: 1,535.7 CPU-hours against 60 h.
"""


def write_inputs(root: Path, *, with_confirmation: bool = False) -> None:
    """The full synthetic input set."""
    ev = root / "evaluation"
    (ev / "discovery" / "decisions").mkdir(parents=True, exist_ok=True)
    write_json(ev / "preseal" / "difficulty.json", {
        "V5": {"L5": 0.767, "C5": 1.5262, "N0": 0.2386, "delta5": 0.1057, "n_cells_selection": 105, "n_systems_selection": 18,
               "lookup_comparator": {"choice": "B3i"}},
        "V5_PAIR": {"SF5_FLAT": 0.7986, "chosen_lookup_derived_logsf_mae": 0.3847, "DIR5": 0.9498},
        "V2": {"comparator_choice": "B3i", "L2_focus7_selection": 0.6032, "C2_B0_focus7_selection": 1.1012},
        "F08_descriptive_spearman": {"comparator": "B3i", "half": "selection",
                                     "values": {"nearest_radius_distance_A": {"rho": 0.21, "n_cells": 105}}}})
    pre = []
    for arm, v5, v1, v2 in (("B0", 1.5262, 1.142, 1.1012), ("B3i", 0.767, 1.024, 0.6032), ("B3x", 0.8269, 1.028, 0.679)):
        pre.append(_summary_row(arm, "V5", "mae", v5, seed_set="deterministic", seed=""))
        pre.append(_summary_row(arm, "V1", "mae", v1, variant="copy", seed_set="deterministic", seed="", unit_reading="registered: outer fold"))
        pre.append(_summary_row(arm, "V1", "mae", v1 + 0.02, variant="copy", seed_set="deterministic", seed="",
                                unit_reading="exploratory: publication group", status="exploratory"))
        pre.append(_summary_row(arm, "V2", "mae", v2, variant="element", seed_set="deterministic", seed="", unit_reading="registered: metal state"))
        for pct, cov in ((50, 0.52), (80, 0.81), (95, 0.95)):
            pre.append(_summary_row(arm, "V5", f"coverage_{pct}", cov))
        pre.append(_summary_row(arm, "V5", "width_80", 2.1))
    write_csv(pd.DataFrame(pre), root / "tables" / "preseal_summary.csv")
    write_csv(pd.DataFrame([_summary_row("B0", "V5-PAIR", "logsf_mae", 0.7986, seed="", seed_set="deterministic")]),
              root / "tables" / "preseal_pair_summary.csv")
    disc = []
    for arm, v5, v1, v2 in (("M2", 0.701, 1.001, 0.581), ("M1", 0.75, 1.02, 0.6), ("B6", 0.73, 1.03, 0.61), ("B6r0", 0.9, 1.1, 0.7),
                            ("B5", 0.8, 1.05, 0.65), ("FLAT_CAT", 0.95, 1.12, 0.72)):
        disc.append(_summary_row(arm, "V5", "mae", v5))
        disc.append(_summary_row(arm, "V1", "mae", v1, variant="copy", unit_reading="registered: outer fold"))
        disc.append(_summary_row(arm, "V2", "mae", v2, variant="element", unit_reading="registered: metal state"))
        for pct, cov in ((50, 0.48), (80, 0.79), (95, 0.94)):
            disc.append(_summary_row(arm, "V5", f"coverage_{pct}", cov))
            disc.append(_summary_row(arm, "V1", f"coverage_{pct}", cov + 0.01, variant="copy", unit_reading="registered: outer fold"))
        disc.append(_summary_row(arm, "V5", "width_80", 1.9))
        disc.append(_summary_row(arm, "V1", "width_80", 2.0, variant="copy", unit_reading="registered: outer fold"))
    disc.append(_summary_row("M2", "V5", "mae", 0.65, stratum="metal_class=lanthanide", n_units=12))
    disc.append(_summary_row("M2", "V5", "mae", 0.82, stratum="metal_class=actinide", n_units=6))
    write_csv(pd.DataFrame(disc), root / "tables" / "discovery_summary.csv")
    cov = []
    for cat, units, c80 in (("IN_DOMAIN", 25, 0.83), ("CROSS_METAL_LIGAND_TRANSFER", 60, 0.78), ("UNSUPPORTED", 4, 0.6)):
        for metric, val in (("coverage_80", c80), ("width_80", 1.8), ("coverage_50", 0.5), ("coverage_95", 0.93)):
            r = _summary_row("M2", "V5", metric, val, stratum=f"domain_status={cat}", n_units=units)
            r.update(category=cat, category_units=units, n_rows_domain_status_ambiguous_excluded=2)
            cov.append(r)
    write_csv(pd.DataFrame(cov), root / "tables" / "discovery_coverage_by_domain_status.csv")
    pcov = []
    for cat, units, c80 in (("IN_DOMAIN", 25, 0.8), ("CROSS_METAL_LIGAND_TRANSFER", 60, 0.77), ("UNSUPPORTED", 4, 0.55)):
        for metric, val in (("coverage_80", c80), ("width_80", 2.2)):
            r = _summary_row("B3i", "V5", metric, val, stratum=f"domain_status_candidate={cat}", n_units=units)
            r.update(category=cat, category_units=units)
            pcov.append(r)
    write_csv(pd.DataFrame(pcov), root / "tables" / "preseal_coverage_by_domain_status.csv")
    ps = [_summary_row("M2", "V5-PAIR", "logsf_mae", 0.612, note="n_pairs=8000"),
          _summary_row("M2", "V5-PAIR", "direction_accuracy_abs_ge_0.3", 0.71, note="n_pairs=6000; n_undefined=0")]
    write_csv(pd.DataFrame(ps), root / "tables" / "discovery_pair_summary.csv")
    rows = []
    rows += _contrast_rows("M2 vs B3i@V5", "primary", "M2", "B3i", "V5", 0.066, -0.01, 0.14, 0.09, "FAIL", ("system", "publication_group"))
    rows += _contrast_rows("B6 vs B3i@V5", "H1b", "B6", "B3i", "V5", 0.037, -0.03, 0.1, 0.3, "FAIL", ("system", "publication_group"))
    rows += _contrast_rows("M2 vs B0@V5", "S1(b)", "M2", "B0", "V5", 0.825, 0.6, 1.05, 0.0, "PASS", ("system", "publication_group"), margin=0.05)
    rows += _contrast_rows("M2 vs B6r0@V5", "S1(b)", "M2", "B6r0", "V5", 0.199, 0.1, 0.3, 0.001, "PASS", ("system", "publication_group"), margin=0.05)
    rows += _contrast_rows("M2 vs FLAT_CAT@V5", "H4", "M2", "FLAT_CAT", "V5", 0.249, 0.15, 0.35, 0.0, "PASS", ("system", "publication_group"))
    rows += _contrast_rows("M2 vs M0@V5", "H4", "M2", "M0", "V5", 0.099, -0.02, 0.2, 0.12, "FAIL", ("system", "publication_group"))
    rows += _contrast_rows("B6 vs B6r0@V5", "H4", "B6", "B6r0", "V5", 0.17, 0.08, 0.26, 0.002, "PASS", ("system", "publication_group"))
    rows += _contrast_rows("M2 vs B3@V1", "secondary designs", "M2", "B3", "V1", 0.023, -0.05, 0.09, 0.4, "FAIL", ("publication_group",), margin=0.05)
    rows += _contrast_rows("M2 vs B3i@V2", "secondary designs", "M2", "B3i", "V2", 0.022, -0.06, 0.1, 0.5, "FAIL", ("metal_state",), margin=0.05)
    rows += _contrast_rows("M1 vs M0@V5#ladder", "ladder", "M1", "M0", "V5", 0.05, -0.03, 0.13, 0.2, "FAIL", ("system", "publication_group"))
    rows += _contrast_rows("M2 vs M0@V5#ladder", "ladder", "M2", "M0", "V5", 0.099, -0.02, 0.2, 0.12, "FAIL", ("system", "publication_group"))
    write_csv(pd.DataFrame(rows), ev / "discovery" / "contrasts_registered.csv")
    write_csv(pd.DataFrame(_contrast_rows("B5 vs B3i@V5", "exploratory", "B5", "B3i", "V5", -0.033, -0.1, 0.03, 0.4, "FAIL",
                                          ("system", "publication_group"))), ev / "discovery" / "contrasts_exploratory.csv")
    write_csv(pd.DataFrame([{"family": "primary", "half": "selection", "seed_set": "discovery", "design": "V5", "stage": "discovery",
                             "contrast": "M2 vs B3i", "point": 0.066, "margin": 0.1057, "verdict": "UNDECIDED", "item": 1,
                             "name": "point_estimate_at_least_margin", "status": "FAIL", "detail": "", "key": "M2 vs B3i@V5"}]),
              ev / "discovery" / "r19_items.csv")
    stop = {"rule": "section 7 item 4", "stop": True, "M2_vs_B3i": {"verdict": "FAIL"}, "B6_vs_B3i": {"verdict": "FAIL"},
            "consequences": ["M3-M6 not run as registered steps", "M7 run once", "H7 not run"], "pending": []}
    write_json(ev / "discovery" / "decisions" / "stop_rule.json", stop)
    comp = lambda point, margin, verdict: {"point": point, "margin": margin, "r19_full": "UNDECIDED", "reported_verdict": verdict,  # noqa: E731
                                           "batching_label": "batched", "scopes": {"stop_rule": verdict, "ladder": verdict,
                                                                                    "freezing_screen": verdict, "full": "UNDECIDED"},
                                           "items": []}
    decisions = {
        "schema": "gen19.discovery.v1", "delta5": 0.1057, "stop_rule": stop,
        # this fixture exercises the section 10 F6 FALLBACK (the deployed arm is the lookup comparator B3i), which POST-HOC
        # addendum 3 item 2 reaches only when the ladder has NO retained step -- hence no M0 row here.  A ladder that
        # retains M0, as the real scorer's block does, deploys M0 and is covered by
        # test_deployed_predictor_reads_the_ladder_runner_state and tests/test_h3.py
        "ladder": {"M1": {"step": "M1", "predecessor": "M0", "kept": False},
                   "M2": {"step": "M2", "predecessor": "M0", "kept": False}, "M3+": {"kept": None}},
        "S1_components": {"S1_forced_undecided": False, "S1a_M2_vs_B3i": comp(0.066, 0.1057, "FAIL"),
                          "S1b_M2_vs_B0": comp(0.825, 0.05, "PASS"), "S1b_M2_vs_B6r0": comp(0.199, 0.05, "PASS"),
                          "S1c_selection": {"min_delta": 0.012, "min_delta_yardstick": "HEAVIER", "n_pairs": 6000,
                                            "verdict_with_confirmation_missing": "UNDECIDED", "reported_verdict": "UNDECIDED",
                                            "direction": {"HEAVIER": {"delta": 0.012, "n_pairs": 4000, "interval_excludes_zero": False},
                                                          "B3i": {"delta": 0.041, "n_pairs": 6000, "interval_excludes_zero": True}},
                                            "logsf_mae": {"FLAT": {"gain": 0.186}, "B3i": {"gain": -0.02}}}},
        "not_run": {"M3+": "M3+ not implemented", "V0": "V0 not run in discovery", "confirmation": "never read",
                    "registered_family": [{"family": "H5", "contrast": "M3 vs M2", "design": "V5", "status": "not_run: M3-M6 not implemented"}],
                    "uncertainty_metrics": {"gaussian_crps": "NOT_RUN: no predictive SD"},
                    "addendum_1": {"discovery_seeds": [130363, 155921, 196613, 262147], "r19_item_4": "NOT_EVALUATED",
                                   "sensitivities": {"V5": ["loose_setting", "V5-P"]}, "v5p_heavy_arm_runs": "not run",
                                   "comparator_interval_jobs": "not run"}},
        "registered_family_accounting": {"m_full": 60, "m_discovery": 57, "m_evaluated": 11, "contrasts": [
            {"family": "primary", "contrast": "M2 vs B3i", "design": "V5", "status": "evaluated"},
            {"family": "H1b", "contrast": "B6 vs B3i", "design": "V5", "status": "evaluated"},
            {"family": "H4", "contrast": "M3 vs M2", "design": "V5", "status": "not_run: M3+ not implemented"},
            {"family": "ladder", "contrast": "M2 vs retained predecessor", "design": "V5", "status": "evaluated"},
            {"family": "confirmation", "contrast": "S2(a)", "design": "V6", "status": "confirmation_only"}]},
        "power_check": {"required_before_reporting_a_null": [{"contrast": "M2 vs B3i@V5", "family": "primary"}]},
        "readings": {"m_target_scaling": "the reading ... (task X finding V-08; needs a POST-HOC addendum)"},
        "plan_state": {"v5_batched_check": "passed"}}
    write_json(ev / "discovery" / "decisions" / "decisions.json", decisions)
    write_json(ev / "discovery" / "decisions" / "plan_state.json",
               {"v5_batched_check": "passed", "v1_tenfold_check": "failed", "heavy_v5_scheme": "batched", "heavy_v5_label": "batched",
                "heavy_v1_scheme": "exact", "freezing_candidates": []})
    write_json(ev / "discovery" / "decisions" / "wall_clock.json",
               {"total_hours": 31.25, "budget": {"budget_hours": 60.0, "exhausted": False}, "invocations": []})
    write_json(ev / "ladder" / "decisions" / "ladder.json",
               {"stop_rule": True, "discovery_ladder": {"M1": False, "M2": False},
                "steps": {**{s: {"step": s, "status": "exploratory_not_run", "kept": None} for s in ("M3", "M4", "M5", "M6")},
                          "M7": {"step": "M7", "status": "judged", "kept": False, "predecessor": "M0"}}, "demoted": [], "notes": []})
    write_json(ev / "ladder" / "M7" / "metrics.json",
               {"predecessor": "M0", "verdict": {"kept": False, "calibration_s1d_pass": True, "mae_non_inferior_all_designs": False},
                "designs": {"V5": {"interval_metrics": {"coverage_50": 0.51, "coverage_80": 0.8, "coverage_95": 0.951, "crps": 0.41},
                                   "knows_when_it_does_not_know": {"established": False, "spearman_pass": True, "coverage_gap_pass": False},
                                   "spearman_abs_error_sd": {"point": 0.18, "low_95": 0.05, "high_95": 0.3}}}})
    write_json(ev / "h3" / "h3_verdicts.json",
               {"M2": {"verdict": "UNDECIDED", "v5_with_beats_without": False, "v5_with_beats_permuted": False,
                       "v1_v2_non_inferior": {"V1": {"WITHOUT": True}}, "tost_v5_with_minus_without": {"verdict": "UNDECIDED"},
                       "kappa_min": 0.5, "kappa_status": "computed", "underpowered": True}})
    write_json(ev / "h3" / "h3_f4.json", {"failure": False, "status": "computed", "deployed_arm": "B3i", "designs_computed": ["V5"]})
    write_json(ev / "h3" / "h3_summary.json", {"deployed": {"arm": "B3i"}, "readings": {"tuning": "INFERRED; needs a POST-HOC addendum"}})
    write_csv(pd.DataFrame([{"model_arm": "M2", "contrast": "M2:WITH vs M2:WITHOUT", "design": "V5", "cluster_unit": "system",
                             "point": 0.011, "percentile_low": -0.04, "percentile_high": 0.06, "p_two_sided": 0.62,
                             "verdict_freezing_screen": "FAIL", "r19_verdict_full": "UNDECIDED", "primary_cluster_unit": True}]),
              ev / "h3" / "h3_contrasts.csv")
    write_json(ev / "power" / "power_checks.json", {"checks": [{"contrast": "M2 vs B3i@V5", "kappa_min": 0.5, "verdict": "UNDECIDED_UNDERPOWERED"}]})
    write_csv(pd.DataFrame([{"quantity": "support_score_components", "unit": "support_s1 per cell", "method": "split_half",
                             "status": "computed", "reliability": 0.91, "floor": 0.3, "gate": "PASS"},
                            {"quantity": "support_score_components", "unit": "support_s7 per cell", "method": "split_half",
                             "status": "computed", "reliability": 0.44, "floor": 0.3, "gate": "PASS"}]),
              root / "tables" / "reliability_before_correlation.csv")
    write_csv(pd.DataFrame([{"arm": "B3i", "design": "V5", "half": "selection", "seed": "deterministic", "x": "support_score",
                             "rho": -0.31, "percentile_low": -0.52, "percentile_high": -0.08, "n_cells": 105, "n_clusters": 18}]),
              root / "tables" / "s1e_error_vs_support.csv")
    write_text(root / "preregistration.md", PREREG_TEXT)
    write_text(root / "manifests" / "prereg_sha256.txt", "135842499a86eb3d478ece01a45718ac5b9673a4134bb4acda550f975bb45641\n")
    write_text(root / "manifests" / "confirmation_seeds_sha256.txt", "65e8ae8ceb8e92947a1da27f42e2689e74b04cc7b820d2b6df26889ceabf5f82\n")
    write_json(root / "manifests" / "g19_score_discovery.json", {"script": "g19_score_discovery", "git_head": "3343d3f", "outputs": [{}, {}]})
    write_csv(pd.DataFrame([{"path": "dataset_all_metals/clean/master_clean.parquet", "sha256": paths.ARCHIVE_MASTER_SHA256,
                             "is_headline_dataset_hash": True},
                            {"path": "dataset_all_metals/README.md", "sha256": "ab" * 32, "is_headline_dataset_hash": False}]),
              root / "data_audit" / "dataset_hashes.csv")
    if with_confirmation:
        # the confirmation decision file of the process runner's schema (gen19.confirmation.v1) plus the per-component
        # blocks and the claim list the confirmation orchestrator will add
        write_json(ev / "confirmation" / "decisions" / "confirmation.json",
                   {"schema": "gen19.confirmation.v1", "S1": {"passed": False, "deployed_predictor": "B3i"},
                    "V6": {"run": True}, "seeds": {"verified": True},
                    "claims": [{"contrast_key": "M2 vs B0@V5", "confirmed": True, "r19_verdict": "PASS", "point": 0.8,
                                "percentile_low": 0.6, "percentile_high": 1.0, "seeds_positive": 5, "n_seeds": 5}],
                    "S2": {"passed": False, "a": {"pass": True, "n_sign_agree": 12, "n_systems": 13},
                           "b": {"pass": False, "logsf_mae": 0.7, "flat": 0.72}, "c": {"pass": True, "coverage_80": 0.81, "coverage_95": 0.93}}})
        write_csv(pd.DataFrame([{"system": f"S{i}", "sign_agrees": i != 3, "n_pairs": 10 + i} for i in range(13)]),
                  ev / "confirmation" / "v6_systems.csv")
        # the process runner's outputs (an --exploratory run: everything labelled transfer-unsupported)
        winners = [{"cell": "p0.95_r0.80", "candidate": 17, "support_rank": 1, "p_both_feasible": 0.62, "p_both": 0.7,
                    "p_feasible": 0.9, "n_stages_total": 12, "s2d_fraction_kept": 0.85, "s2d_passes": True,
                    "statuses_used": "IN_DOMAIN|INTERPOLATION", "label": "transfer-unsupported"},
                   {"cell": "p0.99_r0.90", "candidate": 3, "support_rank": 2, "p_both_feasible": 0.31, "p_both": 0.4,
                    "p_feasible": 0.8, "n_stages_total": 18, "s2d_fraction_kept": 0.55, "s2d_passes": False,
                    "statuses_used": "IN_DOMAIN|CONDITION_EXTRAPOLATION", "label": "transfer-unsupported"}]
        write_csv(pd.DataFrame(winners), root / "tables" / "process_winners.csv")
        write_json(ev / "process" / "summary.json",
                   {"schema": "gen19.process.v1", "label": {"label": "transfer-unsupported", "headline_allowed": False,
                                                           "reasons": ["--exploratory run", "S1 not passed / not decided"]},
                    "gate": {"exploratory": True, "confirmation": {"s1_passed": False}}, "case": {"system": "sys_5cb78e5000d40860"},
                    "winners": winners, "s2d": {"p0.95_r0.80": {"fraction_kept": 0.85, "passes": True},
                                                "p0.99_r0.90": {"fraction_kept": 0.55, "passes": False}},
                    "f5": {"F5": False}, "readings": {"x": "INFERRED; needs a POST-HOC addendum"}})
        write_json(ev / "process" / "f5.json", {"label": "transfer-unsupported", "F5": False, "F5_i_any_cell": False, "F5_ii_any_cell": False})
        write_json(ev / "process" / "stability.json", {"label": "transfer-unsupported", "threshold": 0.8, "cells": {}})


# --------------------------------------------------------------------------------------------- #
# number extraction from the markdown
# --------------------------------------------------------------------------------------------- #

_STRIP = (re.compile(r"`[^`]*`"), re.compile(r"\d{4}-\d{2}-\d{2}"), re.compile(r"\b[0-9a-f]{12,}\b"),
          re.compile(r"\b[A-Za-z]+\d+[A-Za-z0-9_().-]*"),
          re.compile(r"(?:sections?|§|items?|Q)\s*\d+(?:\.\d+)*(?:\s*(?:/|,|and)\s*\d+(?:\.\d+)*)*"),
          re.compile(r"\b\d+\s*%"), re.compile(r"#+ "), re.compile(r"\bR\^2\b"))
_NUM = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w.])")
_INT = re.compile(r"(?<![\w.,-])\d{4,}(?![\w.])")


def markdown_numbers(text: str) -> tuple[set[str], set[str]]:
    """Decimal tokens and >= 4-digit integer tokens of a markdown text after stripping code spans, dates, digests,
    identifiers (V5, M2, B3i, F07 ...) and section / item references."""
    t = text
    for rx in _STRIP:
        t = rx.sub(" ", t)
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    return set(_NUM.findall(t)), set(_INT.findall(t))


def prereg_literals(root: Path) -> str:
    p = root / "preregistration.md"
    if not p.exists():
        return ""
    return re.sub(r"(?<=\d),(?=\d{3})", "", p.read_text(encoding="utf-8").replace("−", "-"))


def assert_traceable(root: Path, text: str, numbers: pd.DataFrame) -> None:
    printed = set(numbers["printed"].astype(str))
    lit = prereg_literals(root)
    decimals, ints = markdown_numbers(text)
    untraced = sorted(d for d in decimals if d not in printed and d not in lit)
    assert not untraced, f"decimal numbers without a cited source or a sealed-text literal: {untraced[:10]}"
    untraced_i = sorted(i for i in ints if i not in printed and i not in lit)
    assert not untraced_i, f"integers without a cited source or a sealed-text literal: {untraced_i[:10]}"


# --------------------------------------------------------------------------------------------- #
# the report generator
# --------------------------------------------------------------------------------------------- #

@pytest.fixture()
def full_root(tmp_path: Path) -> Path:
    write_inputs(tmp_path, with_confirmation=True)
    return tmp_path


def test_every_number_in_the_report_is_traceable_and_re_resolves(full_root: Path) -> None:
    res = R.build_report(full_root, extra={"git_head": "deadbeef0000"})
    outs = R.write_outputs(full_root, res)
    assert (full_root / "GEN19_REPORT.md").exists() and (full_root / "SUMMARY.md").exists()
    assert (full_root / "decisions" / "D02_factorization.md").exists() and (full_root / "tables" / "claims.json").exists()
    numbers = res["numbers"]
    assert len(numbers) > 40
    for text in (res["report"], res["summary"], res["d02"]):
        assert_traceable(full_root, text, numbers)
    ver = R.verify_numbers(full_root, numbers)
    bad = ver[~ver["ok"].astype(bool)]
    assert bad.empty, bad.to_dict("records")[:5]
    # the CSV / JSON sourced numbers really come from those files (kinds), constants from the sealed text only
    kinds = set(numbers["kind"])
    assert kinds <= {"value", "count", "constant", "text"}
    assert (numbers.loc[numbers["kind"] == "constant", "source_path"] == R.PREREG).all()
    assert all(Path(p).exists() for p in outs)


def test_section_order_headlines_and_labels(full_root: Path) -> None:
    res = R.build_report(full_root, extra={"git_head": "x"})
    rep = res["report"]
    positions = [rep.index(f"## {qid}.") for qid, _ in R.QUESTIONS]
    assert positions == sorted(positions), "the ten questions of brief section 34 must appear in order"
    for start in ("## Verdict table", "## What was not run and why", "## Deviations from the brief", "## Reproducibility"):
        assert start in rep and rep.index(start) > positions[-1]
    # every question is answered in one sentence first
    for qid, _ in R.QUESTIONS:
        block = rep.split(f"## {qid}.")[1].split("\n## ")[0]
        assert block.strip().splitlines()[2].startswith("**Under deliberately hidden chemistry, the system ")
    # regime columns, the two comparators, the FLAT floor, BH beside raw p, the addendum label, statuses
    for token in ("| design | half | seeds | averaging unit | parameter status |", "constant baseline", "cheapest sensible alternative",
                  "FLAT floor", "p_BH", R.ADDENDUM_LABEL, "| discovery |", "| not run |", "| confirmation |", "TOST"):
        assert token in rep, token
    assert "reduced sensitivity set (addendum 1) -- not run: loose_setting, V5-P" in rep
    # the confirmed claim is labelled confirmation, the H1 claim discovery
    claims = {c["key"]: c for c in res["claims"]["claims"]}
    assert claims["M2 vs B0@V5"]["status"] == "confirmation" and claims["M2 vs B0@V5"]["confirmation"]["confirmed"] is True
    assert claims["M2 vs B3i@V5"]["status"] == "discovery" and claims["M3 vs M2@V5"]["status"] == "not run"
    # the S2 verdicts come from the confirmation file
    verdicts = {r["criterion"]: r for r in res["claims"]["verdicts"]}
    assert verdicts["S2(a)"]["verdict"] == "PASS" and verdicts["S2(b)"]["verdict"] == "FAIL" and verdicts["S2(a)"]["status"] == "confirmation"
    assert verdicts["S1(a)"]["verdict"] == "FAIL" and verdicts["F6"]["verdict"].startswith("HOLDS")
    assert verdicts["S1(e)"]["verdict"] == "PASS" and verdicts["S1(d)"]["verdict"] == "PASS"
    assert verdicts["S1 (confirmation)"]["verdict"] == "FAIL" and verdicts["S2 (overall)"]["verdict"] == "FAIL"
    assert verdicts["F5"]["verdict"] == "does not hold" and "transfer-unsupported" in verdicts["F5"]["status"]
    # Q9 / Q10 from the process runner's files: the label, the per-cell S2(d), the support classification
    assert "every process result is labelled transfer-unsupported" in rep
    assert "p0.95_r0.80: fraction kept 0.85, passes True" in rep
    assert "1 directly supported, 0 transfer-supported, 1 speculative, 0 unsupported" in rep
    assert (full_root / "tables" / "process_support_status.csv").exists() is False     # written by write_outputs, not build
    # deviations and addenda parsed from the sealed text; readings needing an addendum collected
    assert "1. PC88A/P507, Cyanex 272, D2EHPA/P204 absent." in rep and "addendum 1 (2026-09-15" in rep
    assert "needs a POST-HOC addendum" in rep


def test_removed_input_becomes_not_computed(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    for p in ("evaluation/h3/h3_verdicts.json", "evaluation/h3/h3_f4.json", "evaluation/h3/h3_contrasts.csv",
              "evaluation/ladder/M7/metrics.json", "tables/s1e_error_vs_support.csv"):
        (tmp_path / p).unlink()
    res = R.build_report(tmp_path, extra={"git_head": "x"})
    rep = res["report"]
    for p in ("evaluation/h3/h3_verdicts.json", "evaluation/h3/h3_f4.json", "evaluation/ladder/M7/metrics.json",
              "tables/s1e_error_vs_support.csv", "evaluation/confirmation/decisions/confirmation.json",
              "evaluation/process/summary.json", "tables/process_winners.csv"):
        assert f"not computed (input missing: {p}" in rep, p
    # no number of the removed files survives in the ledger, and the report stays traceable
    src = set(res["numbers"]["source_path"])
    assert not src & {"evaluation/h3/h3_verdicts.json", "evaluation/h3/h3_contrasts.csv", "evaluation/ladder/M7/metrics.json",
                      "tables/s1e_error_vs_support.csv"}
    assert_traceable(tmp_path, rep, res["numbers"])
    assert R.verify_numbers(tmp_path, res["numbers"])["ok"].all()
    verdicts = {r["criterion"]: r for r in res["claims"]["verdicts"]}
    assert verdicts["F4"]["status"] == "not run" and "not computed (input missing" in verdicts["F4"]["verdict"]
    assert verdicts["S1(e)"]["status"] == "not run" and verdicts["S2(a)"]["status"] == "not run"
    assert sorted(res["inputs"]["missing_inputs"])[:1] == ["evaluation/confirmation/decisions/confirmation.json"]


def test_empty_root_invents_no_number(tmp_path: Path) -> None:
    res = R.build_report(tmp_path, extra={})
    rep = res["report"]
    assert "not computed (input missing: evaluation/discovery/decisions/decisions.json" in rep
    assert "not computed (input missing: evaluation/preseal/difficulty.json" in rep
    numbers = res["numbers"]
    # nothing was read from a data file: only registered constants (cited to the sealed text) may be printed
    assert set(numbers["kind"]) <= {"constant"}
    assert all(c["status"] == "not run" for c in res["claims"]["claims"])
    assert all(r["status"] == "not run" for r in res["claims"]["verdicts"])
    for h in res["claims"]["headlines"]:
        assert h["headline"].startswith("Under deliberately hidden chemistry, the system has not been shown to"), h
    assert res["inputs"]["inputs"]["decisions"]["exists"] is False
    R.write_outputs(tmp_path, res)
    assert (tmp_path / "SUMMARY.md").exists()


def test_summary_is_one_page_and_d02_has_the_section_29_format(full_root: Path) -> None:
    res = R.build_report(full_root, extra={"git_head": "x"})
    summ = res["summary"]
    assert len(summ.splitlines()) <= 60
    assert sum(1 for ln in summ.splitlines() if ln[:3].strip(". ").isdigit() and "**" in ln) == 10
    assert "| criterion | verdict | status |" in summ
    d02 = res["d02"]
    order = ["## question", "## evidence", "## metrics", "## null / supported / ambiguous", "## decision", "## next action"]
    pos = [d02.index(h) for h in order]
    assert pos == sorted(pos)
    # task X finding V-P01: a FAIL prints as a null only where its own power record screened it as an informative null.
    # This fixture's power check returns UNDECIDED_UNDERPOWERED for H1, so H1 reads UNDECIDED, not "null".
    assert "H1 (M2 vs B3i@V5): **UNDECIDED (underpowered)**" in d02
    assert "H4 factorised vs flat (M2 vs FLAT_CAT@V5): supported" in d02
    assert "H4 + mechanism (M3 vs M2@V5): ambiguous / not run" in d02
    assert "deployed predictor: **B3i**" in d02
    assert_traceable(full_root, d02, res["numbers"])


def test_claims_json_schema(full_root: Path) -> None:
    res = R.build_report(full_root, extra={"git_head": "x"})
    R.write_outputs(full_root, res)
    body = json.loads((full_root / "tables" / "claims.json").read_text(encoding="utf-8"))
    assert body["schema"] == R.SCHEMA and len(body["claims"]) >= 5
    con = pd.read_csv(full_root / "evaluation" / "discovery" / "contrasts_registered.csv")
    assert set(con["key"]) <= {c["key"] for c in body["claims"]}, "every evaluated registered contrast is a claim"
    need = {"claim", "design", "status", "comparator", "delta", "interval_percentile_95", "r19_verdict_full", "reported_verdict",
            "source_files", "half", "seeds", "averaging_unit", "parameter_status", "sensitivity_set", "comparators_required"}
    for c in body["claims"]:
        assert need <= set(c), sorted(need - set(c))
        assert c["status"] in ("discovery", "confirmation", "not run")
        assert c["comparators_required"]["constant_baseline"] == "B0"
    h1 = next(c for c in body["claims"] if c["key"] == "M2 vs B3i@V5")
    assert h1["delta"] == pytest.approx(0.066) and h1["sensitivity_set"] == R.ADDENDUM_LABEL
    assert "evaluation/discovery/contrasts_registered.csv" in h1["source_files"]


def test_resolve_source_grammar(tmp_path: Path) -> None:
    write_json(tmp_path / "a.json", {"x": {"y": [1, {"z": 2.5}]}, "s2d": {"p0.95_r0.80": {"passes": True}}})
    write_csv(pd.DataFrame({"k": ["a", "b", "b"], "v": [1.0, 2.0, 3.0]}), tmp_path / "t.csv")
    write_text(tmp_path / "p.md", "margin 0.05 and 10,000 resamples and −0.02")
    assert R.resolve_source(tmp_path, "a.json", "x.y[1].z") == 2.5
    assert R.jpath("s2d", "p0.95_r0.80", "passes") == 's2d["p0.95_r0.80"].passes'
    assert R.resolve_source(tmp_path, "a.json", R.jpath("s2d", "p0.95_r0.80", "passes")) is True
    assert float(R.resolve_source(tmp_path, "t.csv", "where:k=a;column:v")) == 1.0
    assert R.resolve_source(tmp_path, "p.md", "count:addenda") == 0 and R.literal_in_file(tmp_path, "p.md", "-0.02")
    with pytest.raises(KeyError):
        R.resolve_source(tmp_path, "t.csv", "where:k=b;column:v")          # two rows: ambiguous
    assert R.resolve_source(tmp_path, "t.csv", "count:where:k=b") == 2 and R.resolve_source(tmp_path, "t.csv", "count:rows") == 3
    assert R.resolve_source(tmp_path, "p.md", "10000") == "10000" and R.resolve_source(tmp_path, "p.md", "-0.02") == "-0.02"
    with pytest.raises(FileNotFoundError):
        R.resolve_source(tmp_path, "nope.json", "x")
    ver = R.verify_numbers(tmp_path, pd.DataFrame([{"printed": "2.5", "raw_value": 2.5, "source_path": "a.json", "source_key": "x.y[1].z", "kind": "value", "rule": ""},
                                                   {"printed": "9.9", "raw_value": 9.9, "source_path": "a.json", "source_key": "x.y[1].z", "kind": "value", "rule": ""}]))
    assert list(ver["ok"]) == [True, False]


# --------------------------------------------------------------------------------------------- #
# the figures
# --------------------------------------------------------------------------------------------- #

def _pred_frame(n: int, seed: int, spread: float = 0.4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1.2, n)
    pred = y + rng.normal(0, spread, n)
    q = np.array([0.4, 0.8, 1.3])
    fr = pd.DataFrame({"row_id": [f"r{i}" for i in range(n)], "fold_id": [f"f{i % 6}" for i in range(n)], "log_D": y,
                       "mean_logD": pred, "metal_class": rng.choice(["lanthanide", "actinide", "other"], n),
                       "unit": [f"u{i % 12}" for i in range(n)]})
    for lvl, qq in zip((0.5, 0.8, 0.95), q):
        lo, hi = EM.interval_columns(lvl)
        fr[lo], fr[hi] = pred - qq, pred + qq
    fr["domain_status"] = rng.choice(list(EC.DOMAIN_STATUSES[:4]), n)
    return fr


def test_figures_write_png_and_data_and_report_inputs(tmp_path: Path) -> None:
    figs = tmp_path / "figures"
    frames = {d: {"M2": _pred_frame(120, i), "B3i": _pred_frame(120, i + 10, 0.7)} for i, d in enumerate(("V5", "V1", "V2"))}
    r7 = FG.fig07_pred_vs_measured(frames, figs, inputs=["evaluation/discovery/M2/V5__primary_batched/s104729/", "x.parquet"], deployed="M2")
    assert r7.status == "written" and Path(tmp_path, r7.path).exists() if not Path(r7.path).is_absolute() else Path(r7.path).exists()
    assert r7.inputs == ["evaluation/discovery/M2/V5__primary_batched/s104729/", "x.parquet"]
    assert (figs / "data" / "F07_pred_vs_measured.csv").exists()
    panels = pd.read_csv(figs / "data" / "F07_pred_vs_measured.csv")
    assert set(panels["arm"]) == {"M2", "B3i"} and set(panels["design"]) == {"V5", "V1", "V2"}
    mae_v5, n_units = FG.unit_macro_mae(frames["V5"]["M2"])
    assert panels.set_index(["arm", "design"]).loc[("M2", "V5"), "macro_mae"] == pytest.approx(mae_v5) and n_units == 12
    # F08 with the S1(e) statistic on a small resample count
    rng = np.random.default_rng(3)
    cells = pd.DataFrame({"unit": [f"c{i}" for i in range(30)], "system": [f"S{i % 6}" for i in range(30)],
                          "support_score": rng.uniform(0, 1, 30), "domain_status": rng.choice(list(EC.DOMAIN_STATUSES), 30),
                          "metal_class": "lanthanide"})
    cells["mae"] = 1.2 - 0.8 * cells["support_score"] + rng.normal(0, 0.1, 30)
    st = FG.error_vs_support_stats(cells, n_resamples=200)
    assert st["status"] == "computed" and st["rho"] < -0.5 and st["percentile_high"] < 0 and st["n_clusters"] == 6
    r8 = FG.fig08_error_vs_support({"M2": cells}, figs, inputs=["evaluation/discovery/_support/V5__primary_batched/s104729/"], stats={"M2": st})
    assert r8.status == "written" and (figs / "F08_error_vs_support.png").exists() and (figs / "data" / "F08_error_vs_support.csv").exists()
    assert r8.stats["per_arm"]["M2"]["rho"] == st["rho"]
    # F09
    r9 = FG.fig09_calibration({d: frames[d]["M2"] for d in ("V5", "V1", "V2")}, figs, inputs=["a"], arm="M2")
    assert r9.status == "written" and (figs / "F09_uncertainty_calibration.png").exists()
    cov = pd.read_csv(figs / "data" / "F09_uncertainty_calibration.csv")
    assert set(cov["level"]) == {0.5, 0.8, 0.95} and (cov["coverage_unit_macro"].between(0, 1)).all()
    assert FG.coverage_table(frames["V5"]["M2"])[0.95] == pytest.approx(cov[(cov["design"] == "V5") & (cov["category"] == "all") & (cov["level"] == 0.95)]["coverage_unit_macro"].iloc[0])
    # F10 / F11 with replicates
    states = ["La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)", "Am(III)", "Cm(III)", "Sr(II)", "Fe(III)"]
    base = np.linspace(0, 1, len(states))[:, None] * np.array([[1.0, 0.5, -0.3, 0.1]]) + rng.normal(0, 0.05, (len(states), 4))
    emb = pd.DataFrame(base, index=pd.Index(states, name="g19_metal_state"), columns=[f"e_m_{j:02d}" for j in range(4)])
    reps = [pd.DataFrame(base @ np.linalg.qr(rng.normal(size=(4, 4)))[0] + rng.normal(0, 0.03, base.shape), index=emb.index, columns=emb.columns)
            for _ in range(4)]
    r10 = FG.fig10_metal_embedding(emb, figs, inputs=["evaluation/power/embeddings/metal_embeddings.csv"], replicates=reps,
                                   stability={"stability": 0.9, "null_stability": 0.3, "gate": "PASS"})
    assert r10.status == "written" and r10.stats["n_replicates"] == 4 and (figs / "data" / "F10_metal_embedding.csv").exists()
    d10 = pd.read_csv(figs / "data" / "F10_metal_embedding.csv")
    assert set(d10[d10["class"] != "replicate"]["class"]) == {"lanthanide", "actinide", "other"}
    sysemb = pd.DataFrame(rng.normal(size=(9, 4)), index=pd.Index([f"S{i}" for i in range(9)], name="extractant_system_key"),
                          columns=[f"e_l_{j:02d}" for j in range(4)])
    sysemb["family"] = ["diglycolamide"] * 4 + ["monoamide"] * 3 + ["malonamide"] * 2
    r11 = FG.fig11_extractant_embedding(sysemb, figs, inputs=["evaluation/power/embeddings/system_embeddings.csv"])
    assert r11.status == "written" and r11.stats["n_families"] == 3
    # F12 from confirmation-shaped frames; F13 confusion counts
    rows = pd.DataFrame({"system": [f"S{i % 5}" for i in range(40)], "metal_state": ["Pr(III)", "Nd(III)"] * 20,
                         "log_D": rng.normal(size=40), "mean_logD": rng.normal(size=40)})
    rows["lower_80"], rows["upper_80"] = rows["mean_logD"] - 0.5, rows["mean_logD"] + 0.5
    pairs = pd.DataFrame({"system": [f"S{i % 5}" for i in range(50)], "observed_logsf": rng.normal(0.3, 0.5, 50)})
    pairs["predicted_logsf"] = pairs["observed_logsf"] * 0.8 + rng.normal(0, 0.2, 50)
    r12 = FG.fig12_prnd_reconstruction(rows, pairs, figs, inputs=["evaluation/confirmation/v6_rows.csv", "evaluation/confirmation/v6_pairs.csv"])
    assert r12.status == "written" and (figs / "F12_prnd_reconstruction.png").exists()
    cm = FG.direction_confusion(pd.DataFrame({"observed_logsf": [0.5, -0.5, 0.2, 0.8, -0.4], "predicted_logsf": [0.1, -0.2, 0.3, 0.0, 0.3]}), 0.3)
    assert cm["n_pairs"] == 4 and cm["counts"]["obs_pos_pred_pos"] == 1 and cm["counts"]["obs_pos_pred_zero"] == 1
    assert cm["counts"]["obs_neg_pred_neg"] == 1 and cm["counts"]["obs_neg_pred_pos"] == 1 and cm["direction_accuracy"] == pytest.approx(2.5 / 4)
    r13 = FG.fig13_direction_confusion({"V5-PAIR": pairs, "V6": pairs}, figs, inputs=["p.parquet"])
    # task X finding V-08: the registered 0.3 reading on EVERY design plus the additional 0.1 reading on V6 only
    per = r13.stats["per_design"]
    assert r13.status == "written" and [(d["design"], d["threshold"]) for d in per] == [("V5-PAIR", 0.3), ("V6", 0.3), ("V6", 0.1)]
    assert [d["registered_primary"] for d in per] == [True, True, False]
    d13 = pd.read_csv(figs / "data" / "F13_direction_confusion.csv")
    assert list(d13["threshold"]) == [0.3, 0.3, 0.1] and d13["registered_primary"].tolist() == [True, True, False]
    assert FG.DIRECTION_THRESHOLDS == {"V5-PAIR": (0.3,), "V6": (0.3, 0.1)} and FG.PRIMARY_DIRECTION_THRESHOLD == 0.3
    # every result records its inputs and the index counts them
    index = FG.figures_index([r7, r8, r9, r10, r11, r12, r13])
    assert index["n_written"] == 7 and index["n_skipped"] == 0 and all(f["inputs"] for f in index["figures"])


def test_figures_skip_with_a_reason_when_inputs_are_absent(tmp_path: Path) -> None:
    figs = tmp_path / "figures"
    results = [FG.fig07_pred_vs_measured({}, figs, inputs=[], deployed="M2"),
               FG.fig08_error_vs_support({}, figs, inputs=[]),
               FG.fig09_calibration({}, figs, inputs=[], arm="M2"),
               FG.fig10_metal_embedding(None, figs, inputs=[]),
               FG.fig11_extractant_embedding(None, figs, inputs=[]),
               FG.fig12_prnd_reconstruction(None, None, figs, inputs=[]),
               FG.fig13_direction_confusion({}, figs, inputs=[])]
    assert all(r.status == "skipped" and r.reason and r.path is None for r in results)
    assert not figs.exists() or not list(figs.glob("*.png"))
    assert "confirmation" in results[5].reason and "power" in results[3].reason
    st = FG.error_vs_support_stats(pd.DataFrame({"mae": [1.0, 2.0], "support_score": [0.1, 0.2], "system": ["a", "b"]}), n_resamples=10)
    assert st["status"].startswith("not computed") and np.isnan(st["rho"])


# --------------------------------------------------------------------------------------------- #
# the runners' gates (never touch the real discovery run)
# --------------------------------------------------------------------------------------------- #

def test_runners_refuse_without_the_seal_or_a_complete_discovery(tmp_path: Path) -> None:
    BR = _load("g19_build_report")
    MF = _load("g19_make_figures")
    from gen19ct.evaluation import discovery as _D
    from gen19ct.evaluation import registry as REG

    def _sealed(stage: str) -> dict:
        """The sealed digests a runner of ``stage`` expects: its registry entry (addendum 2 item 5), not a literal."""
        exp = REG.gate_expectations(stage)
        return {"footer": _D.REGISTERED_PREREG_SHA256, "recomputed": _D.REGISTERED_PREREG_SHA256,
                "digest_file": _D.REGISTERED_PREREG_SHA256, "addenda_sha256": exp["below_footer_sha256"],
                "n_addenda": exp["n_addenda"]}
    good = _sealed("report")
    assert _sealed("figures") == good        # both are registered under the same sealed text
    for mod, stage in ((BR, "report"), (MF, "figures")):
        good = _sealed(stage)
        with pytest.raises(SystemExit, match="--check exited 1"):
            mod.refuse_unless_ready(tmp_path, check=lambda: 1, digests=lambda: good, jobs=[], discovery_code="c")
        # the seal passes but no wall_clock.json reached the final stage: not COMPLETE
        with pytest.raises(SystemExit, match="not COMPLETE"):
            mod.refuse_unless_ready(tmp_path, check=lambda: 0, digests=lambda: good, jobs=[], discovery_code="c",
                                    state=__import__("gen19ct.evaluation.discovery", fromlist=["PlanState"]).PlanState())
    # a wall clock that reached the final stage with no fit jobs is complete for the report (the scorer may be absent),
    # but the ladder must have run every step M3-M7 (task X finding V-01)
    from gen19ct.evaluation import discovery as D
    from gen19ct.evaluation import h3 as H3
    write_json(tmp_path / "evaluation" / "discovery" / "decisions" / "wall_clock.json",
               {"invocations": [{"seconds": 1.0, "stages_done": [D.STAGES["not_implemented"]]}]})
    with pytest.raises(SystemExit, match="ladder .M3-M7. is not complete"):
        BR.refuse_unless_ready(tmp_path, check=lambda: 0, digests=lambda: good, jobs=[], discovery_code="c", state=D.PlanState())
    gate0 = BR.refuse_unless_ready(tmp_path, check=lambda: 0, digests=lambda: good, jobs=[], discovery_code="c", state=D.PlanState(),
                                   require_ladder=False)
    assert gate0["ladder_complete"]["complete"] is False and gate0["ladder_complete"]["present"] is False
    write_json(tmp_path / H3.LADDER_STATE_FILE,
               {"stop_rule": True, "discovery_ladder": {"M1": False, "M2": False},
                "steps": {**{s: {"step": s, "status": "exploratory_not_run", "kept": None} for s in ("M3", "M4", "M5", "M6")},
                          "M7": {"step": "M7", "status": "judged", "kept": False}}})
    gate = BR.refuse_unless_ready(tmp_path, check=lambda: 0, digests=lambda: good, jobs=[], discovery_code="c", state=D.PlanState())
    assert gate["discovery_complete"]["complete"] is True and gate["scorer_decisions"]["ok"] is False
    assert gate["ladder_complete"]["complete"] is True
    with pytest.raises(SystemExit, match="scorer"):
        MF.refuse_unless_ready(tmp_path, check=lambda: 0, digests=lambda: good, jobs=[], discovery_code="c", state=D.PlanState())
    (tmp_path / H3.LADDER_STATE_FILE).unlink()
    with pytest.raises(SystemExit, match="ladder .M3-M7. is not complete"):
        MF.refuse_unless_ready(tmp_path, check=lambda: 0, digests=lambda: good, jobs=[], discovery_code="c", state=D.PlanState(),
                               require_scorer=False)
    # the cheap gate runs before the feasibility frame / corpus in every main (finding VL2-04)
    with pytest.raises(SystemExit, match="nothing heavy was loaded"):
        H3.refuse_unless_cheap_complete(tmp_path / "nowhere")
    assert H3.refuse_unless_cheap_complete(tmp_path)["complete"]


def test_build_and_write_verifies_every_number(full_root: Path) -> None:
    BR = _load("g19_build_report")
    res = BR.build_and_write(full_root, extra={"git_head": "x"})
    assert res["verification"]["all_ok"] and res["verification"]["n_numbers"] == len(res["numbers"])
    assert (full_root / "tables" / "report_numbers_verification.csv").exists()
    assert (full_root / "evaluation" / "report" / "report_inputs.json").exists()


# --------------------------------------------------------------------------------------------- task X findings V-01, VL2-01, VL2-02

def test_classify_support_never_promotes_an_unknown_or_outside_table_status() -> None:
    from gen19ct.evaluation import support as SUP
    assert R.classify_support("OUTSIDE_TABLE") == "unsupported"
    assert R.classify_support("IN_DOMAIN|OUTSIDE_TABLE") == "unsupported"
    assert R.classify_support("IN_DOMAIN|INTERPOLATION") == "directly supported"
    assert R.classify_support("IN_DOMAIN|CROSS_METAL_TRANSFER") == "transfer-supported"
    assert R.classify_support("CONDITION_EXTRAPOLATION|IN_DOMAIN") == "speculative"
    assert R.classify_support("") == "unclassified" and R.classify_support("nan") == "unclassified"
    for status in SUP.DOMAIN_STATUS_ORDER:
        assert R.classify_support(status) in R.SUPPORT_CATEGORIES[:4]
    for bad in ("PHASE_BEHAVIOUR_UNKNOWN", "OUTSIDE_TABLE|IN_DOMAN", "MODEL_PREDICTED"):
        with pytest.raises(ValueError, match="unknown domain status"):
            R.classify_support(bad)
    assert "unclassified" in R.SUPPORT_CATEGORIES
    win = pd.DataFrame({"cell": ["a", "b"], "candidate": [1, 2], "label": ["x", "x"], "support_rank": [2, 0],
                        "statuses_used": ["IN_DOMAIN", "OUTSIDE_TABLE"]})
    assert list(R.support_status_table(win)["support_category"]) == ["directly supported", "unsupported"]


def test_deployed_predictor_reads_the_ladder_runner_state(full_root: Path) -> None:
    """Task X finding V-01: the report's deployed arm follows the ladder runner's ladder.json (as the H3 runner does), so
    Q5's kept steps and the deployed-arm sections agree; without ladder.json the rule is pending (not computed)."""
    from gen19ct.evaluation import h3 as H3
    dec = json.loads((full_root / "evaluation" / "discovery" / "decisions" / "decisions.json").read_text(encoding="utf-8"))
    lad = json.loads((full_root / H3.LADDER_STATE_FILE).read_text(encoding="utf-8"))
    # this fixture's ladder retains NO step (no M0 row), so addendum 3 item 2 reaches addendum 2's order and, with both
    # stop-rule contrasts FAILing, the section 10 F6 best-passing baseline
    assert R.deployed_predictor(dec, lad)["arm"] == "B3i"
    # POST-HOC addendum 3 item 2: a ladder that RETAINS its base step M0 (= B5) deploys M0, as the real run's does, and
    # the report names M0 -- never M2 by the stop-rule fallback, which would contradict the ladder decision
    with_m0 = {**dec, "ladder": {"M0": {"step": "M0", "kept": True, "note": "base (B5)"}, **dec["ladder"]},
               "stop_rule": {**dec["stop_rule"], "M2_vs_B3i": {"verdict": "PASS"}}}
    dep0 = R.deployed_predictor(with_m0, lad)
    assert dep0["arm"] == "M0" and dep0["status"] == "decided" and "addendum 3 item 2" in dep0["basis"]
    kept = dict(lad, stop_rule=False)
    kept["steps"] = {**{s: {"step": s, "status": "kept", "kept": True} for s in ("M3", "M4")},
                     **{s: {"step": s, "status": "removed", "kept": False} for s in ("M5", "M6")},
                     "M7": {"step": "M7", "status": "judged", "kept": False}}
    assert R.deployed_predictor(dec, kept)["arm"] == "M4"
    pend = R.deployed_predictor(dec, None)
    assert pend["arm"] is None and pend["status"] == "pending"
    assert R.deployed_predictor(None, lad)["status"] == "not computed"
    # the report built without ladder.json prints the deployed sections as not computed and stays traceable
    (full_root / H3.LADDER_STATE_FILE).unlink()
    res = R.build_report(full_root, extra={"git_head": "x"})
    assert res["context"]["deployed"]["status"] == "pending"
    assert "not complete" in res["context"]["deployed"]["basis"]
    R.write_outputs(full_root, res)                       # tables/process_support_status.csv is written here
    assert_traceable(full_root, res["report"], res["numbers"])
    ver = R.verify_numbers(full_root, res["numbers"])
    assert ver["ok"].all(), ver[~ver["ok"]].to_dict("records")[:5]


def test_reproducibility_numbers_go_through_the_ledger(full_root: Path) -> None:
    """Task X finding VL2-01: the manifest file count, every manifest's output count, the addenda count and the digests
    of the reproducibility section are ledger entries that re-resolve from their cited files."""
    man = full_root / "manifests"
    write_json(man / "g19_x.json", {"script": "g19_x", "git_head": "abc", "outputs": ["a", "b", "c"]})
    write_json(man / "g19_y.json", {"script": "g19_y", "git_head": "abc"})
    (man / "run_info").mkdir(exist_ok=True)
    write_json(man / "run_info" / "g19_x.json", {"volatile": True})
    (man / "prereg_sha256.txt").write_text("f" * 64 + "\n", encoding="utf-8")
    extra = {"git_head": "x", "addenda_sha256": "05f36fc0354adb75ae41655977115d827ffce1227b2bdbfeeba28a975f0e8e54", "n_addenda": 1,
             "recomputed": "f" * 64}
    (full_root / "gen19ct" / "evaluation").mkdir(parents=True, exist_ok=True)
    (full_root / "gen19ct" / "evaluation" / "discovery.py").write_text(
        'REGISTERED_ADDENDA_SHA256 = "05f36fc0354adb75ae41655977115d827ffce1227b2bdbfeeba28a975f0e8e54"\n', encoding="utf-8")
    res = R.build_report(full_root, extra=extra)
    R.write_outputs(full_root, res)                       # tables/process_support_status.csv is written here
    rep = res["report"]
    nums = res["numbers"]
    n_man = len([q for q in man.glob("*.json") if q.is_file()])          # the fixture's own manifest(s) + the two above
    assert n_man >= 3 and (man / "run_info" / "g19_x.json").exists()
    assert f"Manifests (`manifests/<script>.json`, deterministic; `manifests/run_info/` volatile): {n_man} files" in rep
    assert "`manifests/g19_x.json`: script g19_x, git HEAD `abc`, outputs 3" in rep
    assert "`manifests/g19_y.json`: script g19_y, git HEAD `abc`, outputs: not recorded" in rep
    man_rows = nums[nums["source_path"] == "manifests"]
    assert len(man_rows) == 1 and man_rows.iloc[0]["source_key"] == "count:glob:*.json" and man_rows.iloc[0]["printed"] == str(n_man)
    out_rows = nums[nums["source_path"] == "manifests/g19_x.json"]
    assert len(out_rows) == 1 and out_rows.iloc[0]["source_key"] == "count:len:outputs" and out_rows.iloc[0]["printed"] == "3"
    assert (nums["source_path"] == "gen19ct/evaluation/discovery.py").sum() == 1            # the registered addenda digest
    assert (nums["printed"] == "f" * 64).sum() >= 1                                        # the recomputed footer digest
    assert R.resolve_source(full_root, "manifests", "count:glob:*.json") == n_man            # run_info/ is not counted
    assert R.resolve_source(full_root, "manifests/g19_x.json", "count:len:outputs") == 3
    ver = R.verify_numbers(full_root, nums)
    assert ver["ok"].all(), ver[~ver["ok"]].to_dict("records")[:5]
    assert_traceable(full_root, rep, nums)
    # a digest that no file holds is stated as differing, never printed as a bare token
    res2 = R.build_report(full_root, extra={**extra, "recomputed": "e" * 64, "addenda_sha256": "d" * 64})
    assert "DIFFERS from `REGISTERED_ADDENDA_SHA256`" in res2["report"] and "DIFFERS from `manifests/prereg_sha256.txt`" in res2["report"]
    assert "e" * 64 not in res2["report"] and "d" * 64 not in res2["report"]


def _deploy_m0(root: Path) -> None:
    """Make the synthetic decisions.json the real run's state: the ladder retains its base step M0 (= B5) and drops
    M1 / M2, so ``h3.deployed_rule`` names M0 (POST-HOC addendum 3 item 2)."""
    p = root / "evaluation" / "discovery" / "decisions" / "decisions.json"
    dec = json.loads(p.read_text(encoding="utf-8"))
    dec["ladder"] = {"M0": {"step": "M0", "kept": True, "note": "base (B5)"},
                     "M1": {"step": "M1", "kept": False, "predecessor": "M0"},
                     "M2": {"step": "M2", "kept": False, "predecessor": "M0"}}
    write_json(p, dec)


def test_a_failed_contrast_prints_as_a_null_only_where_its_own_power_check_says_so(full_root: Path) -> None:
    """Task X finding V-P01.  The "null / supported / ambiguous" section mapped every ``reported_verdict == "FAIL"`` to
    "null" with no reference to the section 8 power check, so H1b printed as a null although its power check returned
    UNDECIDED_UNDERPOWERED, and H4 contrasts with NO registered power check printed as nulls too.  Section 8: "A null
    with kappa_min > 0.25 log D is reported as UNDECIDED (underpowered)" and "Otherwise the verdict is UNDECIDED, never
    'no effect'"; addendum 3 item 3 requires the power check to run "before any null is reported"."""
    ev = full_root / "evaluation"
    # the three FAIL contrasts of this fixture are H1 (M2 vs B3i), H1b (B6 vs B3i) and H4 "factorised vs descriptor"
    # (M2 vs M0); the last gets no power record at all
    write_json(ev / "power" / "power_checks.json", {"checks": [
        {"contrast": "M2 vs B3i@V5", "kappa_min": 0.1, "verdict": "INFORMATIVE_NULL"},
        {"contrast": "B6 vs B3i@V5", "kappa_min": 0.5, "verdict": "UNDECIDED_UNDERPOWERED"}]})
    d02 = R.build_report(full_root, extra={"git_head": "abc"})["d02"]
    block = d02.split("## null / supported / ambiguous")[1].split("## decision")[0]
    lines = {ln.split(" (")[0].removeprefix("- "): ln for ln in block.strip().split("\n") if ln.startswith("- ")}
    # a FAIL whose power record is underpowered: UNDECIDED, with kappa_min, never "null"
    assert "**UNDECIDED (underpowered)**" in lines["H1b"] and "0.50" in lines["H1b"]
    assert ": null" not in lines["H1b"]
    # a FAIL with NO registered power check at all: UNDECIDED, and it says which
    assert "**UNDECIDED (no registered power check)**" in lines["H4 factorised vs descriptor"]
    assert "evaluation/power/power_checks.json" in lines["H4 factorised vs descriptor"]
    assert ": null" not in lines["H4 factorised vs descriptor"]
    # a FAIL whose power record screened it as an informative null IS a null, and cites the record it rests on
    assert "null (informative: kappa_min 0.10" in lines["H1"] and "checks[0].verdict" in lines["H1"]
    # a PASS is unchanged, and a contrast with no row at all is still ambiguous / not run, never a null
    assert lines["H4 factorised vs flat"].endswith("supported")
    assert "ambiguous / not run" in lines["H4 + mechanism"]
    # ... and with no power file at all nothing reads as a null
    (ev / "power" / "power_checks.json").unlink()
    d02b = R.build_report(full_root, extra={"git_head": "abc"})["d02"]
    block2 = d02b.split("## null / supported / ambiguous")[1].split("## decision")[0]
    assert block2.count("UNDECIDED (no registered power check)") == 3        # every FAIL, none of them a null
    assert ": null" not in block2


def test_the_deployed_predictors_tables_and_verdicts_are_looked_up_by_the_record_name(full_root: Path) -> None:
    """Task X finding V-P02: ``tables/discovery_summary.csv``, ``h3_verdicts.json`` and the ``_support`` files write the
    deployed configuration's rows under the arm its RECORDS use (M0 -> B5, ``discovery.ARM_ALIASES``), so the report
    looked up "M0", found nothing and printed "not computed" for the deployed predictor's error strata and the
    UNDECIDED branch of Q4 whatever the real verdict was."""
    ev = full_root / "evaluation"
    _deploy_m0(full_root)
    ds = pd.read_csv(full_root / "tables" / "discovery_summary.csv")
    strata = pd.DataFrame([_summary_row("B5", "V5", "mae", 0.31 + 0.01 * i, stratum=s, n_units=12)
                           for i, s in enumerate(("light_Ln", "heavy_Ln", "mid_Ln"))])
    write_csv(pd.concat([ds, strata], ignore_index=True), full_root / "tables" / "discovery_summary.csv")
    write_json(ev / "h3" / "h3_summary.json", {"deployed": {"arm": "M0", "arm_alias": "B5"}, "transforms": ["WITHOUT"]})
    write_json(ev / "h3" / "h3_verdicts.json", {"B5": {"verdict": "helps", "v5_with_beats_without": True,
                                                      "v5_with_beats_permuted": True, "v1_v2_non_inferior": {},
                                                      "tost_v5_with_minus_without": {"verdict": "PASS"}}})
    rep = R.build_report(full_root, extra={"git_head": "abc"})["report"]
    q2 = rep.split("## Q2.")[1].split("## Q3.")[0]
    assert "Strata of the deployed predictor (M0)'s V5 error" in q2
    for s in ("light_Ln", "heavy_Ln", "mid_Ln"):
        assert f"- {s}: " in q2
    assert "arm=M0 strata rows" not in q2 and "arm=B5 strata rows" not in q2
    q4 = rep.split("## Q4.")[1].split("## Q5.")[0]
    assert "use actinide rows: WITH beats WITHOUT and ACT_PERMUTED" in q4       # the *helps* branch, not UNDECIDED
    assert "| B5 (deployed = M0) |" in q4
