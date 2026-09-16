"""``scripts/g19_run_preseal.py`` -- the pre-seal Phase C difficulty run (pre-registration section 0, order of work
item 1; sections 3, 4, 5, 9, 12, 13; brief section 28 Phase C).

What runs
---------
Only the closed-form arms B0, B1, B2, B3, B3x, B3i, B3l, B4, B4x, B4l and B7 (``gen19ct.models.baselines`` /
``mass_action``), the pair yardsticks FLAT and HEAVIER (``gen19ct.evaluation.pairs``) and split-conformal intervals
on the arms (``gen19ct.models.interface.ConformalWrapper``).  No learned arm is fitted or scored, nothing runs on V6,
and no ``V6_TARGET_ROWS`` row is scored: every scoring index passes ``registered.assert_not_scored`` before a metric
is computed, and the script re-checks every written prediction at the end.

Designs (fold files of ``scripts/g19_build_folds.py``; the registered outer folds are only read, never rebuilt)
    V5 primary exact leave-one-cell-out (intervals: section 7 inner cells, seed 104729) and the registered V5
    sensitivities: loose, strict, HNO3-only, cell-only, parent-structure (refits), Sr(III)-dropped training (refit on
    the primary folds), and the scoring filters non-DGA / metal-class / acid-medium strata, censoring-candidate
    excluded and acid-grid excluded; V5-P (cell x publication group); V5-PAIR (cell pairs; logSF and direction);
    V1 exact (104 folds; intervals: grouped 3-fold inner) with its near-duplicate-key and compilation-DOI groupings and
    Sr(III)-dropped training; V2 element-level (23 states; intervals: 3 inner metals) with state-level hiding and
    Sr(III)-dropped training; V0 (diagnostic; 5 x 5 folds; intervals: grouped 3-fold inner).  V3, V4, V7: not run
    pre-seal.

Every outer split passes ``fold_isolation_check`` at its registered level before any fit (the fold builders' guard
functions ``cell_holdout.guard_v5`` / ``guard_v5p``, ``source_holdout.guard_v1``, ``metal_holdout.guard_v2``); V0 has
no guard level (section 2).  The conformal inner designs take their inner units from the fold builders' own functions
(one implementation per design, ``tests/test_inner_designs.py``), and their calibration rows are the rows a fold
scores (no X(?) row).  The section 13 functions are ``gen19ct.evaluation.support``; each fold's thresholds are
recomputed and checked against the ``support_tau`` the fold builder wrote beside the fold hash.  Inner calibration splits pass ``context.isolation_check`` at the design level:
``every_split`` for V1 / V2 / V0, and for V5 the wrapper's ``nested_certificate`` mode (one check per inner cell on a
superset split, nesting asserted), cross-checked against ``every_split`` on ``--verify-inner-guard-folds`` outer folds.

Outputs (``generations/gen19_chem_transfer/``)
    evaluation/preseal/predictions/<job>.parquet   per (fold, scored row, arm): mean, std, intervals, fallback level
    evaluation/preseal/support/<job>.parquet       per (fold, scored row): section 13 support features on training rows
    evaluation/preseal/units/<job>__units.csv      per (half, arm, unit) metrics
    evaluation/preseal/pairs/V5PAIR__*.csv         per cell pair logSF / direction
    evaluation/preseal/difficulty.json             section 9 difficulty numbers (selection half) + V1 / V2 / V0: the AS-RUN
                                                   record whose SHA-256 section 9 quotes (a full-data rerun must render it
                                                   byte-identical, REGISTERED_DIFFICULTY_SHA256, or nothing is written)
    evaluation/preseal/difficulty_resolved.json    the same numbers under the resolutions of 2026-09-15
    evaluation/preseal/support/<job>__support_score.parquet, support_status.json, domain_status_counts.csv
                                                   the section 13 support_score under the resolved s4 reading
                                                   (scripts/g19_update_support_preseal.apply_support_update)
    evaluation/preseal/guard_log.csv, run_checks.json
    tables/preseal_summary.csv (+ .md), preseal_fallback_counts.csv, preseal_contrasts.csv, preseal_pair_summary.csv,
    preseal_coverage_by_domain_status.csv
    figures/F07_preseal_pred_vs_measured.png, figures/F08_preseal_error_vs_support.png

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_preseal.py

Resolutions of 2026-09-15 (task X, finding VR-01): every table aggregates through ``metrics.design_logd_summary`` /
``design_interval_summary`` / ``design_per_unit_table`` / ``design_unit_clusters`` (V1: the registered outer-fold unit,
the per-publication-group reading beside it as exploratory, column ``unit_reading``), scoring-filter statuses come from
``transfer.sensitivity_status`` (the wildcard-copy filter is registered for V1, V5, V5-P and V5-PAIR), and the support
status from ``support.s4_registered``.  ``READINGS`` stays the as-run record inside ``difficulty.json``;
``RESOLVED_READINGS`` is written everywhere else.  ``code_sha256`` also records ``scripts/g19_update_support_preseal.py``,
whose ``apply_support_update`` this run calls.

``--skip-compute`` re-aggregates the stored prediction and support files and rewrites every output above except the
predictions, support feature parquets, guard log and inner-guard record; it first requires those to match
``manifests/g19_run_preseal.json`` and records, under ``aggregation_rebuild``, the carried compute-run record and every
output a later manifest (``g19_update_support_preseal``) had superseded.  ``--tables-only`` (implies ``--skip-compute``) fits
nothing and rewrites only ``tables/``: it first checks that every non-table output and every stored prediction / support
file still has the digest the manifest being replaced recorded; it then renders every output into a temporary staging
directory, requires each non-table output to be byte-identical to the file on disk (none is rewritten), and moves into
place only the tables whose bytes changed.  The manifest keeps the compute run's git HEAD, arguments and code digests
under ``tables_only_rebuild.compute_run`` and lists which tables changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.data import leakage as L  # noqa: E402
from gen19ct.data import load  # noqa: E402
from gen19ct.data import normalize as N  # noqa: E402
from gen19ct.evaluation import calibration as EC  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.evaluation import pairs as EP  # noqa: E402
from gen19ct.evaluation import support as ES  # noqa: E402
from gen19ct.evaluation import transfer as ET  # noqa: E402
from gen19ct.folds import cell_holdout as CH  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.folds import metal_holdout as MH  # noqa: E402
from gen19ct.folds import registered as FR  # noqa: E402
from gen19ct.folds import source_holdout as SH  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json, write_text  # noqa: E402
from gen19ct.models import baselines as B  # noqa: E402
from gen19ct.models import interface as I  # noqa: E402
from gen19ct.models import mass_action as MA  # noqa: E402

NAME = "g19_run_preseal"
ROW_ID = FI.ROW_ID
METAL, ELEMENT, SYSTEM = SG.METAL_COL, SG.ELEMENT_COL, SG.SYSTEM_COL
ARMS: tuple[str, ...] = tuple(B.ARM_NAMES) + ("B7",)
PAIR_YARDSTICKS: tuple[str, ...] = ("FLAT", "HEAVIER")
#: the exploratory HEAVIER reading printed beside the registered one (column ``yardstick_reading``)
HEAVIER_HALF_READING = "HEAVIER_undefined_counts_half"
#: (yardstick_reading, pair_subset) -> the arm label of an exploratory HEAVIER reading, distinct from "HEAVIER"
PAIR_ALT_ARM: dict[tuple[str, str], str] = {(HEAVIER_HALF_READING, "all_pairs"): "HEAVIER_alt_half",
                                            (HEAVIER_HALF_READING, "Ln-Ln_only"): "HEAVIER_alt_half",
                                            ("HEAVIER", "Ln-Ln_only"): "HEAVIER_alt_lnln"}
#: arm / yardstick labels that may appear in an output of this run
ALLOWED_LABELS = frozenset(ARMS) | frozenset(PAIR_YARDSTICKS) | frozenset(PAIR_ALT_ARM.values()) | {"lookup_oracle"}
#: learned arms that must never be fitted or scored before sealing (checked on every data output)
FORBIDDEN_TOKEN = re.compile(r"\b(B5|B6|B6r0|B8|FLAT_CAT|M[0-7])\b")
CONFORMAL_SEED = FI.DISCOVERY_SEEDS[0]
#: smoke tests (--max-folds) tolerate a half without scored units in the difficulty block
LENIENT = {"on": False}
HALF_LABEL = {"S": "selection", "C": "confirmation", "NA": "none"}
DESIGN_LABEL = {"V0": "V0", "V1": "V1", "V2": "V2", "V5": "V5", "V5P": "V5-P", "V5PAIR": "V5-PAIR"}
FOCUS7 = ("La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)")
LEVELS = I.LEVELS
DPI = 130
#: reference categorical palette slots 1-3 (the Phase A/B figures' CATEGORY_COLORS order)
CLASS_COLORS = {"lanthanide": "#2a78d6", "actinide": "#eb6834", "other": "#1baf7a"}
ORGANIC_ACIDS = ("malonic_acid", "lactic_acid", "tartaric_acid", "citric_acid")

PRED_KEEP: tuple[str, ...] = (
    "mean_logD", "std_logD", "lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95",
    "fallback_level", "fallback_reason", "anion_dropped", "n_candidates", "nn_canonical_measurement_id", "nn_distance",
    "lookup_metal", "nearest_radius_metal", "nearest_radius_distance_A", "nearest_radius_basis", "nearest_radius_pool",
    "bracket_lower_metal", "bracket_upper_metal", "interp_fraction", "nearest_ligand_system", "nearest_ligand_tanimoto",
    "b7_intercept_metal", "b7_n", "b7_p_eff", "b7_n_status", "b7_p_eff_status", "conformal_n_calibration",
    "conformal_q50", "conformal_q80", "conformal_q95")
#: columns kept for every job; the lookup diagnostics of PRED_KEEP are kept only for the registered primary regimes
PRED_CORE: tuple[str, ...] = ("mean_logD", "std_logD", "lower_50", "upper_50", "lower_80", "upper_80", "lower_95",
                              "upper_95", "fallback_level", "fallback_reason", "lookup_metal")
DIAGNOSTIC_JOBS = frozenset({"V5__primary", "V1__copy", "V2__element"})
SUPPORT_FEATURES: tuple[str, ...] = (
    "exact_pair_rows", "exact_pair_publications", "alias_unknown_state_rows", "n_neighbour_metals",
    "n_neighbour_metals_same_category", "n_neighbour_metals_same_charge", "system_family", "n_same_family_rows_for_metal",
    "n_same_family_other_system_rows_for_metal", "nearest_radius_metal", "nearest_radius_distance_A",
    "nearest_radius_same_charge", "nearest_radius_same_species_charge", "nearest_radius_basis",
    "n_radius_neighbours_within_tol", "radius_bracket_lower_metal", "radius_bracket_upper_metal",
    "radius_bracketed_same_charge", "nearest_ligand_system", "nearest_ligand_tanimoto", "n_systems_for_metal",
    "n_publications_pair", "n_publications_system", "n_publications_metal", "condition_distance_system",
    "condition_distance_system_same_acid", "condition_distance_pair", "condition_dims_used", "n_series_neighbours_pm1",
    "n_series_neighbours_pm2", "series_bracketed")

#: readings taken where the registration leaves a choice (the conservative one); printed with every output
READINGS: dict[str, str] = {
    "scoring_exclusions": "fold scored rows (known state, not Sr(III), not V6_TARGET_ROWS) minus the acidic "
                          "co-extractant modifier rows (section 2: never scored as rows of a neutral cell), in every "
                          "design",
    "halves": "every table is labelled with its half; difficulty numbers and descriptive contrasts use the selection "
              "half (section 9); confirmation-half values are printed, labelled, and inform nothing",
    "V5_units_clusters": "unit = hidden cell; clusters = extractant_system_key and the publication group holding most "
                         "of the cell's scored rows (ties: smallest label)",
    "V1_units": "unit = the row's group under the fold's grouping (group_cross_publication_copy for the registered "
                "design); half = the registered V1 half of the row's copy-group unit",
    "V2_summaries": "focus-7 lanthanides (primary), all 23 states, Ln(III) 14, actinide 9, each within a half",
    "V0": "diagnostic; unit = publication group (copy group), per discovery seed and the mean over the 5 seeds",
    "acid_strata": "acid_primary HNO3 / HCl / organic_acid (malonic, lactic, tartaric, citric) / other_mineral "
                   "(H2SO4, HClO4)",
    "dga_stratum": "system_family == diglycolamide vs every other family (the feasibility non-DGA definition)",
    "sr_iii_dropped_training": "the Sr(III) MODEL rows are removed from every outer training set (and from the inner "
                               "calibration of nothing: point predictions only)",
    "intervals": "split-conformal intervals (ConformalWrapper) are computed for the registered calibration designs "
                 "of section 12 (V5 primary, V1 copy exact, V2 element) and for V0; sensitivity variants, V5-P and "
                 "V5-PAIR carry point predictions only (their registered use, R19 item 6, is the MAE Delta)",
    "conformal_seed": "inner calibration designs draw from seed 104729 (first discovery seed); V0 from its fold seed; "
                      "the deterministic arms' interval widths therefore depend on this seed",
    "V5_inner_guard": "nested_certificate: one fold_isolation_check (level V5, component-aware, tol 0.005) per inner "
                      "cell on the superset split (MODEL rows minus the cell's registered hiding, test = every row of "
                      "the cell), shared across outer folds; the inner split is asserted nested in it. Every violation "
                      "class of fold_isolation_check needs a training row and a test row, so removing rows from either "
                      "side cannot create one. Cross-checked against every_split on sampled outer folds",
    "V0_inner_design": "section 7 names no V0 inner design; the publication-grouped 3-fold (section 12 "
                       "'publication-grouped calibration folds') with a level-V1 guard",
    "DIR5": "pairs.pair_summary convention: each yardstick's unit-macro direction accuracy over the cell pairs where it "
            "is defined (HEAVIER: Ln(III)-Ln(III) pairs only; undefined pairs excluded and counted); DIR5 = the max "
            "of the three. Alternatives printed beside: HEAVIER undefined counted 1/2 on all pairs, and all three on "
            "Ln-Ln pairs only",
    "V2_comparator": "stronger of B3x / B3i = lower selection-half focus-7 macro MAE (section 3.3 primary summary)",
    "tau_percentiles": "numpy.percentile linear interpolation of the leave-one-row-out condition_distance_pair over "
                       "training rows whose (state, system) pair has >= 2 training rows; query dimensions missing "
                       "in the row are skipped, training values imputed as in SupportIndex",
    "support_score_domain_status": "computed per design only when the registration is unambiguous on every scored "
                                   "row (s4: query-system inclusion and missing descriptors; UNSUPPORTED anion "
                                   "clause read 'system when present, else family' vs 'neither system nor family'); "
                                   "otherwise written as not computed with the ambiguous-row counts",
    "inner_units": "every conformal inner design takes its inner units from the fold builder's functions and random "
                   "streams (folds.source_holdout.inner_group_assignment for V1 and V0, "
                   "folds.cell_holdout.inner_cell_majority + inner_cell_assignment for V5, "
                   "folds.metal_holdout.inner_state_pick for V2)",
    "calibration_population": "conformal calibration rows = the rows a fold would score: known metal state, not "
                              "Sr(III), not V6_TARGET_ROWS, not an acidic co-extractant row; X(?) rows are never "
                              "calibration rows (they are scored in no design)",
    "support_tau": "tau_in / tau_ext / tau_max are recomputed per fold by gen19ct.evaluation.support.tau_thresholds and "
                   "must equal the support_tau written by the fold builder beside the fold hash",
    "B3x_B3i_anion_pool": "B3x / B3i (and B4x): the nearest-radius and radius-bracket pool is the training metal states "
                          "of the query's system measured with the query's acid anion; the anion is dropped (flag "
                          "radius_pool_anion_dropped) only when no other metal state of the system has a training row "
                          "with it (baselines.REGISTRATION_CHOICES['B3x_pool_anion'], fixed in code before any score)",
    "V1_unit_sensitivities": "labelled sensitivities of the V1 averaging unit (the halves were balanced with the pooled "
                             "remainder fold as ONE unit, section 3.2; the registered scoring unit is the publication "
                             "group, section 4): scoring_filter 'remainder_fold_as_one_unit' (the remainder fold's rows "
                             "form one unit) and 'remainder_fold_excluded_scoring' (its rows are not scored); both "
                             "exploratory",
    "wildcard_copies": "leakage sensitivity (verification finding VL-04), exploratory: scoring filters "
                       "'wildcard_copies_excluded_scoring' (a scored row with any leakage.wildcard_copy_pairs partner "
                       "in its fold's training rows, folds/wildcard_copy_crossings.csv) and "
                       "'wildcard_copies_strict_excluded_scoring' (only strict_copy partners), for V5-primary and "
                       "V1-copy",
}
#: the readings above are the AS-RUN record frozen into ``evaluation/preseal/difficulty.json`` (pre-registration section
#: 9 quotes that file's SHA-256).  The orchestrator's resolutions of 2026-09-15 replace the following ones in every other
#: output of this script (tables, units, contrasts, support status, ``difficulty_resolved.json``; task X, finding VR-01).
RESOLVED_READINGS: dict[str, str] = {
    "V1_units": "section 3.2 resolution: the registered V1 scoring unit is the OUTER FOLD (each single-group fold one unit, "
                "the pooled remainder fold one unit; metrics.v1_scoring_units via design_logd_summary / "
                "design_interval_summary / design_per_unit_table, clusters via design_unit_clusters: the remainder fold is "
                "one publication-group cluster); the per-publication-group reading (the as-run unit) is printed beside "
                "it with status exploratory (column unit_reading)",
    "V1_unit_sensitivities": "exploratory: 'remainder_fold_excluded_scoring' (the remainder fold's rows are not scored) "
                             "under the registered unit; the as-run 'remainder_fold_as_one_unit' filter IS the registered "
                             "reading now and is not printed separately (difficulty_resolved.json checks the equality)",
    "wildcard_copies": "section 2 resolution: 'wildcard_copies_excluded_scoring' (any partner) is a registered R19 item 6 "
                       "sensitivity of V1 and V5, and of V5-P and V5-PAIR by the section 8 R19 item 6 resolution of "
                       "2026-09-15 (transfer.sensitivity_status); the strict-partner filter stays exploratory",
    "scoring_filter_status": "status of a scoring-filter row = transfer.sensitivity_status of the design's registry "
                             "(transfer.SCORING_FILTER_OF_SENSITIVITY maps the filter label to its sensitivity); V0 has no "
                             "R19 registry and keeps the as-run rule (censoring and acid-grid filters registered)",
    "support_score_domain_status": "section 13 s4 resolution: support_score is computed on every support job with "
                                   "support.s4_registered (the query's own system counts when it has training rows of the "
                                   "query's metal state; undefined d_desc skipped), through "
                                   "scripts/g19_update_support_preseal.apply_support_update; domain_status as before",
    "conformal_seed": "section 15 resolution registers the mean over the 5 discovery seeds for the deterministic arms' "
                      "intervals in discovery; the pre-seal intervals stay single-seed (104729; V0 its fold seed) and "
                      "descriptive",
    "DIR5": "descriptive difficulty number only: section 9 S1(c) is the paired rule on identical pairs "
            "(transfer.s1c_half / s1c_paired_verdict); 'same fitted folds' resolved 2026-09-15: B3x / B3i re-fitted on "
            "the candidate's batched V5-PAIR folds (transfer.S1C_REGISTERED_FOLD_READING, models.s1c_yardsticks)",
}
#: SHA-256 of ``evaluation/preseal/difficulty.json`` quoted by pre-registration section 9; a full-data rerun must render
#: it byte-identical (the resolved numbers go to ``difficulty_resolved.json``)
REGISTERED_DIFFICULTY_SHA256 = "bbdb72acff5be40b6de0bbd06f512719ea0aa7b05bebc1a9184709df9f28f1d2"
#: the as-run registered scoring filters (section 2 censoring and acid grid); kept only for V0, which has no R19 registry
ASRUN_REGISTERED_FILTERS = frozenset({"none", "censoring_candidates_excluded_scoring", "acid_grid_rows_excluded_scoring"})


@dataclass(frozen=True)
class Job:
    key: str
    design: str
    variant: str
    stem: str
    drop_sr: bool = False
    conformal: str | None = None
    support: bool = False
    threshold_setting: str = ""
    scoring_filters: bool = False


JOBS: tuple[Job, ...] = (
    Job("V5__primary", "V5", "primary", "V5__primary__exact", conformal="V5", support=True,
        threshold_setting="k10_p1_m3", scoring_filters=True),
    Job("V1__copy", "V1", "copy", "V1__copy__exact", conformal="V1", support=True,
        threshold_setting="group_cross_publication_copy_min20", scoring_filters=True),
    Job("V0__rows", "V0", "rows", "V0__rows__random5", conformal="V0", support=True, threshold_setting="random5",
        scoring_filters=True),
    Job("V2__element", "V2", "element", "V2__element__exact", conformal="V2", support=True,
        threshold_setting="rows100_systems5_element_hiding", scoring_filters=True),
    Job("V5PAIR__primary", "V5PAIR", "primary", "V5PAIR__primary__cell_pair", threshold_setting="k10_p1_m3_pairs3"),
    Job("V5P__base", "V5P", "base", "V5P__base__cell_x_group", threshold_setting="k10_p2_m3"),
    Job("V5__loose", "V5", "loose", "V5__loose__exact", threshold_setting="k5_p1_m2"),
    Job("V5__hno3_only", "V5", "hno3_only", "V5__hno3_only__exact", threshold_setting="k10_p1_m3_HNO3"),
    Job("V5__cell_only", "V5", "cell_only", "V5__cell_only__exact", threshold_setting="k10_p1_m3_cell_only"),
    Job("V5__parent_structure", "V5", "parent_structure", "V5__parent_structure__exact",
        threshold_setting="k10_p1_m3_parent_structure"),
    Job("V5__sr_iii_dropped_training", "V5", "sr_iii_dropped_training", "V5__primary__exact", drop_sr=True,
        threshold_setting="k10_p1_m3"),
    Job("V5__strict", "V5", "strict", "V5__strict__exact", threshold_setting="k20_p2_m5"),
    Job("V1__near_duplicate_key", "V1", "near_duplicate_key", "V1__near_duplicate_key__exact",
        threshold_setting="group_near_duplicate_key_min20"),
    Job("V1__compilation_doi", "V1", "compilation_doi", "V1__compilation_doi__exact",
        threshold_setting="group_compilation_doi_min20"),
    Job("V1__sr_iii_dropped_training", "V1", "sr_iii_dropped_training", "V1__copy__exact", drop_sr=True,
        threshold_setting="group_cross_publication_copy_min20"),
    Job("V2__state", "V2", "state", "V2__state__exact", threshold_setting="rows100_systems5_state_hiding"),
    Job("V2__sr_iii_dropped_training", "V2", "sr_iii_dropped_training", "V2__element__exact", drop_sr=True,
        threshold_setting="rows100_systems5_element_hiding"),
)
JOB_BY_KEY = {j.key: j for j in JOBS}
NOT_RUN = {"V3": "not run pre-seal", "V4": "not run pre-seal", "V6": "never run pre-seal (section 3.4)",
           "V7": "not run pre-seal", "V5_batched": "not run (exact leave-one-cell-out is registered for B0-B4, B7)",
           "V1_grouped10": "not run (the exact design is registered for B0-B4, B7)"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", default=",".join(j.key for j in JOBS), help="comma list of job keys")
    ap.add_argument("--workers", type=int, default=2, help="worker processes (<= 2 on this machine)")
    ap.add_argument("--max-folds", type=int, default=0, help="smoke test: scored folds per job (0 = all)")
    ap.add_argument("--out-root", default=str(paths.G19_ROOT), help="root of evaluation/, tables/, figures/")
    ap.add_argument("--verify-inner-guard-folds", type=int, default=6,
                    help="V5 primary outer folds on which nested_certificate is cross-checked against every_split")
    ap.add_argument("--skip-compute", action="store_true", help="aggregate existing prediction files only")
    ap.add_argument("--tables-only", action="store_true",
                    help="rebuild tables/ only, from the stored evaluation/preseal outputs (implies --skip-compute); "
                         "every other output must re-render byte-identical and is not rewritten")
    ap.add_argument("--no-manifest", action="store_true", help="smoke test: write no manifest")
    ns = ap.parse_args(argv)
    if not 1 <= ns.workers <= 2:
        raise SystemExit("--workers must be 1 or 2 (machine limit)")
    if ns.tables_only:
        if ns.max_folds:
            raise SystemExit("--tables-only rebuilds the full run's tables; --max-folds is not allowed with it")
        ns.skip_compute = True
    return ns


def out_dirs(root: Path) -> SimpleNamespace:
    ev = root / "evaluation" / "preseal"
    return SimpleNamespace(root=root, ev=ev, pred=ev / "predictions", support=ev / "support", units=ev / "units",
                           pairs=ev / "pairs", tables=root / "tables", figures=root / "figures")


def rel(p: Path) -> str:
    try:
        return paths.rel(Path(p))
    except ValueError:
        return Path(p).as_posix()


class OutputSink:
    """Where this invocation writes an output whose final location is ``p``.

    A full or ``--skip-compute`` run writes in place (``stage`` None).  Under ``--tables-only`` every output is written
    below ``stage`` instead; ``current(p)`` is the file holding the output's new content, and ``commit`` requires every
    staged non-table output to be byte-identical to ``p`` (it is never rewritten) and copies into place only the
    ``tables/`` files whose bytes changed."""

    def __init__(self, dirs: SimpleNamespace, stage: Path | None):
        self.dirs, self.stage = dirs, stage
        self.staged: dict[Path, Path] = {}

    def __call__(self, p: Path) -> Path:
        p = Path(p)
        if self.stage is None:
            return p
        q = self.stage / p.resolve().relative_to(Path(self.dirs.root).resolve())
        paths.ensure_dir(q.parent)
        self.staged[p] = q
        return q

    def current(self, p: Path) -> Path:
        return self.staged.get(Path(p), Path(p))

    def is_table(self, p: Path) -> bool:
        return Path(p).resolve().parent == Path(self.dirs.tables).resolve()

    def commit(self) -> dict:
        if self.stage is None:
            return {}
        same = {p: p.exists() and p.read_bytes() == q.read_bytes() for p, q in self.staged.items()}
        differ = sorted(rel(p) for p in self.staged if not self.is_table(p) and not same[p])
        if differ:
            raise AssertionError("--tables-only: these non-table outputs would change, so the tables would disagree "
                                 f"with them (run --skip-compute instead); nothing was written: {differ}")
        changed = sorted((p for p in self.staged if self.is_table(p) and not same[p]), key=rel)
        for p in changed:
            shutil.copyfile(self.staged[p], p)
        return {"tables_rewritten": [rel(p) for p in changed],
                "tables_byte_identical_not_rewritten": sorted(rel(p) for p in self.staged if self.is_table(p) and same[p]),
                "other_outputs_rerendered_byte_identical_not_rewritten": sorted(
                    rel(p) for p in self.staged if not self.is_table(p))}


def compute_run_record(manifest_path: Path, dirs: SimpleNamespace) -> dict:
    """The compute run a ``--tables-only`` rebuild rests on, carried from the manifest it replaces (from that manifest's
    own ``tables_only_rebuild.compute_run`` when it is itself a rebuild).  Every recorded output outside ``tables/``, the
    stored predictions and support files included, must still match its recorded digest."""
    if not manifest_path.exists():
        raise SystemExit(f"--tables-only needs the compute run's manifest {rel(manifest_path)}")
    prev = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = Path(dirs.tables).resolve()
    stale, n_checked = [], 0
    for rec in prev.get("outputs", []):
        p = paths.REPO_ROOT / rec["path"]
        if p.resolve().parent == tables:
            continue
        n_checked += 1
        if rec.get("missing") or not p.exists() or not paths.matches(p, rec["sha256"]):
            stale.append(rec["path"])
    stored = [p for d in (dirs.pred, dirs.support) for p in sorted(Path(d).glob("*.parquet"))]
    unrecorded = sorted(set(rel(p) for p in stored) - {r["path"] for r in prev.get("outputs", [])})
    if stale or unrecorded:
        raise SystemExit(f"--tables-only: outputs differ from {rel(manifest_path)} (stale {stale}; unrecorded stored "
                         f"files {unrecorded}); run --skip-compute or the full run instead")
    return {"compute_run": carried_compute_run(prev, tables),
            "n_non_table_outputs_checked_against_replaced_manifest": n_checked,
            "_carried_aggregation_rebuild": prev.get("aggregation_rebuild")}


def carried_compute_run(prev: dict, tables: Path) -> dict:
    """The record of the run that fitted the stored predictions: carried from ``tables_only_rebuild.compute_run`` or
    ``aggregation_rebuild.compute_run`` when the replaced manifest is itself a rebuild, else its own top-level fields."""
    for key in ("tables_only_rebuild", "aggregation_rebuild"):
        carried = (prev.get(key) or {}).get("compute_run")
        if carried is not None:
            return carried
    if (prev.get("arguments") or {}).get("skip_compute"):
        raise SystemExit("the replaced manifest is an aggregation run that carries no compute-run record")
    return {"git_head": prev.get("git_head"), "arguments": prev.get("arguments"), "code_sha256": prev.get("code_sha256"),
            "tables_sha256": {r["path"]: r.get("sha256") for r in prev.get("outputs", [])
                              if (paths.REPO_ROOT / r["path"]).resolve().parent == tables}}


#: manifests of later steps that rewrite outputs of this script (checked when an aggregation run replaces a manifest)
LATER_MANIFESTS: tuple[str, ...] = ("g19_update_support_preseal",)


def compute_outputs(dirs: SimpleNamespace, jobs) -> list[Path]:
    """The compute phase's outputs an aggregation run reads and never rewrites: predictions, support feature files
    (not the ``__support_score`` files), the guard log and the inner-guard record."""
    out = [dirs.ev / "guard_log.csv", dirs.ev / "inner_guard_verification.json"]
    out += [dirs.pred / f"{j.key}.parquet" for j in jobs]
    out += [dirs.support / f"{j.key}.parquet" for j in jobs if j.support]
    return out


def aggregation_run_record(manifest_path: Path, dirs: SimpleNamespace, jobs) -> dict:
    """What a ``--skip-compute`` run rests on (task X, finding VR-01).  Every stored compute output must match the
    replaced manifest (nothing is refitted); the compute-run record is carried forward (:func:`carried_compute_run`).
    Outputs that no longer match the replaced manifest, and stored files it does not record, are listed with the later
    manifest (:data:`LATER_MANIFESTS`) that records their current digest -- the supersession chain this run closes by
    recording every output again."""
    if not manifest_path.exists():
        raise SystemExit(f"--skip-compute needs the compute run's manifest {rel(manifest_path)}")
    prev = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec = {r["path"]: r["sha256"] for r in prev.get("outputs", [])}
    stale = [rel(p) for p in compute_outputs(dirs, jobs) if rel(p) not in rec or not paths.matches(p, rec[rel(p)])]
    if stale:
        raise SystemExit(f"--skip-compute: stored compute outputs differ from {rel(manifest_path)}: {stale}")
    later = {}
    for name in LATER_MANIFESTS:
        m = paths.MANIFESTS_DIR / f"{name}.json"
        if m.exists():
            later[rel(m)] = {r["path"]: r["sha256"] for r in json.loads(m.read_text(encoding="utf-8")).get("outputs", [])}

    def recorded_by(path: str) -> str | None:
        p = paths.REPO_ROOT / path
        return next((m for m, outs in later.items() if path in outs and p.exists() and paths.matches(p, outs[path])), None)
    superseded = {path: {"sha256_in_replaced_manifest": sha,
                         "sha256_on_disk_before_this_run": paths.digests(paths.REPO_ROOT / path)["sha256"]
                         if (paths.REPO_ROOT / path).exists() else None,
                         "current_digest_recorded_by": recorded_by(path)}
                  for path, sha in sorted(rec.items())
                  if not ((paths.REPO_ROOT / path).exists() and paths.matches(paths.REPO_ROOT / path, sha))}
    stored = [p for d in (dirs.pred, dirs.support) for p in sorted(Path(d).glob("*.parquet"))]
    unrecorded = {rel(p): {"sha256_on_disk_before_this_run": paths.digests(p)["sha256"],
                           "current_digest_recorded_by": recorded_by(rel(p))}
                  for p in stored if rel(p) not in rec}
    return {"compute_run": carried_compute_run(prev, Path(dirs.tables).resolve()),
            "replaced_manifest_sha256": paths.digests(manifest_path)["sha256"],
            "n_compute_outputs_verified_unchanged": len(compute_outputs(dirs, jobs)),
            "outputs_superseded_before_this_run": superseded, "stored_files_unrecorded_by_replaced_manifest": unrecorded,
            "later_manifests_checked": sorted(later)}


# ============================================================================================= #
# worker side: fits and predictions
# ============================================================================================= #

W = SimpleNamespace()


def init_worker(coext_ids: list[str], max_folds: int, dirs_root: str, verify_folds: int) -> None:
    model = load.load_model_rows()
    fr = I.prepare_frame(model)
    groups = FI.publication_groups(model)
    if not (groups.reindex(fr.index) == fr[I.PUB_GROUP_COL]).all():
        raise AssertionError("prepare_frame pub_group differs from folds.io.publication_groups")
    fr[FI.GROUP_COL] = fr[I.PUB_GROUP_COL]
    W.fr = fr
    W.slim = FI.slim_frame(fr)
    W.sup = fr[list(SG.REQUIRED_COLUMNS)]
    W.systems, W.comps = I.load_descriptor_tables()
    W.table = I.RowTable(fr, systems=W.systems, components=W.comps)
    W.v6 = FR.v6_target_mask(model, FR.v6_system_set(model)).reindex(fr.index).fillna(False).astype(bool)
    if int(W.v6.sum()) != 580 or set(fr.loc[W.v6.to_numpy(), ROW_ID].astype(str)) != FI.registered_v6_ids():
        raise AssertionError("V6_TARGET_ROWS differ from the registered set")
    W.idmap = pd.Series(fr.index, index=fr[ROW_ID].astype(str))
    W.coext = fr[ROW_ID].astype(str).isin(set(coext_ids))
    W.sr_index = fr.index[(fr[METAL] == I.SR_III).to_numpy()]
    W.pmap = FR.parent_component_map()
    t = W.table
    W.row_family = np.array([t.sys_family[s] if s >= 0 else None for s in t.sys], dtype=object)
    W.row_expert = np.array([t.sys_expert[s] if s >= 0 else None for s in t.sys], dtype=object)
    W.temp = pd.to_numeric(fr[SG.TEMP_COL], errors="coerce").to_numpy(dtype=float)
    W.max_folds = int(max_folds)
    W.dirs = out_dirs(Path(dirs_root))
    W.verify_folds = int(verify_folds)
    W.guard_calls = Counter()


def labels_of(ids) -> pd.Index:
    return pd.Index(W.idmap.loc[list(ids)].to_numpy())


def make_arm(name: str):
    return MA.B7MassAction() if name == "B7" else B.BaselineArm(name)


def outer_guard(job: Job, fold: FI.Fold, universe: pd.Index) -> list[dict]:
    """The fold builders' guard at the registered level (raises on a violation)."""
    if job.design in ("V5", "V5PAIR"):
        cmap = W.pmap if fold.meta.get("parent_structure") else None
        return [CH.guard_v5(fold, W.slim, component_map=cmap, universe_index=universe)]
    if job.design == "V5P":
        return CH.guard_v5p(fold, W.slim)
    if job.design == "V1":
        return [SH.guard_v1(fold, W.slim, universe_index=universe)]
    if job.design == "V2":
        return [MH.guard_v2(fold, W.slim, universe_index=universe)]
    if job.design == "V0":
        FI.training_ids(fold, W.idmap.index)
        return []
    raise ValueError(job.design)


def inner_check(kind: str, counter_key: str):
    """``(train_labels, test_labels) -> report``: ``fold_isolation_check`` at the design level of the inner split."""
    if kind == "V5":
        kw = dict(level="V5", component_aware=True)
    elif kind in ("V1", "V0"):
        kw = dict(level="V1")
    elif kind == "V2":
        kw = dict(level="V2", element_level=True)
    else:
        raise ValueError(kind)

    def check(tr: pd.Index, te: pd.Index) -> dict:
        W.guard_calls[counter_key] += 1
        return L.fold_isolation_check(tr, te, W.slim, near_dup_sig=FI.NEAR_DUP_SIG, near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL,
                                      raise_on_violation=False, **kw)
    return check


def splitter_for(kind: str):
    if kind == "V5":
        return I.InnerCellCalibration(k=10, p=1, m=3, n_folds=3, max_cells_per_fold=30, component_aware=True), \
            "nested_certificate"
    if kind in ("V1", "V0"):
        return I.GroupKFoldCalibration(3), "every_split"
    if kind == "V2":
        return I.InnerMetalCalibration(n_metals=3, min_rows=100, min_systems=5), "every_split"
    raise ValueError(kind)


def fold_support(fold: FI.Fold, mask: np.ndarray, ctx: I.FitContext, pos: np.ndarray, seed_label,
                 builder_tau: dict | None) -> pd.DataFrame:
    """Section 13 support features of every scored row, from a SupportIndex fitted on the fold's training rows, plus
    the registered support_score components and domain-status labels (``gen19ct.evaluation.support``) with their
    ambiguity flags.  The fold's thresholds are recomputed and must equal the fold builder's (``builder_tau``)."""
    t = W.table
    train = W.sup.loc[t.index[mask]]
    si = SG.SupportIndex(train, systems=W.systems)
    tau_in, tau_ext, tau_max, n_tau = ES.tau_thresholds(si, train)
    if builder_tau is not None:
        mine = {"tau_in": tau_in, "tau_ext": tau_ext, "tau_max": tau_max, "n_tau_rows": n_tau}
        same = all((np.isnan(mine[k]) and builder_tau[k] is None) or mine[k] == builder_tau[k] for k in mine)
        if not same:
            raise AssertionError(f"{fold.fold_id}: tau {mine} differ from the fold builder's {builder_tau}")
    eng = B.make_engine(t, mask, ctx)
    mu, sd = eng._d_desc_scale(B._BASE)
    fam_counts = Counter(W.row_family[mask])
    exp_counts = Counter(W.row_expert[mask])
    msk_pos = np.flatnonzero(mask & (t.sys >= 0))
    sys_anion = set(zip(t.sys[msk_pos].tolist(), t.anion[msk_pos].tolist()))
    fam_anion = set(zip(W.row_family[msk_pos].tolist(), t.anion[msk_pos].tolist()))
    fam_systems: dict = {}
    for c in np.unique(t.sys[msk_pos]):
        fam_systems.setdefault(t.sys_family[c], []).append(t.sys_labels[c])
    feat_cache: dict = {}
    s4_cache: dict = {}
    recs = []
    for p in pos:
        m, s = t.state_labels[t.state[p]], t.sys_labels[t.sys[p]]
        anion = t.anion_labels[t.anion[p]]
        acid, ext, temp = float(t.acid[p]), float(t.ext[p]), float(W.temp[p])
        key = (m, s, anion, repr(acid), repr(ext), repr(temp))
        if key not in feat_cache:
            cond = {SG.LOG_ACID_COL: acid, SG.LOG_EXT_COL: ext, SG.TEMP_COL: temp,
                    SG.ACID_ANION_COL: None if anion == I.NA_ANION else anion}
            feat_cache[key] = (si.features(m, s, cond), cond)
        f, cond = feat_cache[key]
        st = t.static(s)
        fam, expert = st["family"], st["expert"]
        rec = {"row_id": t.ids[p], "fold_id": fold.fold_id, "seed": seed_label}
        rec.update({k: f[k] for k in SUPPORT_FEATURES})
        system_present = f["n_publications_system"] > 0
        metal_present = f["n_systems_for_metal"] > 0
        family_rows = int(fam_counts.get(fam, 0)) if fam is not None else 0
        mechanism_rows = int(exp_counts.get(expert, 0)) if expert not in (None, I.NO_EXPERT) else 0
        ac = t.anion_code.get(anion)
        a_sys = (t.sys_code.get(s), ac) in sys_anion
        a_fam = (fam, ac) in fam_anion
        fam_cd = float("nan")
        if not system_present and fam in fam_systems:
            ds = [si.condition_distance(k, cond)[0] for k in fam_systems[fam] if k != s]
            ds = [d for d in ds if np.isfinite(d)]
            fam_cd = min(ds) if ds else float("nan")
        # s4 and its two registration gaps
        if (m, s) not in s4_cache:
            mc = t.state_code.get(m)
            codes = sorted({c for c, pp in t.systems_by_state.get(mc, []) if mask[pp].any()}) if mc is not None else []
            cands = [(t.sys_labels[c], t.sys_fp[c], (t.sys_desc[c] - mu) / sd) for c in codes]
            qfp, qz = st["fp"], (st["desc"] - mu) / sd
            v = ES.s4_component(s, qfp, qz, cands, include_query_system=True, missing_descriptor="skip")
            v_excl = ES.s4_component(s, qfp, qz, cands, include_query_system=False, missing_descriptor="skip")
            v_zero = ES.s4_component(s, qfp, qz, cands, include_query_system=True, missing_descriptor="zero")
            s4_cache[(m, s)] = (v, v != v_excl, v != v_zero or (qfp is None and len(cands) > 0))
        s4v, amb_incl, amb_nan = s4_cache[(m, s)]
        comps = ES.support_components(f, s4=s4v, tau_ext=tau_ext, metal_series=SG.metal_properties(m)["series"])
        rec.update({f"support_{k}": v for k, v in comps.items()})
        rec["support_score_candidate"] = ES.support_score(comps)
        rec["support_s4_ambiguous_query_system_inclusion"] = bool(amb_incl)
        rec["support_s4_ambiguous_missing_descriptor"] = bool(amb_nan)
        base = dict(system_present=system_present, metal_present=metal_present, family_rows=family_rows,
                    mechanism_rows=mechanism_rows, expert=expert, f=f, fam_cd=fam_cd, tau=(tau_in, tau_ext, tau_max))
        sa = ES.domain_status(**base, anion_unseen=(not a_sys) if system_present else (not a_fam))
        sb = ES.domain_status(**base, anion_unseen=not (a_sys or a_fam))
        rec.update({"family_rows": family_rows, "mechanism_rows": mechanism_rows, "mechanism_group": expert,
                    "anion_seen_with_system": a_sys, "anion_seen_with_family": a_fam,
                    "family_condition_distance": fam_cd, "tau_in": tau_in, "tau_ext": tau_ext, "tau_max": tau_max,
                    "n_tau_rows": n_tau, "domain_status_candidate": sa[0], "domain_status_reading_neither": sb[0],
                    "domain_status_ambiguous": sa[0] != sb[0], "condition_extrapolated": sa[1],
                    "both_nodes_new": sa[2], "conditions_unknown": sa[3]})
        recs.append(rec)
    return pd.DataFrame(recs)


def drop_sr_tau_mismatch(job: Job) -> bool:
    """A Sr(III)-dropped training set is not the fold builder's training set, so its thresholds would differ."""
    return bool(job.drop_sr)


def run_job(job: Job) -> dict:
    t0 = time.perf_counter()
    folds = FI.read_design(job.stem)
    builder_tau = FI.read_fold_fields(job.stem, "support_tau")
    t = W.table
    universe_all = W.fr.index
    guard_cache: dict = {}
    pred_frames, sup_frames, flog = [], [], []
    n_done = n_verify = 0
    verify_log = []
    for f in folds:
        rec = {"job": job.key, "fold_id": f.fold_id, "half": f.half, "seed": f.seed, "n_hidden": len(f.hidden_row_ids),
               "n_scored_fold": len(f.scored_row_ids)}
        if not f.scored_row_ids:
            flog.append({**rec, "status": "no_scored_rows_not_fitted", "n_scored_used": 0, "n_acidic_coextractant_removed": 0,
                         "n_outer_guard_checks": 0, "outer_guard_ok": None})
            continue
        if W.max_folds and n_done >= W.max_folds:
            break
        hid = labels_of(f.hidden_row_ids)
        sr_extra = W.sr_index.difference(hid) if job.drop_sr else pd.Index([], dtype=hid.dtype)
        universe = universe_all.difference(sr_extra)
        reps = outer_guard(job, f, universe)
        if not all(r["ok"] for r in reps):
            raise AssertionError(f"{job.key}/{f.fold_id}: outer guard failed")
        sc_all = labels_of(f.scored_row_ids)
        keep = ~W.coext.reindex(sc_all).to_numpy(dtype=bool)
        sc = sc_all[keep]
        rec.update(n_scored_used=int(len(sc)), n_acidic_coextractant_removed=int((~keep).sum()),
                   n_outer_guard_checks=len(reps), outer_guard_ok=True)
        if not len(sc):
            flog.append({**rec, "status": "only_excluded_rows"})
            continue
        FR.assert_not_scored(sc, W.v6, what=f"{job.key}/{f.fold_id}")
        st = W.fr.loc[sc, METAL]
        if st.isna().any() or (st == I.SR_III).any():
            raise AssertionError(f"{job.key}/{f.fold_id}: an X(?) or Sr(III) row is scored")
        mask = np.ones(t.n, dtype=bool)
        mask[t.positions(hid)] = False
        if len(sr_extra):
            mask[t.positions(sr_extra)] = False
        seed = f.seed if job.design == "V0" else CONFORMAL_SEED
        check = inner_check(job.conformal, f"{job.key}_inner") if job.conformal else None
        ctx = I.FitContext(systems=W.systems, components=W.comps, table=t, hidden_index=hid.union(sr_extra),
                           v6_mask=W.v6, exclude_from_scoring=W.coext, isolation_check=check,
                           guard_cache=guard_cache, seed=seed)
        pos = t.positions(sc)
        row_unit = [f.row_unit.get(r, f.units[0] if f.units else "") for r in t.ids[pos]]
        row_half = [f.row_half.get(r, f.half) for r in t.ids[pos]]
        for arm in ARMS:
            if job.conformal:
                spl, gmode = splitter_for(job.conformal)
                model = I.ConformalWrapper(make_arm(arm), splitter=spl, guard=gmode).fit_table(t, mask, ctx)
            else:
                model = make_arm(arm).fit_table(t, mask, ctx)
            p = model.predict_positions(pos)
            if not np.isfinite(p["mean_logD"].to_numpy(dtype=float)).all():
                raise AssertionError(f"{job.key}/{f.fold_id}/{arm}: non-finite prediction")
            out = p[list(PRED_KEEP if job.key in DIAGNOSTIC_JOBS else PRED_CORE)].copy()
            out.insert(0, "row_id", t.ids[pos])
            out.insert(1, "fold_id", f.fold_id)
            out.insert(2, "arm", arm)
            out.insert(3, "half", row_half)
            out.insert(4, "unit", row_unit)
            out.insert(5, "seed", -1 if f.seed is None else int(f.seed))
            pred_frames.append(out)
            if job.conformal == "V5" and arm == "B3x" and n_verify < W.verify_folds:
                # cross-check nested_certificate against every_split on the same outer fold
                spl, _ = splitter_for("V5")
                vctx = I.FitContext(systems=W.systems, components=W.comps, table=t, hidden_index=hid, v6_mask=W.v6,
                                    exclude_from_scoring=W.coext, isolation_check=inner_check("V5", "verify_every_split"),
                                    guard_cache={}, seed=seed)
                before = W.guard_calls["verify_every_split"]
                ev = I.ConformalWrapper(make_arm("B3x"), splitter=spl, guard="every_split").fit_table(t, mask, vctx)
                same = bool(np.array_equal(ev.residuals, model.residuals)) and ev.quantiles == model.quantiles
                verify_log.append({"fold_id": f.fold_id, "half": f.half, "n_inner_splits": len(ev.calibration_units),
                                   "every_split_checks": W.guard_calls["verify_every_split"] - before,
                                   "all_passed": True, "residuals_and_quantiles_identical": same})
                if not same:
                    raise AssertionError(f"{f.fold_id}: nested_certificate and every_split calibrations differ")
                n_verify += 1
        if job.support:
            if drop_sr_tau_mismatch(job):
                raise AssertionError(f"{job.key}: support features are computed only on the registered training rows")
            sup_frames.append(fold_support(f, mask, ctx, pos, -1 if f.seed is None else int(f.seed),
                                           builder_tau.get(f.fold_id)))
        flog.append({**rec, "status": "fitted"})
        n_done += 1
    dirs = W.dirs
    pred = pd.concat(pred_frames, ignore_index=True).sort_values(["fold_id", "row_id", "arm"], kind="mergesort")
    if "anion_dropped" in pred.columns:
        pred["anion_dropped"] = pred["anion_dropped"].map(
            lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v))
        pred["n_candidates"] = pd.to_numeric(pred["n_candidates"], errors="coerce").astype(float)
    paths.ensure_dir(dirs.pred)
    pp = dirs.pred / f"{job.key}.parquet"
    pred.reset_index(drop=True).to_parquet(pp, index=False, compression="zstd")
    files = [str(pp)]
    if sup_frames:
        sup = pd.concat(sup_frames, ignore_index=True).sort_values(["fold_id", "row_id"], kind="mergesort")
        paths.ensure_dir(dirs.support)
        sp = dirs.support / f"{job.key}.parquet"
        sup.reset_index(drop=True).to_parquet(sp, index=False, compression="zstd")
        files.append(str(sp))
    return {"job": job.key, "n_folds_in_file": len(folds), "n_folds_fitted": n_done, "fold_log": flog,
            "inner_guard_calls": int(W.guard_calls[f"{job.key}_inner"]), "verify": verify_log, "files": files,
            "runtime_s": round(time.perf_counter() - t0, 1)}


# ============================================================================================= #
# parent side: row attributes
# ============================================================================================= #

def load_support_update_module():
    """``scripts/g19_update_support_preseal.py`` as a module (its ``apply_support_update`` is the section 13 s4 update)."""
    name = "g19_update_support_preseal"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_feasibility_module():
    spec = importlib.util.spec_from_file_location("g19_feasibility", Path(__file__).resolve().parent / "g19_feasibility.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WILDCARD_CROSSINGS_CSV = paths.FOLDS_DIR / "wildcard_copy_crossings.csv"
#: the exact designs whose scored rows carry the row-level wildcard-copy flags (one scoring fold per row)
WILDCARD_STEMS: tuple[str, ...] = ("V5__primary__exact", "V1__copy__exact")


def build_attrs(feas_json: dict) -> pd.DataFrame:
    """Per MODEL row: target, cell, publication group, condition key, strata and the section 2 flags (the
    censoring-candidate and acidic co-extractant flags come from ``g19_feasibility.load_frame``)."""
    mod = load_feasibility_module()
    fr, _ = mod.load_frame(N.DEFAULT_SIG)
    model = load.load_model_rows(copy=False)
    v6 = FR.v6_target_mask(model, FR.v6_system_set(model)).reindex(fr.index).fillna(False).astype(bool)
    acid = fr["acid_primary"].astype(object)
    acid_stratum = np.where(acid == "HNO3", "HNO3", np.where(acid == "HCl", "HCl", np.where(
        acid.isin(ORGANIC_ACIDS), "organic_acid", np.where(acid.isna(), "NA", "other_mineral"))))
    cls = []
    for s in fr[METAL]:
        c = SG.metal_properties(s)["category"] if isinstance(s, str) else None
        cls.append(c if c in ("lanthanide", "actinide") else "other")
    attrs = pd.DataFrame({
        ROW_ID: fr[ROW_ID].astype(str).to_numpy(), "log_D": fr["log_D"].to_numpy(dtype=float),
        METAL: fr[METAL].to_numpy(dtype=object), ELEMENT: fr[ELEMENT].to_numpy(dtype=object),
        SYSTEM: fr[SYSTEM].to_numpy(dtype=object), "system_label": fr["system_label"].to_numpy(dtype=object),
        "system_family": fr["system_family"].to_numpy(dtype=object), "mechanism": fr["mechanism"].to_numpy(dtype=object),
        EM.PUB_GROUP_COL: fr["group_cross_publication_copy"].astype(str).to_numpy(),
        EM.CONDITION_KEY_COL: fr["ck"].to_numpy(dtype=object), "acid_primary": acid.to_numpy(),
        "acid_stratum": acid_stratum, "metal_class": cls,
        "dga_stratum": np.where(fr["system_family"] == "diglycolamide", "diglycolamide", "non_DGA"),
        "censoring_candidate": fr["log_D_censoring_candidate"].to_numpy(dtype=bool),
        "acid_grid_flag": fr["acid_M_log10_grid"].to_numpy(dtype=bool),
        "acidic_coextractant_modifier": fr["acidic_coextractant_modifier"].to_numpy(dtype=bool),
        "v6_target_row": v6.to_numpy(dtype=bool)}).set_index(ROW_ID, drop=False)
    cen = feas_json["log_D_censoring_candidates"]
    if int(attrs["censoring_candidate"].sum()) != cen["n_floor_rows"] + cen["n_ceiling_rows"]:
        raise AssertionError("censoring candidates differ from feasibility.json")
    if int(attrs["acid_grid_flag"].sum()) != feas_json["V7_condition_regions"]["acid_M_log10_grid_flag_rows"]:
        raise AssertionError("acid-grid rows differ from feasibility.json")
    if int(attrs["acidic_coextractant_modifier"].sum()) != 14 or int(attrs["v6_target_row"].sum()) != 580:
        raise AssertionError("acidic co-extractant (14) or V6_TARGET_ROWS (580) counts differ")
    cr = pd.read_csv(WILDCARD_CROSSINGS_CSV, dtype={"row_id": str, "partner_id": str})
    for stem in WILDCARD_STEMS:
        sub = cr[cr["stem"] == stem]
        if sub.groupby("row_id")["fold_id"].nunique().gt(1).any():
            raise AssertionError(f"{stem}: a row is scored in two folds; the row-level copy flag needs one")
        attrs[f"wc_any__{stem}"] = attrs.index.isin(set(sub["row_id"]))
        attrs[f"wc_strict__{stem}"] = attrs.index.isin(set(sub.loc[sub["strict_copy"].astype(str).str.lower() == "true",
                                                                   "row_id"]))
    return attrs


# ============================================================================================= #
# parent side: metrics
# ============================================================================================= #

def unit_cols_of(design: str) -> tuple[str, ...]:
    """The AS-RUN averaging-unit columns (V1: the row's group under the fold's grouping, column ``unit``).  Used only for
    the frozen ``difficulty.json`` numbers (:func:`asrun_v1_summary`); every table uses the registered unit through
    ``metrics.design_*`` (:func:`design_kw`)."""
    return {"V5": EM.CELL_COLS, "V5P": EM.CELL_COLS, "V1": ("unit",), "V2": (METAL,),
            "V0": (EM.PUB_GROUP_COL,)}[design]


def design_kw(job: Job) -> dict:
    """Keyword arguments of the ``metrics.design_*`` functions for a job: the outer-fold column and, for V1, the row's
    group under the fold design's own grouping (column ``unit``: the copy group, the near-duplicate-key group or the
    compilation-DOI group), which ``metrics.v1_scoring_units`` checks against every single-group fold id."""
    return {"fold_col": "fold_id", "v1_group_col": "unit" if job.design == "V1" else EM.PUB_GROUP_COL}


def unit_reading_label(design: str, reading: str = "registered") -> str:
    """The ``unit_reading`` label of the ``metrics.design_*`` summaries (registered unit, or the V1 per-group reading)."""
    if reading == "registered":
        return f"registered: {EM.REGISTERED_UNIT_READING[design]}"
    if reading == "publication_group" and design == "V1":
        return "exploratory: publication group (the as-run pre-seal V1 reading)"
    raise ValueError(f"unit reading {reading!r} of {design}")


def filter_status(design: str, fname: str) -> str:
    """Status of a scoring-filter row (:data:`RESOLVED_READINGS` ``scoring_filter_status``)."""
    if fname == "none":
        return "registered"
    if design in ET.REGISTERED_SENSITIVITIES:
        names = [n for n, lab in ET.SCORING_FILTER_OF_SENSITIVITY.items() if lab == fname]
        return "registered" if any(ET.sensitivity_status(design, n) == "registered" for n in names) else "exploratory"
    return "registered" if fname in ASRUN_REGISTERED_FILTERS else "exploratory"


def cluster_unit_of(design: str) -> str:
    return {"V5": "extractant_system_key; publication group of the cell's majority rows",
            "V5P": "extractant_system_key; publication group of the cell's majority rows",
            "V5PAIR": "extractant_system_key; publication group of the cell pair's majority rows",
            "V1": "publication group (registered outer-fold unit: the remainder fold is one cluster)",
            "V2": "metal state", "V0": "none (diagnostic)"}[design]


def scoring_frame(pred: pd.DataFrame, attrs: pd.DataFrame, arm: str) -> pd.DataFrame:
    sub = pred[pred["arm"] == arm]
    fr = sub.join(attrs.drop(columns=[ROW_ID]), on="row_id")
    fr = fr.set_index("row_id", drop=False)
    if fr.index.has_duplicates:
        raise AssertionError("a scoring frame holds a row twice")
    fr[EM.PRED_COL] = fr["mean_logD"].astype(float)
    return fr


def tag(df: pd.DataFrame, job: Job, **extra) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "job", job.key)
    df["threshold_setting"] = job.threshold_setting
    df["cluster_unit_registered"] = cluster_unit_of(job.design)
    for k, v in extra.items():
        df[k] = v
    return df


def has_intervals(fr: pd.DataFrame) -> bool:
    return bool(np.isfinite(fr["lower_80"].to_numpy(dtype=float)).all() and len(fr))


def logd_block(job: Job, fr: pd.DataFrame, regime: EM.Regime, v6: pd.Series, filter_name: str,
               strata: tuple[str, ...], *, exploratory_units: bool = True) -> list[pd.DataFrame]:
    """log D summaries under the design's registered unit (``metrics.design_logd_summary``) and, for V1 when
    ``exploratory_units``, the per-publication-group reading beside it (status exploratory)."""
    kw = design_kw(job)
    expl = bool(exploratory_units and job.design == "V1")
    out = []
    base = EM.design_logd_summary(fr, regime, v6_mask=v6, exploratory_unit_readings=expl, **kw)
    out.append(tag(base, job, scoring_filter=filter_name))
    for sc in strata:
        s = EM.design_logd_summary(fr, regime, v6_mask=v6, exploratory_unit_readings=expl, strata_col=sc, **kw)
        out.append(tag(s[s["stratum"] != "all"], job, scoring_filter=filter_name))
    return out


def summaries_for_job(job: Job, pred: pd.DataFrame, attrs: pd.DataFrame, v6: pd.Series) -> tuple[list, list, list]:
    """(tidy summary frames, per-unit frames, interval frames) of one non-pair job, under the resolutions of 2026-09-15:
    registered averaging units (V1: the outer fold, the per-group reading beside it), filter statuses from the R19
    registry (:func:`filter_status`)."""
    design = DESIGN_LABEL[job.design]
    kw = design_kw(job)
    summ, units, cov = [], [], []
    seeds = sorted(pred["seed"].unique()) if job.design == "V0" else [-1]
    halves = ["NA"] if job.design == "V0" else ["S", "C"]
    filters = [("none", None)]
    if job.scoring_filters:
        filters += [("censoring_candidates_excluded_scoring", "censoring_candidate"),
                    ("acid_grid_rows_excluded_scoring", "acid_grid_flag")]
    if job.key in ("V5__primary", "V1__copy"):
        filters += [("wildcard_copies_excluded_scoring", f"wc_any__{job.stem}"),
                    ("wildcard_copies_strict_excluded_scoring", f"wc_strict__{job.stem}")]
    strata = ("metal_class", "dga_stratum", "acid_stratum") if job.key == "V5__primary" else ()
    readings = ("registered", "publication_group") if job.design == "V1" else ("registered",)
    for arm in ARMS:
        full = scoring_frame(pred, attrs, arm) if job.design != "V0" else None
        for seed in seeds:
            if job.design == "V0":
                fr_seed = pred[(pred["seed"] == seed)]
                fr_all = scoring_frame(fr_seed, attrs, arm)
            else:
                fr_all = full
            for half in halves:
                fr_h = fr_all[fr_all["half"] == half]
                if not len(fr_h):
                    continue
                det = seed == -1
                regime = EM.Regime(design=design, arm=arm, variant=job.variant, half=HALF_LABEL[half],
                                   seed=None if det else int(seed), seed_set="deterministic" if det else "discovery")
                for fname, col in filters:
                    frx = fr_h if col is None else fr_h[~fr_h[col].to_numpy(dtype=bool)]
                    reg = regime if filter_status(design, fname) == "registered" else replace(regime, status="exploratory")
                    summ += logd_block(job, frx, reg, v6, fname, strata if col is None else ())
                    if job.design == "V2":
                        for label, keep in (("focus7_lanthanides", lambda s: s.isin(FOCUS7)),
                                            ("ln_iii_14", lambda s: s.str.endswith("(III)") & frx["metal_class"].eq("lanthanide")),
                                            ("actinide_9", lambda s: frx["metal_class"].eq("actinide"))):
                            sub = frx[keep(frx[METAL]).to_numpy(dtype=bool)]
                            if len(sub):
                                s = EM.design_logd_summary(sub, reg, v6_mask=v6, exploratory_unit_readings=False, **kw)
                                s["stratum"] = label
                                summ.append(tag(s, job, scoring_filter=fname))
                if job.key == "V1__copy":
                    ereg = replace(regime, status="exploratory")
                    frx = fr_h[(fr_h["fold_id"] != SH.REMAINDER).to_numpy(dtype=bool)]
                    if len(frx):
                        summ += logd_block(job, frx, ereg, v6, "remainder_fold_excluded_scoring", (),
                                           exploratory_units=False)
                if has_intervals(fr_h):
                    iregime = EM.Regime(design=design, arm=arm, variant=job.variant, half=HALF_LABEL[half],
                                        seed=int(fr_h["seed"].iloc[0]) if job.design == "V0" else CONFORMAL_SEED,
                                        seed_set="discovery")
                    s = EC_interval(fr_h, iregime, job, v6)
                    summ.append(tag(s, job, scoring_filter="none"))
                for reading in readings:
                    pu = EM.design_per_unit_table(fr_h, design, v6_mask=v6, reading=reading,
                                                  levels=LEVELS if has_intervals(fr_h) else None, **kw)
                    pu = pu.reset_index(drop=True)
                    pu.insert(0, "arm", arm)
                    pu.insert(1, "half", HALF_LABEL[half])
                    pu.insert(2, "seed", None if det else int(seed))
                    pu["unit_reading"] = unit_reading_label(design, reading)
                    units.append(pu)
    return summ, units, cov


def asrun_v1_unit_variants(fr: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The AS-RUN sensitivities of the V1 averaging unit (frozen in ``difficulty.json``): the remainder fold's rows as
    ONE unit (since the section 3.2 resolution this is the registered unit), and the remainder fold's rows not scored."""
    rem = (fr["fold_id"] == SH.REMAINDER).to_numpy(dtype=bool)
    one = fr.copy()
    one.loc[rem, "unit"] = SH.REMAINDER
    return {"remainder_fold_as_one_unit": one, "remainder_fold_excluded_scoring": fr[~rem]}


def asrun_v1_summary(job: Job, pred: pd.DataFrame, attrs: pd.DataFrame, v6: pd.Series) -> pd.DataFrame:
    """The V1 rows ``difficulty.json`` reads, computed as the pre-seal run computed them (unit = the row's publication
    group, :func:`unit_cols_of`; filters none, the wildcard-copy filters and :func:`asrun_v1_unit_variants`).  They are
    never written to a table: ``difficulty.json`` is the as-run record whose SHA-256 pre-registration section 9 quotes."""
    out = []
    for arm in ARMS:
        full = scoring_frame(pred, attrs, arm)
        for half in ("S", "C"):
            fr_h = full[full["half"] == half]
            if not len(fr_h):
                continue
            regime = EM.Regime(design="V1", arm=arm, variant=job.variant, half=HALF_LABEL[half])
            frames = {"none": fr_h,
                      "wildcard_copies_excluded_scoring": fr_h[~fr_h[f"wc_any__{job.stem}"].to_numpy(dtype=bool)],
                      "wildcard_copies_strict_excluded_scoring": fr_h[~fr_h[f"wc_strict__{job.stem}"].to_numpy(dtype=bool)],
                      **asrun_v1_unit_variants(fr_h)}
            for fname, frx in frames.items():
                if len(frx):
                    s = EM.logd_summary(frx, regime, unit_cols=unit_cols_of("V1"), v6_mask=v6)
                    out.append(tag(s, job, scoring_filter=fname, unit_reading="as-run: publication group"))
    return pd.concat(out, ignore_index=True)


def EC_interval(fr: pd.DataFrame, regime: EM.Regime, job: Job, v6: pd.Series) -> pd.DataFrame:
    return EM.design_interval_summary(fr, regime, v6_mask=v6, levels=LEVELS,
                                      exploratory_unit_readings=job.design == "V1", **design_kw(job))


def v0_seed_means(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean over the five discovery seeds of every V0 per-seed summary value (a labelled extra row)."""
    s = summary[(summary["design"] == "V0") & summary["seed"].notna()]
    if s.empty:
        return s
    keys = ["job", "design", "variant", "half", "arm", "seed_set", "status", "averaging_unit", "stratum", "metric",
            "aggregation", "role", "scoring_filter", "threshold_setting", "cluster_unit_registered", "unit_reading"]
    g = s.groupby(keys, dropna=False).agg(value=("value", "mean"), n_units=("n_units", "mean"), n_rows=("n_rows", "mean"),
                                          n_seeds=("seed", "nunique")).reset_index()
    g["seed"] = "mean_of_discovery_seeds"
    g["note"] = "mean over " + g["n_seeds"].astype(str) + " discovery seeds (n_units, n_rows averaged)"
    return g.drop(columns=["n_seeds"])


def value_of(summary: pd.DataFrame, **sel) -> float:
    s = summary
    for k, v in sel.items():
        s = s[s[k].astype(object).eq(v)] if v is not None else s[s[k].isna()]
    if len(s) != 1:
        if LENIENT["on"] and len(s) == 0:
            return float("nan")
        raise KeyError(f"{len(s)} rows for {sel}")
    return float(s["value"].iloc[0])


# ============================================================================================= #
# V5-PAIR
# ============================================================================================= #

def pair_frames(pred: pd.DataFrame, attrs: pd.DataFrame, builder_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comparable test-test pairs regenerated after fold assignment inside each V5-PAIR fold (pairs.comparable_pairs,
    which runs pair_isolation_check), checked against the fold builder's pair list."""
    rows = pred[pred["arm"] == ARMS[0]][["row_id", "fold_id"]].drop_duplicates()
    df = rows.join(attrs.drop(columns=[ROW_ID]), on="row_id")
    df.index = pd.Index(df["fold_id"] + "|" + df["row_id"], name="label")
    df["fold"] = df["fold_id"]
    pairs = EP.comparable_pairs(df, fold_col="fold", carry_cols=("row_id",))
    got = pairs.assign(a=pairs["row_id_a"], b=pairs["row_id_b"])
    key_got = sorted(zip(got["fold"], [tuple(sorted(x)) for x in zip(got["a"], got["b"])]))
    key_b = sorted(zip(builder_pairs["fold_id"], [tuple(sorted(x)) for x in zip(builder_pairs["row_id_a"],
                                                                                builder_pairs["row_id_b"])]))
    if key_got != key_b:
        raise AssertionError(f"V5-PAIR pairs: {len(key_got)} regenerated != {len(key_b)} from the fold builder")
    return pairs, df


def pair_regime(lab: str, subset: str, half: str) -> EM.Regime:
    """Regime of one pair-summary block.  Registered: every arm-derived reading, FLAT and HEAVIER (undefined pairs
    excluded and counted, section 4) on all cell pairs.  Exploratory, role side, and never under the arm name HEAVIER:
    the undefined-counted-1/2 reading and every row of the Ln-Ln-only subset (the DIR5 alternatives the section 9
    S1(c) disclosure names)."""
    exploratory = lab == HEAVIER_HALF_READING or subset != "all_pairs"
    arm = PAIR_ALT_ARM.get((lab, subset), lab)
    if (exploratory and arm == "HEAVIER") or arm not in ALLOWED_LABELS:
        raise AssertionError(f"pair reading {lab!r} / {subset!r}: arm label {arm!r}")
    return EM.Regime(design="V5-PAIR", arm=arm, variant="primary", half=HALF_LABEL[half],
                     status="exploratory" if exploratory else "registered")


def pair_summaries(pred: pd.DataFrame, attrs: pd.DataFrame, pairs: pd.DataFrame, rows: pd.DataFrame,
                   v6: pd.Series, halves: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    job = JOB_BY_KEY["V5PAIR__primary"]
    folds_all = pd.Series("test", index=rows.index, dtype=object)
    pairs = pairs.copy()
    pairs["half"] = pairs[SYSTEM].map(halves)
    if pairs["half"].isna().any():
        raise AssertionError("V5-PAIR pair without a registered half")
    v6_lab = pd.Series(rows["row_id"].map(v6).to_numpy(dtype=bool), index=rows.index)
    out, units = [], []
    labels = list(ARMS) + list(PAIR_YARDSTICKS) + [HEAVIER_HALF_READING]
    for lab in labels:
        if lab in ARMS:
            pm = pred[pred["arm"] == lab]
            series = pd.Series(pm["mean_logD"].to_numpy(dtype=float),
                               index=pd.Index(pm["fold_id"] + "|" + pm["row_id"]))
            pred_ls, donly = EP.derived_logsf(pairs, series), False
        elif lab == "FLAT":
            pred_ls, donly = EP.flat_logsf(pairs), False
        elif lab == "HEAVIER":
            pred_ls, donly = EP.heavier_direction(pairs), True
        else:
            pred_ls, donly = EP.heavier_direction(pairs).fillna(0.0), True
        for half in ("S", "C"):
            sel = pairs["half"] == half
            ps, pl = pairs[sel], pred_ls[sel]
            for subset, keep in (("all_pairs", None), ("Ln-Ln_only", "Ln-Ln")):
                if keep is not None:
                    k = ps["category_class"] == keep
                    ps2, pl2 = ps[k], pl[k]
                else:
                    ps2, pl2 = ps, pl
                if not len(ps2):
                    continue
                regime = pair_regime(lab, subset, half)
                s = EP.pair_summary(ps2, pl2, regime, v6_mask=v6_lab, folds=folds_all, direction_only=donly,
                                    strata_col="category_class" if keep is None else None)
                if regime.status == "exploratory":
                    s["role"] = "side"
                s = tag(s, job, scoring_filter="none", pair_subset=subset, yardstick_reading=lab)
                out.append(s)
                if keep is None:
                    sc = EP.score_pairs(ps2, pl2, design="V5-PAIR", direction_only=donly)
                    pu = EP.pair_unit_table(sc, design="V5-PAIR")
                    cc = ps2.groupby(list(EP.CELL_PAIR_COLS))["category_class"].first().reset_index()
                    pu = pu.merge(cc, on=list(EP.CELL_PAIR_COLS), how="left")
                    pu.insert(0, "yardstick_or_arm", lab)
                    pu.insert(1, "half", HALF_LABEL[half])
                    units.append(pu)
    return pd.concat(out, ignore_index=True), pd.concat(units, ignore_index=True)


# ============================================================================================= #
# contrasts
# ============================================================================================= #

CONTRASTS: tuple[tuple[str, str], ...] = (("B3x", "B0"), ("B3i", "B0"), ("B3i", "B3x"), ("B3l", "B3x"),
                                          ("B4x", "B3x"), ("B7", "B3x"), ("B3x", "B1"), ("B3x", "B2"),
                                          ("B3i", "B1"), ("B3i", "B2"))
V1_EXTRA: tuple[tuple[str, str], ...] = (("B3", "B0"), ("B3", "B1"), ("B3", "B2"))
#: (unit_variant label, metrics unit reading, rows filter) of the descriptive contrasts; V1 prints the registered
#: outer-fold unit, the exploratory per-group reading and the remainder fold unscored (section 3.2 resolution)
CONTRAST_VARIANTS: dict[str, tuple[tuple[str, str, str | None], ...]] = {
    "V1__copy": (("registered_unit", "registered", None),
                 ("publication_group_exploratory", "publication_group", None),
                 ("remainder_fold_excluded_scoring", "registered", "remainder_fold_excluded")),
}


def contrasts_for(job: Job, pred: pd.DataFrame, attrs: pd.DataFrame, v6: pd.Series) -> pd.DataFrame:
    recs = []
    for unit_variant, reading, rows_filter in CONTRAST_VARIANTS.get(job.key, (("registered_unit", "registered", None),)):
        recs += _contrasts_one(job, pred, attrs, v6, unit_variant, reading, rows_filter)
    return pd.DataFrame(recs)


def _contrasts_one(job: Job, pred: pd.DataFrame, attrs: pd.DataFrame, v6: pd.Series, unit_variant: str, reading: str,
                   rows_filter: str | None) -> list:
    """Per-unit values from ``metrics.design_per_unit_table`` (the section 2 V6 guard runs on every scoring index) and
    every registered cluster unit from ``metrics.design_unit_clusters``."""
    design = DESIGN_LABEL[job.design]
    kw = design_kw(job)
    ucols = EM.registered_unit_cols(design) if reading == "registered" else (kw["v1_group_col"],)
    recs = []
    pairs = CONTRASTS + (V1_EXTRA if job.design == "V1" else ())
    for half in ("S", "C"):
        mae, clusters = {}, {}
        if not (pred["half"] == half).any():
            continue
        for arm in ARMS:
            fr = scoring_frame(pred, attrs, arm)
            fr = fr[fr["half"] == half]
            if rows_filter == "remainder_fold_excluded":
                fr = fr[(fr["fold_id"] != SH.REMAINDER).to_numpy(dtype=bool)]
            pu = EM.design_per_unit_table(fr, design, v6_mask=v6, reading=reading, **kw)
            mae[arm] = pu["mae"]
            if not clusters:
                clusters = EM.design_unit_clusters(fr, design, reading=reading, pub_group_col=EM.PUB_GROUP_COL, **kw)
        for cand, comp in pairs:
            meta = {"family": "DESCRIPTIVE_baseline_difficulty", "claim": "none (pre-seal baseline difficulty; not an R19 "
                    "contrast)", "job": job.key, "design": design, "variant": job.variant,
                    "unit_variant": unit_variant, "unit_reading": unit_reading_label(design, reading),
                    "half": HALF_LABEL[half], "metric": "mae",
                    "averaging_unit": "+".join(ucols), "comparator": comp, "candidate": cand, "seed_set": "deterministic",
                    "status": "exploratory", "comparator_macro": float(mae[comp].mean()),
                    "candidate_macro": float(mae[cand].mean())}
            name = f"{cand} vs {comp}"
            for cu, cl in clusters.items():
                br = ET.paired_cluster_bootstrap(mae[comp], mae[cand], cl, contrast=name, cluster_unit=cu)
                r = br.record(meta)
                tt = ET.tost(br)
                r.update({"tost_verdict_eps0.05": tt["verdict"], "tost_low_90": tt["low_90"], "tost_high_90": tt["high_90"],
                          "decides": False})
                recs.append(r)
            br = ET.unclustered_bootstrap(mae[comp], mae[cand], contrast=name)
            r = br.record(meta)
            r["decides"] = False
            recs.append(r)
    return recs


# ============================================================================================= #
# figures
# ============================================================================================= #

def figure_f07(pred: pd.DataFrame, attrs: pd.DataFrame, macro: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), sharex=True, sharey=True)
    frs = {arm: scoring_frame(pred, attrs, arm).query("half == 'S'") for arm in ("B0", "B3x", "B3i")}
    allv = np.concatenate([np.r_[f["log_D"].to_numpy(), f["pred"].to_numpy()] for f in frs.values()])
    lo, hi = float(np.nanmin(allv)) - 0.3, float(np.nanmax(allv)) + 0.3
    desc = {"B0": "B0 global mean", "B3x": "B3x nearest-radius metal, same system",
            "B3i": "B3i radius interpolation, same system"}
    for ax, (arm, fr) in zip(axes, frs.items()):
        ax.plot([lo, hi], [lo, hi], color="#8a8a8a", lw=1.0, zorder=1)
        for cls in ("lanthanide", "actinide", "other"):
            s = fr[fr["metal_class"] == cls]
            if len(s):
                ax.scatter(s["log_D"], s["pred"], s=9, alpha=0.45, color=CLASS_COLORS[cls], edgecolors="none",
                           label=f"{cls} ({s[EM.CELL_COLS[0]].astype(str).str.cat(s[SYSTEM].astype(str)).nunique()} cells, "
                                 f"{len(s)} rows)", zorder=2)
        ax.set_title(f"{desc[arm]}\nmacro MAE over cells {macro[arm]:.3f} log D", fontsize=9.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(color="#e6e6e6", lw=0.6)
        ax.set_xlabel("measured log D")
        ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    axes[0].set_ylabel("predicted log D (hidden cell)")
    n_cells = int(frs["B0"][list(EM.CELL_COLS)].drop_duplicates().shape[0])
    fig.suptitle("Does borrowing a neighbouring metal inside the same extractant system reconstruct a hidden "
                 "metal x extractant cell better than a global mean?\n"
                 f"V5-primary exact leave-one-cell-out, selection half ({n_cells} cells, {len(frs['B0'])} rows); "
                 "pre-seal baselines only", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths.ensure_dir(out.parent)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def figure_f08(cell: pd.DataFrame, comparator: str, rho: dict, out: Path) -> None:
    panels = (("nearest_radius_distance_A", "|delta Shannon radius| to the nearest-radius training metal (A)"),
              ("n_neighbour_metals", "other metal states measured with the system (training)"),
              ("condition_distance_system_mean", "mean standardised condition distance to the system's training rows"))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True)
    for ax, (col, lab) in zip(axes, panels):
        for cls in ("lanthanide", "actinide", "other"):
            s = cell[cell["metal_class"] == cls]
            if len(s):
                ax.scatter(s[col], s["mae"], s=22, alpha=0.7, color=CLASS_COLORS[cls], edgecolors="white",
                           linewidths=0.6, label=f"{cls} ({len(s)} cells)")
        ax.set_xlabel(lab, fontsize=8.5)
        ax.grid(color="#e6e6e6", lw=0.6)
        ax.set_title(f"Spearman rho {rho[col]['rho']:.2f} over {rho[col]['n_cells']} cells (descriptive)", fontsize=9)
    axes[0].set_ylabel(f"cell MAE of {comparator} (log D)")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle(f"Is the V5 lookup comparator ({comparator}) wrong by more on hidden cells that sit farther from "
                 "their training support?\nV5-primary, selection half; support features from a SupportIndex fitted on "
                 "each fold's training rows; the section 8 reliability floor is not assessed, so no correlation is "
                 "interpreted", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths.ensure_dir(out.parent)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# ============================================================================================= #
# main
# ============================================================================================= #

def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _macro(summary: pd.DataFrame, job: str, fname: str, half: str, *, unit_reading: str | None = None,
           status: str | None = None) -> dict:
    """``{arm: value}`` and ``n_units`` of the unit-macro MAE (stratum ``all``) of one job / filter / half / reading."""
    s = summary[(summary["job"] == job) & (summary["scoring_filter"] == fname) & (summary["half"] == half)
                & (summary["aggregation"] == "unit_macro") & (summary["stratum"] == "all") & (summary["metric"] == "mae")]
    if unit_reading is not None:
        s = s[s["unit_reading"].astype(str).str.startswith(unit_reading)]
    if status is not None:
        s = s[s["status"] == status]
    vals = {a: value_of(s, arm=a) for a in ARMS if (s["arm"] == a).any() or not LENIENT["on"]}
    n = s.loc[s["arm"] == "B0", "n_units"]
    return {"macro_mae": vals, "n_units": int(n.iloc[0]) if len(n) == 1 else None,
            "status": str(s["status"].iloc[0]) if len(s) else None,
            "unit_reading": str(s["unit_reading"].iloc[0]) if len(s) else None}


def resolved_difficulty(summary: pd.DataFrame, diff: dict, preds: dict, attrs: pd.DataFrame, support_update: dict | None,
                        status_out: dict, diff_sha: str, full_data: bool) -> dict:
    """``evaluation/preseal/difficulty_resolved.json``: the pre-seal numbers under the orchestrator's resolutions of
    2026-09-15 that ``difficulty.json`` (the as-run record quoted by section 9) cannot carry without changing its digest
    (task X, finding VR-01).  Every value is read from the resolved summary table this run writes."""
    out: dict = {
        "schema": "gen19.preseal_difficulty_resolved.v1",
        "note": ("difficulty.json is the AS-RUN record whose SHA-256 pre-registration section 9 quotes; it is re-rendered "
                 "byte-identical and never rewritten with resolved labels. This file carries the pre-seal numbers under "
                 "the resolutions of 2026-09-15 (V1 outer-fold unit, registered wildcard-copy sensitivity, s4 reading, "
                 "V0 averaging unit), read from tables/preseal_summary.csv (column unit_reading). Baselines only; no "
                 "learned model; no V6_TARGET_ROWS row scored; DESCRIPTIVE."),
        "difficulty_json": {"sha256_rendered": diff_sha, "sha256_section9": REGISTERED_DIFFICULTY_SHA256,
                            "byte_identical_to_section9": diff_sha == REGISTERED_DIFFICULTY_SHA256,
                            "full_data_run": full_data},
        "readings": RESOLVED_READINGS, "registered_unit_reading": dict(EM.REGISTERED_UNIT_READING),
        "sensitivity_status": {d: {n: ET.sensitivity_status(d, n) for n in
                                   ET.REGISTERED_SENSITIVITIES[d] + ET.EXPLORATORY_SENSITIVITIES.get(d, ())}
                               for d in ("V5", "V1", "V5-P", "V5-PAIR", "V2")},
        "resolved_orchestrator_decisions": {
            "S1c_same_fitted_folds": {"registered_reading": ET.S1C_REGISTERED_FOLD_READING,
                                      "readings": ET.S1C_FOLD_READINGS, "finding": "task X VR-02 (consistency)",
                                      "resolution": "section 9 S1(c) 'Same fitted folds' (2026-09-15)"},
            "wildcard_copy_sensitivity_V5P_V5PAIR": {
                "registered": all(ET.sensitivity_status(d, ET.WILDCARD_COPY_SENSITIVITY) == "registered"
                                  for d in ("V5-P", "V5-PAIR")),
                "finding": "task X VR-05 (leakage)", "resolution": "section 8 R19 item 6 (2026-09-15)"}}}
    if "V1__copy" in preds:
        v1 = {"design": "V1", "variant": "copy", "comparator": "B3 (identical to B4 by construction)",
              "registered_unit": EM.REGISTERED_UNIT_READING["V1"]}
        for half, lab in (("selection", "selection"), ("confirmation", "confirmation_descriptive")):
            v1[lab] = _macro(summary, "V1__copy", "none", half, unit_reading="registered", status="registered")
            v1[f"{lab}_exploratory_publication_group"] = _macro(summary, "V1__copy", "none", half,
                                                                unit_reading="exploratory")
        v1["exploratory_remainder_fold_excluded_scoring_selection"] = _macro(
            summary, "V1__copy", "remainder_fold_excluded_scoring", "selection")
        asrun_one = diff["V1"]["unit_sensitivities_exploratory"]["remainder_fold_as_one_unit"]["macro_mae_selection"]
        v1["registered_unit_equals_asrun_remainder_fold_as_one_unit_selection"] = bool(
            v1["selection"]["macro_mae"] == asrun_one)
        v1["registered_exploratory_per_group_equals_asrun_headline_selection"] = bool(
            v1["selection_exploratory_publication_group"]["macro_mae"] == diff["V1"]["macro_mae_selection"])
        for fname in (ET.WILDCARD_COPY_SENSITIVITY, ET.WILDCARD_COPY_STRICT_SENSITIVITY):
            v1[f"{fname}_selection"] = _macro(summary, "V1__copy", fname, "selection", unit_reading="registered")
        out["V1"] = v1
    if "V5__primary" in preds:
        v5 = {"design": "V5", "variant": "primary", "unit": EM.REGISTERED_UNIT_READING["V5"]}
        for fname in ("none", ET.WILDCARD_COPY_SENSITIVITY, ET.WILDCARD_COPY_STRICT_SENSITIVITY):
            v5[f"{fname}_selection"] = _macro(summary, "V5__primary", fname, "selection")
        rows_s = preds["V5__primary"][(preds["V5__primary"]["arm"] == "B0") & (preds["V5__primary"]["half"] == "S")]["row_id"]
        v5["n_selection_rows_excluded_by_filter"] = {
            ET.WILDCARD_COPY_SENSITIVITY: int(attrs.loc[rows_s, "wc_any__V5__primary__exact"].sum()),
            ET.WILDCARD_COPY_STRICT_SENSITIVITY: int(attrs.loc[rows_s, "wc_strict__V5__primary__exact"].sum())}
        v5["difficulty_numbers"] = "unchanged by the resolutions: difficulty.json -> V5 (L5, C5, N0, delta5, comparator)"
        out["V5"] = v5
    if "V0__rows" in preds:
        out["V0"] = {"unit": EM.REGISTERED_UNIT_READING["V0"],
                     "F1_interval": "transfer.seed_mean_cluster_bootstrap (publication-group clusters, same resampled "
                                    "groups in every seed); not computed pre-seal (no candidate exists)",
                     "macro_mae_mean_of_seeds": diff.get("V0_diagnostic", {}).get("macro_mae_mean_of_seeds")}
    out["V5_PAIR"] = {"DIR5": diff.get("V5_PAIR", {}).get("DIR5"), "role": RESOLVED_READINGS["DIR5"]}
    out["support"] = {"s4_reading": ES.S4_READING,
                      "jobs": {k: {"support_score": v.get("support_score"), "domain_status": v.get("domain_status"),
                                   "support_score_file": v.get("support_score_file"),
                                   "support_score_summary": v.get("support_score_summary")}
                               for k, v in status_out.get("jobs", {}).items()},
                      "update": None if support_update is None else {
                          k: support_update[k] for k in ("function", "as_run_status_equals_update_manifest_pre_update_sha256")}}
    out["intervals"] = RESOLVED_READINGS["conformal_seed"]
    return out


def main(argv=None) -> int:
    ns = parse_args(argv)
    LENIENT["on"] = bool(ns.max_folds)
    t_start = time.perf_counter()
    # digests of the code as imported (taken now, not at the end of a long run, so a later edit is not recorded)
    code_sha = {rel(p): sha_file(p) for p in (
        Path(__file__), Path(B.__file__), Path(MA.__file__), Path(I.__file__), Path(EM.__file__), Path(EP.__file__),
        Path(ET.__file__), Path(EC.__file__), Path(SG.__file__), Path(FI.__file__), Path(CH.__file__), Path(SH.__file__),
        Path(MH.__file__), Path(ES.__file__), Path(L.__file__),
        Path(__file__).resolve().parent / "g19_update_support_preseal.py")}      # the module apply_support_update runs from
    dirs = out_dirs(Path(ns.out_root))
    keys = [k for k in ns.jobs.split(",") if k]
    bad = [k for k in keys if k not in JOB_BY_KEY]
    if bad:
        raise SystemExit(f"unknown jobs {bad}")
    jobs = [JOB_BY_KEY[k] for k in keys]
    feas_json = json.loads((paths.DATA_AUDIT_DIR / "feasibility.json").read_text(encoding="utf-8"))
    index_json = json.loads((paths.FOLDS_DIR / "INDEX.json").read_text(encoding="utf-8"))
    rebuild = compute_run_record(paths.MANIFESTS_DIR / f"{NAME}.json", dirs) \
        if ns.tables_only and not ns.no_manifest else None
    aggregation = aggregation_run_record(paths.MANIFESTS_DIR / f"{NAME}.json", dirs, jobs) \
        if ns.skip_compute and not ns.tables_only and not ns.no_manifest and not ns.max_folds else None
    ctx_run = nullcontext(SimpleNamespace(inputs=lambda *a: None, outputs=lambda *a: None, extra={})) if ns.no_manifest \
        else Run(NAME, args=vars(ns), seed=None, extra={"discovery_seeds": list(FI.DISCOVERY_SEEDS)})
    ctx_stage = tempfile.TemporaryDirectory(prefix="g19_preseal_tables_") if ns.tables_only else nullcontext(None)
    outputs: list[Path] = []
    with ctx_run as run, ctx_stage as stage_dir:
        sink = OutputSink(dirs, Path(stage_dir) if stage_dir else None)
        fold_manifests = [m for m in (paths.MANIFESTS_DIR / "g19_build_folds.json",
                                      paths.MANIFESTS_DIR / "g19_build_folds_incremental.json") if m.exists()]
        run.inputs(paths.ARCHIVE_MASTER, FI.PUB_COMPONENTS_CSV, FI.HALVES_CSV, paths.DATA_AUDIT_DIR / "feasibility.json",
                   paths.DESCRIPTORS_DIR / "extractant_systems.csv", paths.DESCRIPTORS_DIR / "extractant_components.csv",
                   paths.DESCRIPTORS_DIR / "metals.csv", paths.FOLDS_DIR / "INDEX.json",
                   paths.FOLDS_DIR / "V5PAIR__primary__pairs.parquet", WILDCARD_CROSSINGS_CSV, *fold_manifests,
                   *sorted({paths.FOLDS_DIR / f"{j.stem}.{e}" for j in jobs for e in ("json", "parquet")}))
        log("row attributes (g19_feasibility.load_frame flags)")
        attrs = build_attrs(feas_json)
        coext_ids = sorted(attrs.index[attrs["acidic_coextractant_modifier"]])
        results = {}
        if not ns.skip_compute:
            order = sorted(jobs, key=lambda j: [x.key for x in JOBS].index(j.key))
            log(f"computing {len(order)} jobs on {ns.workers} worker process(es)")
            with ProcessPoolExecutor(max_workers=ns.workers, initializer=init_worker,
                                     initargs=(coext_ids, ns.max_folds, str(dirs.root), ns.verify_inner_guard_folds)) as ex:
                futs = {j.key: ex.submit(run_job, j) for j in order}
                for k, fu in futs.items():
                    results[k] = fu.result()
                    log(f"{k}: {results[k]['n_folds_fitted']} folds fitted, {results[k]['runtime_s']} s, "
                        f"inner guard calls {results[k]['inner_guard_calls']}")
            paths.ensure_dir(dirs.ev)
            flog = pd.DataFrame([r for k in results for r in results[k]["fold_log"]])
            write_csv(flog, dirs.ev / "guard_log.csv")
            vol = {k: {kk: v for kk, v in r.items() if kk in ("runtime_s", "n_folds_fitted", "inner_guard_calls")}
                   for k, r in results.items()}
            vol["total_runtime_s_compute"] = round(time.perf_counter() - t_start, 1)
            paths.ensure_dir(paths.MANIFESTS_DIR / "run_info")
            if not ns.no_manifest:
                write_json(paths.MANIFESTS_DIR / "run_info" / f"{NAME}_jobs.json", vol)
            write_json(dirs.ev / "inner_guard_verification.json",
                       {"rule": READINGS["V5_inner_guard"],
                        "folds": [v for k in results for v in results[k]["verify"]],
                        "inner_guard_calls_by_job": {k: r["inner_guard_calls"] for k, r in results.items()}})
        outputs += [dirs.ev / "guard_log.csv", dirs.ev / "inner_guard_verification.json"]
        v6 = pd.Series(attrs["v6_target_row"].to_numpy(dtype=bool), index=attrs.index)
        preds = {j.key: pd.read_parquet(dirs.pred / f"{j.key}.parquet") for j in jobs}
        outputs += [dirs.pred / f"{j.key}.parquet" for j in jobs]
        # ---- run checks: no V6 row scored, no forbidden arm, B4 == B3 under V1
        checks = {"n_prediction_rows_by_job": {k: int(len(p)) for k, p in preds.items()}}
        for k, p in preds.items():
            ids = pd.Index(p["row_id"].unique())
            FR.assert_not_scored(ids, v6, what=f"predictions {k}")
            if not set(p["arm"]) <= ALLOWED_LABELS:
                raise AssertionError(f"{k}: arm labels {set(p['arm']) - ALLOWED_LABELS}")
            if attrs.loc[ids, "acidic_coextractant_modifier"].any() or attrs.loc[ids, METAL].isna().any() \
                    or attrs.loc[ids, METAL].eq(I.SR_III).any():
                raise AssertionError(f"{k}: an excluded row is scored")
        checks["v6_target_rows_scored"] = 0
        if "V1__copy" in preds:
            p = preds["V1__copy"]
            b3 = p[p["arm"] == "B3"].set_index(["fold_id", "row_id"])["mean_logD"]
            b4 = p[p["arm"] == "B4"].set_index(["fold_id", "row_id"])["mean_logD"]
            checks["V1_B4_equals_B3"] = bool(b3.sort_index().equals(b4.sort_index()))
            checks["V1_B4_minus_B3_max_abs"] = float((b3.sort_index() - b4.sort_index()).abs().max())
        # ---- summaries
        log("summaries")
        summ, units_all = [], {}
        for j in jobs:
            if j.design == "V5PAIR":
                continue
            s, u, _ = summaries_for_job(j, preds[j.key], attrs, v6)
            summ += s
            units_all[j.key] = pd.concat(u, ignore_index=True)
        summary = pd.concat(summ, ignore_index=True)
        summary["seed"] = summary["seed"].astype(object)
        summary = pd.concat([summary, v0_seed_means(summary)], ignore_index=True)
        halves = FI.registered_halves("V5_system")
        pair_summary = pair_units = None
        if "V5PAIR__primary" in preds:
            bp = pd.read_parquet(paths.FOLDS_DIR / "V5PAIR__primary__pairs.parquet")
            if ns.max_folds:
                bp = bp[bp["fold_id"].isin(set(preds["V5PAIR__primary"]["fold_id"]))]
            pairs, rows = pair_frames(preds["V5PAIR__primary"], attrs, bp)
            pair_summary, pair_units = pair_summaries(preds["V5PAIR__primary"], attrs, pairs, rows, v6, halves)
            checks["V5PAIR_test_test_pairs"] = int(len(pairs))
        # ---- fallback counts
        fb = []
        for k, p in preds.items():
            g = p.assign(fallback_reason=p["fallback_reason"].fillna("")).groupby(
                ["half", "arm", "fallback_level"], dropna=False).size().rename("n_row_entries").reset_index()
            g["share_of_arm_half"] = g["n_row_entries"] / g.groupby(["half", "arm"])["n_row_entries"].transform("sum")
            g.insert(0, "job", k)
            g["half"] = g["half"].map(HALF_LABEL)
            fb.append(g)
        fallback = pd.concat(fb, ignore_index=True)
        # ---- contrasts
        log("contrasts (paired cluster bootstrap, 10,000 resamples, seed 19)")
        con = [contrasts_for(JOB_BY_KEY[k], preds[k], attrs, v6) for k in ("V5__primary", "V1__copy", "V2__element")
               if k in preds]
        contrasts = pd.concat(con, ignore_index=True) if con else pd.DataFrame()
        # ---- support tables
        support_status, dom_counts, cov_dom, f08_cells, f08_rho = {}, [], [], None, {}
        for j in jobs:
            if not j.support:
                continue
            sp = pd.read_parquet(dirs.support / f"{j.key}.parquet")
            outputs.append(dirs.support / f"{j.key}.parquet")
            n = len(sp)
            amb_s4 = int((sp["support_s4_ambiguous_query_system_inclusion"] | sp["support_s4_ambiguous_missing_descriptor"]).sum())
            amb_dom = int(sp["domain_status_ambiguous"].sum())
            support_status[j.key] = {
                "n_scored_row_entries": n,
                "support_score": "computed" if amb_s4 == 0 else "not computed (registration ambiguous on scored rows)",
                "n_rows_s4_ambiguous_query_system_inclusion": int(sp["support_s4_ambiguous_query_system_inclusion"].sum()),
                "n_rows_s4_ambiguous_missing_descriptor": int(sp["support_s4_ambiguous_missing_descriptor"].sum()),
                "domain_status": "computed" if amb_dom == 0 else "not computed (registration ambiguous on scored rows)",
                "n_rows_domain_status_ambiguous_anion_clause": amb_dom,
                "tau_by_fold_summary": {c: {"min": float(sp.groupby("fold_id")[c].first().min()),
                                            "max": float(sp.groupby("fold_id")[c].first().max())}
                                        for c in ("tau_in", "tau_ext", "tau_max")}}
            if amb_dom == 0:
                h = preds[j.key][preds[j.key]["arm"] == "B0"][["fold_id", "row_id", "half"]]
                d = sp.merge(h, on=["fold_id", "row_id"], how="left")
                d = d.join(attrs[[METAL, SYSTEM]], on="row_id")
                c = d.groupby(["half", "domain_status_candidate"]).agg(
                    n_row_entries=("row_id", "size"), n_cells=(METAL, lambda s: 0)).reset_index()
                c["n_cells"] = [int(d[(d["half"] == hh) & (d["domain_status_candidate"] == st)][[METAL, SYSTEM]]
                                    .drop_duplicates().shape[0]) for hh, st in zip(c["half"], c["domain_status_candidate"])]
                c.insert(0, "job", j.key)
                c["half"] = c["half"].map(HALF_LABEL)
                c = c.rename(columns={"domain_status_candidate": "domain_status"})
                dom_counts.append(c)
                if j.key == "V5__primary":
                    for arm in ARMS:
                        fr = scoring_frame(preds[j.key], attrs, arm)
                        fr["domain_status_candidate"] = fr["row_id"].map(
                            sp.set_index("row_id")["domain_status_candidate"])
                        for half in ("S", "C"):
                            frh = fr[fr["half"] == half]
                            if has_intervals(frh):
                                reg = EM.Regime(design="V5", arm=arm, variant="primary", half=HALF_LABEL[half],
                                                seed=CONFORMAL_SEED, seed_set="discovery")
                                cov_dom.append(tag(EC.coverage_by_category(frh, reg, category_col="domain_status_candidate",
                                                                           unit_cols=EM.CELL_COLS, v6_mask=v6), j))
            if j.key == "V5__primary":
                f08_src = sp
        # ---- support status under the section 13 s4 resolution (task X, finding VR-01): the as-run status above is only
        #      the input of scripts/g19_update_support_preseal.apply_support_update, never written as it stands
        status_out = {"readings": READINGS["support_score_domain_status"], "jobs": support_status}
        counts_out = pd.concat(dom_counts, ignore_index=True) if dom_counts else None
        support_scores, support_update = {}, None
        sup_jobs = [j.key for j in jobs if j.support]
        if sup_jobs and counts_out is not None:
            log("support_score under the section 13 s4 resolution (g19_update_support_preseal.apply_support_update)")
            upd = load_support_update_module()
            with tempfile.TemporaryDirectory(prefix="g19_preseal_asrun_support_") as tdir:
                asrun_sha = {rel(dirs.ev / "support_status.json"): sha_file(write_json(Path(tdir) / "s.json", status_out)),
                             rel(dirs.ev / "domain_status_counts.csv"): sha_file(write_csv(counts_out, Path(tdir) / "c.csv"))}
            model_rows, t_sup, idmap_sup, sys_sup, comps_sup = upd.row_table()
            status_out, counts_out, support_scores, sup_checks = upd.apply_support_update(
                status_out, counts_out, sup_jobs, ev=dirs.ev, t=t_sup, idmap=idmap_sup, systems=sys_sup, comps=comps_sup,
                model=model_rows, preds=preds, say=log)
            del model_rows, t_sup, idmap_sup
            upd_manifest = paths.MANIFESTS_DIR / f"{upd.NAME}.json"
            pre = json.loads(upd_manifest.read_text(encoding="utf-8")).get("pre_update_sha256") \
                if upd_manifest.exists() else None
            support_update = {
                "function": "scripts/g19_update_support_preseal.py apply_support_update", "s4_reading": ES.S4_READING,
                "jobs": sup_jobs, "checks": sup_checks, "as_run_status_sha256_rendered": asrun_sha,
                "as_run_status_equals_update_manifest_pre_update_sha256": (None if pre is None else bool(
                    all(pre.get(k) == v for k, v in asrun_sha.items())))}
        # ---- difficulty numbers (the AS-RUN record: V1 per publication group, as frozen in difficulty.json)
        log("difficulty numbers")
        asrun = summary if "V1__copy" not in preds else pd.concat(
            [summary[summary["job"] != "V1__copy"], asrun_v1_summary(JOB_BY_KEY["V1__copy"], preds["V1__copy"], attrs, v6)],
            ignore_index=True)
        diff = {"schema": "gen19.preseal_difficulty.v1", "half_used_for_decisions": "selection",
                "readings": READINGS, "not_run_pre_seal": NOT_RUN, "sources": {
                    "summary": rel(dirs.tables / "preseal_summary.csv"),
                    "pair_summary": rel(dirs.tables / "preseal_pair_summary.csv")}}
        S = asrun[(asrun["scoring_filter"] == "none") & (asrun["aggregation"] == "unit_macro")]
        if "V5__primary" in preds:
            v5 = S[(S["job"] == "V5__primary") & (S["stratum"] == "all") & (S["metric"] == "mae")]
            mac = {h: {a: value_of(v5, half=HALF_LABEL[h], arm=a) for a in ARMS} for h in ("S", "C")}
            choice, l5 = ET.stronger_lookup(mac["S"])
            u = units_all["V5__primary"]
            us = u[(u["half"] == "selection") & u["arm"].isin(["B3x", "B3i", "B3l", "B4x"])]
            oracle = us.groupby(list(EM.CELL_COLS))["mae"].min()
            n_cells_s = int(us[us["arm"] == "B3x"].shape[0])
            n_sys_s = int(us.loc[us["arm"] == "B3x", SYSTEM].nunique())
            diff["V5"] = {
                "design": "V5", "variant": "primary", "scheme": "exact leave-one-cell-out", "averaging_unit": "hidden cell",
                "n_cells_selection": n_cells_s, "n_systems_selection": n_sys_s,
                "macro_mae_selection": mac["S"], "macro_mae_confirmation_descriptive": mac["C"],
                "lookup_comparator": {"choice": choice, "rule": "lower selection-half macro MAE of B3x and B3i (section 5), "
                                      "fixed here before any learned model is scored; a tie keeps B3x",
                                      "B3x": mac["S"]["B3x"], "B3i": mac["S"]["B3i"]},
                "L5": l5, "C5": mac["S"]["B0"], "N0": ET.N0, "L5_minus_N0": l5 - ET.N0, "rho5": ET.RHO5,
                "delta5": ET.delta5(l5), "delta5_formula": "max(0.05, 0.20 x (L5 - N0))",
                "A5": "not computed pre-seal (section 9: the additive factorisation arm is fitted only after sealing)",
                "lookup_oracle_macro_mae_selection": float(oracle.mean()),
                "lookup_oracle_note": "per-cell minimum over B3x, B3i, B3l, B4x (optimistic; selected on the scored rows)"}
        if pair_summary is not None:
            ps = pair_summary[(pair_summary["aggregation"] == "unit_macro") & (pair_summary["stratum"] == "all")
                              & (pair_summary["half"] == "selection")]
            ps_all, ps_ln = ps[ps["pair_subset"] == "all_pairs"], ps[ps["pair_subset"] == "Ln-Ln_only"]
            mthr = "direction_accuracy_abs_ge_0.3"

            def pv(frame, lab, metric):
                s = frame[(frame["yardstick_reading"] == lab) & (frame["metric"] == metric)]
                return {"value": float(s["value"].iloc[0]), "n_units": int(s["n_units"].iloc[0]),
                        "note": str(s["note"].iloc[0])} if len(s) == 1 else None
            def maxv(items) -> float:
                vals = [v["value"] for v in items if v and np.isfinite(v["value"])]
                return max(vals) if vals else float("nan")
            comp = {lab: pv(ps_all, lab, mthr) for lab in ("HEAVIER", "B3x", "B3i")}
            dir5 = maxv(comp.values())
            alt_half = maxv([pv(ps_all, HEAVIER_HALF_READING, mthr), comp["B3x"], comp["B3i"]])
            ln = {lab: pv(ps_ln, lab, mthr) for lab in ("HEAVIER", "B3x", "B3i")}
            cnts = index_json["placeholders"]["V5PAIR_eligible_cell_pairs"]
            diff["V5_PAIR"] = {
                "retained": cnts["retained"], "n_cell_pairs_selection": cnts["by_half"]["S"]["n_cell_pairs"],
                "SF5_FLAT": pv(ps_all, "FLAT", "logsf_mae")["value"],
                "SF5_FLAT_detail": pv(ps_all, "FLAT", "logsf_mae"),
                "DIR5": dir5, "DIR5_components": comp, "DIR5_reading": READINGS["DIR5"],
                "DIR5_alternative_undefined_counts_half": alt_half,
                "DIR5_alternative_ln_ln_pairs_only": maxv(ln.values()),
                "DIR5_alternative_ln_ln_components": ln,
                "lookup_derived_logsf_mae_selection": {lab: pv(ps_all, lab, "logsf_mae")["value"] for lab in ARMS},
                "chosen_lookup_derived_logsf_mae": pv(ps_all, diff["V5"]["lookup_comparator"]["choice"], "logsf_mae")["value"]
                if "V5" in diff else None}
        if "V1__copy" in preds:
            v1 = S[(S["job"] == "V1__copy") & (S["stratum"] == "all") & (S["metric"] == "mae")]
            m1 = {h: {a: value_of(v1, half=HALF_LABEL[h], arm=a) for a in ARMS} for h in ("S", "C")}
            diff["V1"] = {"comparator": "B3 (identical to B4 by construction; checked)", "L1_B3_macro_mae_selection": m1["S"]["B3"],
                          "C1_B0_macro_mae_selection": m1["S"]["B0"], "macro_mae_selection": m1["S"],
                          "macro_mae_confirmation_descriptive": m1["C"], "averaging_unit": "publication group",
                          "n_units_selection": int(v1.loc[(v1["half"] == "selection") & (v1["arm"] == "B3"), "n_units"]
                                                   .sum()),
                          "B4_equals_B3": checks.get("V1_B4_equals_B3")}
            sv = asrun[(asrun["job"] == "V1__copy") & (asrun["aggregation"] == "unit_macro")
                       & (asrun["stratum"] == "all") & (asrun["metric"] == "mae")]
            p1 = preds["V1__copy"]
            ps1 = p1[(p1["arm"] == "B0") & (p1["half"] == "S")]
            diff["V1"]["unit_sensitivities_exploratory"] = {
                "reading": READINGS["V1_unit_sensitivities"],
                "n_selection_units_from_remainder_fold": int(ps1.loc[ps1["fold_id"] == SH.REMAINDER, "unit"].nunique()),
                "n_selection_rows_in_remainder_fold": int((ps1["fold_id"] == SH.REMAINDER).sum()),
                "n_selection_rows": int(len(ps1)),
                "n_confirmation_units_from_remainder_fold": int(p1.loc[(p1["arm"] == "B0") & (p1["half"] == "C")
                                                                       & (p1["fold_id"] == SH.REMAINDER), "unit"].nunique()),
                **{fname: {"macro_mae_selection": {a: value_of(sv[sv["scoring_filter"] == fname], half="selection", arm=a)
                                                   for a in ARMS},
                           "n_units_selection": int(sv.loc[(sv["scoring_filter"] == fname) & (sv["half"] == "selection")
                                                           & (sv["arm"] == "B0"), "n_units"].sum())}
                   for fname in ("remainder_fold_as_one_unit", "remainder_fold_excluded_scoring")}}
        if "V2__element" in preds:
            v2 = S[(S["job"] == "V2__element") & (S["metric"] == "mae")]
            groups = {}
            for grp in ("focus7_lanthanides", "all", "ln_iii_14", "actinide_9"):
                groups[grp] = {h: {a: value_of(v2[v2["stratum"] == grp], half=HALF_LABEL[h], arm=a) for a in ARMS}
                               for h in ("S", "C")}
            ch2, l2 = ET.stronger_lookup(groups["focus7_lanthanides"]["S"])
            diff["V2"] = {"comparator_choice": ch2, "comparator_rule": READINGS["V2_comparator"],
                          "L2_focus7_selection": l2, "C2_B0_focus7_selection": groups["focus7_lanthanides"]["S"]["B0"],
                          "L2_all_states_selection": groups["all"]["S"][ch2],
                          "macro_mae_by_summary": groups, "averaging_unit": "metal state"}
        if "V0__rows" in preds:
            v0 = asrun[(asrun["job"] == "V0__rows") & (asrun["seed"] == "mean_of_discovery_seeds")
                       & (asrun["metric"] == "mae") & (asrun["aggregation"] == "unit_macro")
                       & (asrun["stratum"] == "all") & (asrun["scoring_filter"] == "none")]
            diff["V0_diagnostic"] = {"comparator": "B3", "macro_mae_mean_of_seeds": dict(zip(v0["arm"], v0["value"].astype(float))),
                                     "averaging_unit": "publication group; mean over 5 discovery seeds"}
        cover = {}
        for k in ("V5__primary", "V1__copy", "V2__element"):
            if k in preds:
                f = fallback[(fallback["job"] == k) & (fallback["half"] == "selection")]
                cover[k] = {a: dict(zip(f.loc[f["arm"] == a, "fallback_level"], f.loc[f["arm"] == a, "share_of_arm_half"]
                                        .astype(float).round(6))) for a in ("B3", "B3x", "B3i", "B3l", "B4x", "B7")}
        diff["fallback_share_selection"] = cover
        if "V5__primary" in preds:
            base = preds["V5__primary"].set_index(["fold_id", "row_id", "arm"])["mean_logD"]
            ident = {}
            for k in ("V5__cell_only", "V5__parent_structure", "V5__sr_iii_dropped_training"):
                if k not in preds:
                    continue
                other = preds[k].set_index(["fold_id", "row_id", "arm"])["mean_logD"]
                both = base.to_frame("a").join(other.to_frame("b"), how="inner")
                d = (both["a"] - both["b"]).abs().groupby(level="arm").max()
                ident[k] = {"n_row_entries_compared": int(len(both)),
                            "max_abs_prediction_difference_by_arm": {a: float(d.get(a, float("nan"))) for a in ARMS},
                            "arms_with_identical_predictions": sorted(a for a in ARMS if a in d.index and d[a] == 0.0)}
            diff["V5_sensitivity_prediction_identity"] = {
                "note": "row-level comparison with V5__primary on the same (fold_id, row_id); a sensitivity whose "
                        "predictions are identical adds no independent evidence for that arm",
                "jobs": ident}
        if "V5__primary" in preds or "V1__copy" in preds:
            wc = {}
            for k in ("V5__primary", "V1__copy"):
                if k not in preds:
                    continue
                sk = asrun[(asrun["job"] == k) & (asrun["aggregation"] == "unit_macro") & (asrun["stratum"] == "all")
                           & (asrun["metric"] == "mae") & (asrun["half"] == "selection")]
                stem = JOB_BY_KEY[k].stem
                rows_s = preds[k][(preds[k]["arm"] == "B0") & (preds[k]["half"] == "S")]["row_id"]
                wc[k] = {"n_selection_rows_flagged_any": int(attrs.loc[rows_s, f"wc_any__{stem}"].sum()),
                         "n_selection_rows_flagged_strict": int(attrs.loc[rows_s, f"wc_strict__{stem}"].sum()),
                         **{fname: {a: value_of(sk[sk["scoring_filter"] == fname], half="selection", arm=a) for a in ARMS}
                            for fname in ("none", "wildcard_copies_excluded_scoring",
                                          "wildcard_copies_strict_excluded_scoring")}}
            diff["wildcard_copy_sensitivity_exploratory"] = {"reading": READINGS["wildcard_copies"], "jobs": wc}
        if "V5__primary" in preds and "V5" in diff:
            comp_arm = diff["V5"]["lookup_comparator"]["choice"]
            u = units_all["V5__primary"]
            cm = u[(u["half"] == "selection") & (u["arm"] == comp_arm)].set_index(list(EM.CELL_COLS))
            h = preds["V5__primary"][preds["V5__primary"]["arm"] == "B0"][["fold_id", "row_id", "half"]]
            sp = f08_src.merge(h, on=["fold_id", "row_id"])
            sp = sp[sp["half"] == "S"].join(attrs[[METAL, SYSTEM, "metal_class"]], on="row_id")
            agg = sp.groupby([METAL, SYSTEM]).agg(nearest_radius_distance_A=("nearest_radius_distance_A", "first"),
                                                  n_neighbour_metals=("n_neighbour_metals", "first"),
                                                  condition_distance_system_mean=("condition_distance_system", "mean"),
                                                  nearest_radius_metal=("nearest_radius_metal", "first"),
                                                  metal_class=("metal_class", "first"))
            f08_cells = agg.join(cm[["mae", "n_rows"]]).reset_index()
            for col in ("nearest_radius_distance_A", "n_neighbour_metals", "condition_distance_system_mean"):
                ok = f08_cells[np.isfinite(f08_cells[col].astype(float))]
                f08_rho[col] = {"rho": EM.spearman_rho(ok[col].astype(float), ok["mae"]), "n_cells": int(len(ok))}
            diff["F08_descriptive_spearman"] = {"comparator": comp_arm, "half": "selection", "values": f08_rho,
                                                "note": "descriptive; section 8 reliability floor not assessed"}
        diff["run_checks"] = checks
        # ---- difficulty.json is the as-run record quoted by pre-registration section 9: a full-data rerun must render it
        #      byte-identical before anything is written (task X, finding VR-01)
        full_data = (not ns.max_folds and set(keys) == set(JOB_BY_KEY)
                     and Path(ns.out_root).resolve() == Path(paths.G19_ROOT).resolve())
        with tempfile.TemporaryDirectory(prefix="g19_preseal_difficulty_") as tdir:
            diff_sha = sha_file(write_json(Path(tdir) / "difficulty.json", diff))
        if full_data and diff_sha != REGISTERED_DIFFICULTY_SHA256:
            raise AssertionError(f"difficulty.json would change ({diff_sha} != the section 9 digest "
                                 f"{REGISTERED_DIFFICULTY_SHA256}); nothing was written")
        resolved = resolved_difficulty(summary, diff, preds, attrs, support_update, status_out, diff_sha, full_data)
        # ---- write (sink: in place, or the --tables-only staging directory)
        log("writing tables, figures")
        paths.ensure_dir(dirs.tables)
        paths.ensure_dir(dirs.units)
        paths.ensure_dir(dirs.pairs)
        summary_cols = ["job"] + list(EM.SUMMARY_COLUMNS) + ["scoring_filter", "unit_reading", "threshold_setting",
                                                             "cluster_unit_registered"]
        summary = summary[summary_cols].sort_values(["job", "scoring_filter", "half", "arm", "stratum", "metric",
                                                     "aggregation"], kind="mergesort", key=lambda s: s.astype(str))
        write_csv(summary, sink(dirs.tables / "preseal_summary.csv"))
        write_csv(fallback, sink(dirs.tables / "preseal_fallback_counts.csv"))
        outputs += [dirs.tables / "preseal_summary.csv", dirs.tables / "preseal_fallback_counts.csv"]
        if len(contrasts):
            write_csv(contrasts, sink(dirs.tables / "preseal_contrasts.csv"))
            outputs.append(dirs.tables / "preseal_contrasts.csv")
        if pair_summary is not None:
            write_csv(pair_summary, sink(dirs.tables / "preseal_pair_summary.csv"))
            write_csv(pair_units, sink(dirs.pairs / "V5PAIR__primary__cell_pairs.csv"))
            outputs += [dirs.tables / "preseal_pair_summary.csv", dirs.pairs / "V5PAIR__primary__cell_pairs.csv"]
        for k, u in units_all.items():
            write_csv(u, sink(dirs.units / f"{k}__units.csv"))
            outputs.append(dirs.units / f"{k}__units.csv")
        write_json(sink(dirs.ev / "support_status.json"), status_out)
        outputs.append(dirs.ev / "support_status.json")
        if counts_out is not None:
            write_csv(counts_out, sink(dirs.ev / "domain_status_counts.csv"))
            outputs.append(dirs.ev / "domain_status_counts.csv")
        upd = load_support_update_module() if support_scores else None
        for k, frame in support_scores.items():
            sf = upd.score_file(dirs.ev, k)
            frame.to_parquet(sink(sf), index=False, compression="zstd")
            outputs.append(sf)
        if cov_dom:
            write_csv(pd.concat(cov_dom, ignore_index=True), sink(dirs.tables / "preseal_coverage_by_domain_status.csv"))
            outputs.append(dirs.tables / "preseal_coverage_by_domain_status.csv")
        if f08_cells is not None:
            write_csv(f08_cells, sink(dirs.support / "V5__primary__cell_support_error_selection.csv"))
            outputs.append(dirs.support / "V5__primary__cell_support_error_selection.csv")
            figure_f08(f08_cells, diff["V5"]["lookup_comparator"]["choice"], f08_rho,
                       sink(dirs.figures / "F08_preseal_error_vs_support.png"))
            outputs.append(dirs.figures / "F08_preseal_error_vs_support.png")
        if "V5" in diff:
            figure_f07(preds["V5__primary"], attrs, diff["V5"]["macro_mae_selection"],
                       sink(dirs.figures / "F07_preseal_pred_vs_measured.png"))
            outputs.append(dirs.figures / "F07_preseal_pred_vs_measured.png")
        write_json(sink(dirs.ev / "difficulty.json"), diff)
        write_json(sink(dirs.ev / "difficulty_resolved.json"), resolved)
        outputs.append(dirs.ev / "difficulty_resolved.json")
        outputs.append(dirs.ev / "difficulty.json")
        write_text(sink(dirs.tables / "preseal_summary.md"), summary_markdown(summary, diff))
        outputs.append(dirs.tables / "preseal_summary.md")
        # ---- forbidden-arm scan of every data output (its new content: staged under --tables-only)
        hits = {}
        for p in outputs:
            c = sink.current(p)
            if c.suffix in (".json", ".md") and c.exists():
                found = sorted(set(FORBIDDEN_TOKEN.findall(c.read_text(encoding="utf-8"))))
                if found:
                    hits[rel(p)] = found
            elif c.suffix == ".csv" and c.exists():
                q = pd.read_csv(c, dtype=str, keep_default_na=False)
                # system keys are SMILES (a boron ring closure could read like an arm name): scan headers and the
                # columns that carry arm / yardstick / fallback labels
                label_cols = [col for col in q.columns if col in ("arm", "candidate", "comparator", "contrast",
                                                                  "fallback_level", "yardstick_or_arm",
                                                                  "yardstick_reading", "job", "variant")]
                found = sorted(set(FORBIDDEN_TOKEN.findall(" ".join(q.columns))) |
                               {x for col in label_cols for v in q[col].unique() for x in FORBIDDEN_TOKEN.findall(v)})
                if found:
                    hits[rel(p)] = found
            elif c.suffix == ".parquet" and c.exists() and p.parent == dirs.pred:
                q = pd.read_parquet(c, columns=["arm", "fallback_level"])
                found = sorted({x for col in q.columns for x in q[col].dropna().astype(str).unique()
                                if FORBIDDEN_TOKEN.search(x)})
                if found:
                    hits[rel(p)] = found
        if hits:
            raise AssertionError(f"forbidden arm names in outputs: {hits}")
        big = {rel(p): sink.current(p).stat().st_size for p in outputs
               if sink.current(p).exists() and sink.current(p).stat().st_size > 5_000_000}
        run_checks = {**checks, "forbidden_arm_names_found": {}, "files_over_5MB": big,
                      "assert_not_scored_calls": "every fold scoring index (worker) and every prediction file (parent)",
                      "jobs": [asdict(j) for j in jobs]}
        write_json(sink(dirs.ev / "run_checks.json"), run_checks)
        outputs.append(dirs.ev / "run_checks.json")
        committed = sink.commit()
        if ns.tables_only:
            log(f"--tables-only: rewrote {committed['tables_rewritten']}; byte-identical, not rewritten: "
                f"{committed['tables_byte_identical_not_rewritten'] + committed['other_outputs_rerendered_byte_identical_not_rewritten']}")
        run.outputs(*[p for p in outputs if p.exists()])
        run.extra.update({"readings": READINGS, "resolved_readings": RESOLVED_READINGS, "not_run_pre_seal": NOT_RUN,
                          "files_over_5MB": big,
                          "fold_design_hashes": {j.stem: index_json["designs"][j.stem]["design_hash"] for j in jobs},
                          "fold_build_manifests_sha256": {rel(m): paths.digests(m)["sha256"] for m in fold_manifests},
                          "code_sha256": code_sha, "support_update": support_update,
                          "difficulty_json": {"sha256_rendered": diff_sha, "sha256_section9": REGISTERED_DIFFICULTY_SHA256,
                                              "full_data_run": full_data,
                                              "byte_identical_to_section9": diff_sha == REGISTERED_DIFFICULTY_SHA256},
                          "arms": list(ARMS), "pair_yardsticks": list(PAIR_YARDSTICKS),
                          "learned_arms_fitted_or_scored": []})
        if aggregation is not None:
            run.extra["aggregation_rebuild"] = {
                "rule": "--skip-compute: every output re-aggregated from the stored predictions and support feature files "
                        "(verified unchanged against the replaced manifest; no fold fitted) under the resolutions of "
                        "2026-09-15; this manifest supersedes the replaced one and every later manifest listed under "
                        "outputs_superseded_before_this_run",
                **aggregation}
        if rebuild is not None:
            carried_aggregation = rebuild.pop("_carried_aggregation_rebuild", None)
            if carried_aggregation is not None:          # the aggregation run this table rebuild rests on
                run.extra["aggregation_rebuild"] = carried_aggregation
            base = rebuild["compute_run"].get("tables_sha256") or {}
            run.extra["tables_only_rebuild"] = {
                "rule": "tables/ rebuilt from the stored evaluation/preseal prediction and support files; no fold was "
                        "fitted; every non-table output matched the replaced manifest, re-rendered byte-identical, "
                        "and was not rewritten; only tables whose bytes changed were written",
                **rebuild, **committed,
                "tables_differing_from_compute_run": sorted(
                    rel(p) for p in outputs if sink.is_table(p) and not paths.matches(p, base.get(rel(p)) or ""))}
    log(f"done in {time.perf_counter() - t_start:.0f} s")
    return 0


def summary_markdown(summary: pd.DataFrame, diff: dict) -> str:
    s = summary[(summary["scoring_filter"] == "none") & (summary["aggregation"] == "unit_macro")
                & (summary["metric"] == "mae") & (summary["status"] == "registered")]
    cols = [("V5__primary", "all", "selection", None, "V5 primary S"), ("V5__primary", "all", "confirmation", None, "V5 primary C"),
            ("V1__copy", "all", "selection", None, "V1 S"), ("V1__copy", "all", "confirmation", None, "V1 C"),
            ("V2__element", "focus7_lanthanides", "selection", None, "V2 focus-7 S"),
            ("V2__element", "all", "selection", None, "V2 all S"),
            ("V0__rows", "all", "none", "mean_of_discovery_seeds", "V0 (diagnostic)")]
    lines = ["# Pre-seal Phase C baseline difficulty (macro MAE, log D)", "",
             "Generated by `scripts/g19_run_preseal.py` from `tables/preseal_summary.csv` (metric `mae`, aggregation "
             "`unit_macro`, scoring filter `none`, status `registered`). Units: V5 hidden cell, V1 outer fold with the pooled "
             "remainder fold as one unit (pre-registration section 3.2 resolution; the per-publication-group reading is "
             "printed in the table as exploratory), V2 metal state, V0 publication group (mean over 5 discovery seeds). "
             "S = selection half, C = confirmation half (descriptive only). Baselines only; no learned model exists. "
             "The as-run section 9 numbers are `evaluation/preseal/difficulty.json`; the resolved ones "
             "`evaluation/preseal/difficulty_resolved.json`.", "",
             "| arm | " + " | ".join(c[-1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for arm in ARMS:
        vals = []
        for job, stratum, half, seed, _ in cols:
            x = s[(s["job"] == job) & (s["stratum"] == stratum) & (s["half"] == half) & (s["arm"] == arm)]
            if seed is not None:
                x = x[x["seed"] == seed]
            else:
                x = x[x["seed"].isna() | (x["seed"].astype(str) == "None")]
            vals.append(f"{float(x['value'].iloc[0]):.3f}" if len(x) == 1 else "")
        lines.append(f"| {arm} | " + " | ".join(vals) + " |")
    if "V5" in diff:
        d = diff["V5"]
        lines += ["", f"V5 lookup comparator (selection half): **{d['lookup_comparator']['choice']}**; "
                      f"L5 = {d['L5']:.4f}, C5 = {d['C5']:.4f}, N0 = {d['N0']:.4f}, delta5 = {d['delta5']:.4f} "
                      "(`evaluation/preseal/difficulty.json` > `V5`)."]
    if "V5_PAIR" in diff:
        d = diff["V5_PAIR"]
        lines += [f"V5-PAIR (selection half): SF5_FLAT = {d['SF5_FLAT']:.4f}, DIR5 = {d['DIR5']:.4f} "
                  "(`difficulty.json` > `V5_PAIR`)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
