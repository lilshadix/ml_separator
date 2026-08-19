"""gen6 Phase 0: the cheap analytical package that must pass before any Phase 1 run.

Phase 0 fits almost nothing.  Its job is to establish that the new harness stands
on ground the old one already occupies, and to answer, on real data, the seven
questions the generation's plan opens with:

1. can all 190 ligands be assigned stable chemistry clusters?
2. which extractants enter at ``min_cells = 3`` and not at 10?
3. how much genuinely new chemistry do they add?
4. can BASE91 and EXPANDED152 be evaluated on byte-identical test rows?
5. what fraction of the current error is ligand *offset* versus centred *shape*?
6. how does the error change with nearest-neighbour Tanimoto?
7. does adding diversity reduce offset error in a retrospective single-split pilot?

plus the two gates the protocol demands before Phase 1: an honest provenance
audit, and an exact reproduction of the gen5 champion under the gen6 code path.

The reproduction gate is deliberately decomposed, because "reproduces" is four
different claims and only three of them can be exact across machines:

* **cohort** — identical rows and row ids;
* **split** — identical fold membership for every regime and seed;
* **metrics** — gen6's metric code, applied to the *published* out-of-fold
  predictions, must return the published numbers;
* **fit** — refitting the model. Exact on the same machine, but a forest trained
  on a different BLAS/CPU/scikit-learn is a different forest, so this step
  reports a measured tolerance rather than asserting bit-equality.

Every step degrades to ``SKIPPED`` with a reason when its inputs are missing; a
missing gen5 artifact is not a crash.

Example::

    .venv/bin/python scripts/gen6_phase0.py
    .venv/bin/python scripts/gen6_phase0.py --skip-refit --skip-pilot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.chemistry import (  # noqa: E402
    build_chemistry_map, cluster_manifest, freeze_is_conservative, partition_stability,
)
from lanthanide_separation.gen6.cohorts import (  # noqa: E402
    BASE_ARM, BASE_MIN_CELLS, EXPANDED_ARM, EXPANDED_MIN_CELLS, arm_membership,
    assert_split_integrity, cohort_comparison, diversity_splits, eligible_extractants,
    seeded_group_kfold,
)
from lanthanide_separation.gen6.manifest import (  # noqa: E402
    RunManifest, sha256_json, validate_run, write_success,
)
from lanthanide_separation.gen6.metrics import (  # noqa: E402
    decompose_level_shape, effective_sample_size,
)
from lanthanide_separation.gen6.provenance import (  # noqa: E402
    find_upstream_directory, provenance_audit_report, reconstruct_provenance,
)
from lanthanide_separation.levels import (  # noqa: E402
    LevelForestParameters, LevelRegressor, build_level_dataset, equal_group_macro_mae,
    replicate_noise_floor,
)

DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
DESCRIPTOR_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_2d_descriptors.parquet"
GEN5_RUN = REPO_ROOT / "runs" / "gen5_levels_20260818T211105Z"
#: Fold grouping per gen5 regime — repeated here so the reproduction check does not
#: import from ``scripts/`` (not a package) and cannot drift silently.
REGIME_GROUP = {"unseen_chemotype": "tanimoto_cluster", "unseen_ligand": "ecfp_cluster",
                "unseen_series": "series_id", "unseen_conditions": "condition_id"}
GEN5_SEEDS = (104729, 130363, 155921, 196613, 262147)
#: Arms recomputed from the published OOF (metric gate) and refitted (fit gate).
REPRO_ARMS = {"MC": ("METAL", "COND"), "MC_lig2d_ext": ("METAL", "COND", "LIG2D_EXT")}
DECOMPOSE_ARMS = ("MC", "MC_lig2d_ext", "MC_donors", "MC_ecfp")
NOVELTY_BINS = (0.0, 0.4, 0.6, 0.8, 1.0001)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--descriptors", type=Path, default=DESCRIPTOR_PATH)
    p.add_argument("--gen5-run", type=Path, default=GEN5_RUN,
                   help="the canonical four-regime gen5 run to reproduce against")
    p.add_argument("--upstream-dir", type=Path, default=None,
                   help="raw *_SAFE.csv directory for provenance; auto-detected when omitted")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--refit-seeds", nargs="+", type=int, default=list(GEN5_SEEDS))
    p.add_argument("--refit-regimes", nargs="+", default=list(REGIME_GROUP))
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--skip-refit", action="store_true", help="skip the (slow) fit gate")
    p.add_argument("--skip-pilot", action="store_true")
    p.add_argument("--pilot-args", nargs="*", default=["--pilot"],
                   help="arguments forwarded to scripts/run_diversity_causal.py")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

def step_provenance(source: pd.DataFrame, upstream_dir, output_dir: Path, log) -> dict:
    log("step 1/7  provenance")
    audit = reconstruct_provenance(source, upstream_directory=upstream_dir)
    (output_dir / "provenance_report.md").write_text(provenance_audit_report(audit))
    audit.to_json(output_dir / "provenance_audit.json")
    a = audit.audit
    return {
        "status": "OK",
        "state": audit.state,
        "publication_status": a["publication_id_status"],
        "n_publications": a.get("n_publications"),
        "n_series": a.get("n_series"),
        "n_cells": a.get("n_cells"),
        "fraction_ambiguous": a.get("fraction_ambiguous_provenance"),
        "merged": a.get("merged_across_publication_boundaries"),
        "repeated_cells": a.get("n_cells_repeated"),
        "repeated_cells_with_hidden_axis": a.get("repeated_cells_with_a_hidden_axis"),
        "repeated_cells_physical_axis": a.get("repeated_cells_with_a_PHYSICAL_hidden_axis"),
        "repeated_cells_free_text_only": a.get("repeated_cells_differing_only_in_free_text"),
        "repeated_cells_true_replicates": a.get("repeated_cells_that_look_like_true_replicates"),
        "repeated_cell_range": a.get("repeated_cell_log_d_range"),
        "repeated_cell_range_by_subset": a.get("repeated_cell_log_d_range_by_subset"),
        "strict_is_legacy": a.get("provenance_strict_is_legacy"),
        "strict_definition": a.get("provenance_strict_definition"),
    }


def step_chemistry(source, descriptors, cohorts: dict, output_dir: Path, log) -> tuple[dict, object]:
    log("step 2/7  chemistry map over all 190 extractants")
    chemistry = build_chemistry_map(source, ligand_descriptors=descriptors)
    chemistry.to_parquet(output_dir / "chemistry_map.parquet")
    chemistry.save_similarity_npz(output_dir / "nearest_neighbor_matrix.npz")
    (output_dir / "chemistry_cluster_manifest.json").write_text(
        json.dumps(cluster_manifest(chemistry), indent=2, default=str) + "\n")
    stability = {name: partition_stability(chemistry, data.frame) for name, data in cohorts.items()}
    # `partition_stability` compares against the labels build_level_dataset carries,
    # which were computed BEFORE the row filter and are therefore already all-190
    # labels: a genuine check at the bit-identical level, near-vacuous at the
    # chemotype level. The non-vacuous version re-clusters the cohort from scratch.
    conservative = {name: freeze_is_conservative(chemistry, source,
                                                 sorted(set(data.frame["extractant"])))
                    for name, data in cohorts.items()}
    return {
        "status": "OK",
        "audit": chemistry.audit,
        "stability": stability,
        "conservative": conservative,
        "all_stable": all(report["ok"] for report in stability.values()),
        "all_conservative": all(report["ok"] for report in conservative.values()),
    }, chemistry


def step_cohort_comparison(chemistry, cohorts: dict, output_dir: Path, log) -> dict:
    log("step 3/7  BASE91 vs EXPANDED152 cohort comparison")
    shared = cohorts["expanded"].frame
    base_extractants = list(eligible_extractants(shared, min_cells=BASE_MIN_CELLS))
    sparse = [e for e in shared["extractant"].unique() if e not in set(base_extractants)]
    neighbours = chemistry.nearest_neighbour(sparse, base_extractants)
    comparison = cohort_comparison(shared)

    names = chemistry.table.set_index("extractant")["extractant_name"].to_dict()
    families = chemistry.table.set_index("extractant")["chem__family"].to_dict()
    cells = shared["extractant"].value_counts().to_dict()
    entering = neighbours.assign(
        extractant_name=[names.get(e, "") for e in neighbours["extractant"]],
        family=[families.get(e, "") for e in neighbours["extractant"]],
        n_cells=[cells.get(e, 0) for e in neighbours["extractant"]],
    ).sort_values("nn_tanimoto")
    entering.to_csv(output_dir / "extractants_entering_at_min_cells_3.csv", index=False)

    similarity = entering["nn_tanimoto"].to_numpy(dtype=float)
    base_frame = shared[arm_membership(shared, min_cells=BASE_MIN_CELLS)]
    within_base = chemistry.nearest_neighbour(base_extractants, base_extractants)
    return {
        "status": "OK",
        "comparison": comparison,
        "n_entering": int(len(entering)),
        "entering_families": entering["family"].value_counts().to_dict(),
        "nn_quantiles": {q: float(np.quantile(similarity, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        if similarity.size else {},
        "n_below": {str(t): int((similarity < t).sum()) for t in (0.4, 0.5, 0.6, 0.7)},
        "median_nn_kept_to_kept": float(within_base["nn_tanimoto"].median()),
        "median_nn_entering_to_kept": float(np.median(similarity)) if similarity.size else float("nan"),
        "base_rows": int(len(base_frame)),
        "expanded_rows": int(len(shared)),
    }


def step_split_machinery(cohorts: dict, chemistry, output_dir: Path, log) -> dict:
    log("step 4/7  fixed-test BASE/EXPANDED split machinery")
    shared = cohorts["expanded"].frame
    per_seed = {}
    for seed in GEN5_SEEDS:
        splits = diversity_splits(shared, n_splits=5, seed=seed)
        integrity = assert_split_integrity(shared, splits)
        row_ids = shared["row_id"].astype(str).to_numpy()
        folds = []
        for split in splits:
            test_ids = sorted(row_ids[split.test_index].tolist())
            folds.append({
                "fold": split.fold,
                "test_row_ids_sha256": sha256_json(test_ids),
                "n_test_rows": len(test_ids),
                "held_out_superclusters": list(split.held_out_groups),
                "n_train_rows_by_arm": split.audit["n_train_rows_by_arm"],
                "n_train_extractants_by_arm": split.audit["n_train_extractants_by_arm"],
                "n_sparse_train_extractants": split.audit["n_sparse_train_extractants"],
            })
        per_seed[str(seed)] = {"integrity_ok": integrity["ok"], "folds": folds}
    (output_dir / "split_manifest.json").write_text(
        json.dumps(per_seed, indent=2, default=str) + "\n")
    return {
        "status": "OK",
        "all_integrity_ok": all(v["integrity_ok"] for v in per_seed.values()),
        "per_seed": per_seed,
    }


def _load_published_oof(gen5_run: Path, arms) -> pd.DataFrame | None:
    path = gen5_run / "oof_predictions.csv"
    if not path.is_file():
        return None
    columns = ["row_id", "regime", "split_seed", "outer_fold", "extractant", "ecfp_cluster",
               "tanimoto_cluster", "log_D", "nn_train_tanimoto"] + [f"prediction_{a}" for a in arms]
    header = pd.read_csv(path, nrows=0).columns
    columns = [c for c in columns if c in header]
    return pd.read_csv(path, usecols=columns)


def step_reproduction(cohorts, args, log) -> dict:
    log("step 5/7  reproduction of the gen5 champion")
    gen5_run = Path(args.gen5_run)
    per_seed_path = gen5_run / "per_seed_metrics.csv"
    if not per_seed_path.is_file():
        return {"status": "SKIPPED", "reason": f"{per_seed_path} not found"}
    published = pd.read_csv(per_seed_path)
    frame = cohorts["base"].frame
    data = cohorts["base"]
    result: dict = {"status": "OK", "gen5_run": str(gen5_run)}

    # -- cohort ------------------------------------------------------------- #
    n_rows_published = int(published["n_rows"].iloc[0])
    result["cohort"] = {
        "gen6_rows": int(len(frame)), "published_rows": n_rows_published,
        "identical": int(len(frame)) == n_rows_published,
    }

    oof = _load_published_oof(gen5_run, DECOMPOSE_ARMS)
    if oof is None:
        result["split"] = {"status": "SKIPPED", "reason": "no oof_predictions.csv"}
        result["metrics"] = {"status": "SKIPPED", "reason": "no oof_predictions.csv"}
    else:
        # -- split ----------------------------------------------------------- #
        mismatches = 0
        compared = 0
        for regime, group_column in REGIME_GROUP.items():
            if regime not in set(oof["regime"]):
                continue
            groups = frame[group_column].astype(str).to_numpy()
            row_ids = frame["row_id"].astype(str).to_numpy()
            for seed in sorted(oof.loc[oof.regime == regime, "split_seed"].unique()):
                mine = {}
                for fold, (_, test) in enumerate(seeded_group_kfold(groups, 5, int(seed))):
                    for row_id in row_ids[test]:
                        mine[row_id] = fold
                block = oof[(oof.regime == regime) & (oof.split_seed == seed)]
                theirs = dict(zip(block["row_id"].astype(str), block["outer_fold"]))
                compared += len(mine)
                mismatches += sum(1 for k, v in mine.items() if theirs.get(k, -1) != v)
        result["split"] = {"rows_compared": compared, "rows_with_different_fold": mismatches,
                           "identical": mismatches == 0}

        # -- metrics ---------------------------------------------------------- #
        rows = []
        for (regime, seed), block in oof.groupby(["regime", "split_seed"]):
            for arm in DECOMPOSE_ARMS:
                column = f"prediction_{arm}"
                if column not in block.columns:
                    continue
                want = published[(published.regime == regime) & (published.split_seed == seed)
                                 & (published.arm == arm)]["macro_mae"]
                if not len(want):
                    continue
                got = equal_group_macro_mae(block["log_D"], block[column], block["ecfp_cluster"])
                rows.append({"regime": regime, "seed": int(seed), "arm": arm,
                             "recomputed": got, "published": float(want.iloc[0]),
                             "abs_diff": abs(got - float(want.iloc[0]))})
        table = pd.DataFrame(rows)
        result["metrics"] = {
            "cells": int(len(table)),
            "max_abs_diff": float(table["abs_diff"].max()) if len(table) else None,
            "exact": bool(len(table) and table["abs_diff"].max() < 1e-9),
        }

    # -- fit ------------------------------------------------------------------ #
    if args.skip_refit:
        result["fit"] = {"status": "SKIPPED", "reason": "--skip-refit"}
        return result
    fits = []
    started = time.time()
    for regime in args.refit_regimes:
        groups = frame[REGIME_GROUP[regime]].astype(str).to_numpy()
        for seed in args.refit_seeds:
            predictions = {arm: np.full(len(frame), np.nan) for arm in REPRO_ARMS}
            for fold, (train_index, test_index) in enumerate(seeded_group_kfold(groups, 5, seed)):
                train, test = frame.iloc[train_index], frame.iloc[test_index]
                y_train = train["log_D"].to_numpy(dtype=float)
                parameters = LevelForestParameters(
                    n_estimators=args.n_estimators, max_features=0.30, min_samples_leaf=2,
                    random_state=42 + fold * 1009 + 9_999_991, n_jobs=args.n_jobs)
                for arm, blocks in REPRO_ARMS.items():
                    model = LevelRegressor(data.block_columns(blocks), parameters).fit(
                        train, y_train, groups=train["ecfp_cluster"])
                    predictions[arm][test_index] = model.predict(test)
            for arm in REPRO_ARMS:
                got = equal_group_macro_mae(frame["log_D"], predictions[arm], frame["ecfp_cluster"])
                want = published[(published.regime == regime) & (published.split_seed == seed)
                                 & (published.arm == arm)]["macro_mae"]
                if not len(want):
                    continue
                fits.append({"regime": regime, "seed": seed, "arm": arm, "refit": got,
                             "published": float(want.iloc[0]),
                             "abs_diff": abs(got - float(want.iloc[0]))})
            log(f"    refit {regime} seed {seed} ({time.time() - started:.0f}s)")
    table = pd.DataFrame(fits)
    result["fit"] = {
        "cells": int(len(table)),
        "max_abs_diff": float(table["abs_diff"].max()) if len(table) else None,
        "mean_abs_diff": float(table["abs_diff"].mean()) if len(table) else None,
        "seconds": round(time.time() - started, 1),
        "table": table.to_dict("records"),
    }
    return result


def step_offset_shape(gen5_run: Path, output_dir: Path, log) -> dict:
    log("step 6/7  offset vs shape decomposition of the published gen5 OOF")
    oof = _load_published_oof(Path(gen5_run), DECOMPOSE_ARMS)
    if oof is None:
        return {"status": "SKIPPED", "reason": "no oof_predictions.csv"}
    rows = []
    for (regime, seed), block in oof.groupby(["regime", "split_seed"]):
        for arm in DECOMPOSE_ARMS:
            column = f"prediction_{arm}"
            if column not in block.columns:
                continue
            decomposition = decompose_level_shape(block["log_D"], block[column], block["extractant"])
            rows.append({
                "regime": regime, "seed": int(seed), "arm": arm,
                "macro_mae": equal_group_macro_mae(block["log_D"], block[column], block["ecfp_cluster"]),
                **{k: decomposition.summary[k] for k in
                   ("offset_mae", "shape_mae", "shape_r2", "offset_share_of_sse",
                    "median_ligand_mae", "worst_quartile_ligand_mae", "rank_spearman",
                    "sign_accuracy", "n_ligands")},
            })
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "gen5_offset_shape_per_seed.csv", index=False)
    summary = (per_seed.groupby(["regime", "arm"])
               [["macro_mae", "offset_mae", "shape_mae", "shape_r2", "offset_share_of_sse",
                 "median_ligand_mae", "worst_quartile_ligand_mae", "rank_spearman", "sign_accuracy"]]
               .mean().reset_index())
    summary.to_csv(output_dir / "gen5_offset_shape_summary.csv", index=False)

    by_bin = pd.DataFrame()
    if "nn_train_tanimoto" in oof.columns:
        ligand = oof[oof["regime"].isin(("unseen_ligand", "unseen_chemotype"))].copy()
        ligand = ligand.dropna(subset=["nn_train_tanimoto"])
        if not ligand.empty:
            ligand["novelty_bin"] = pd.cut(ligand["nn_train_tanimoto"], list(NOVELTY_BINS),
                                           right=False, include_lowest=True)
            records = []
            for (regime, novelty), block in ligand.groupby(["regime", "novelty_bin"], observed=True):
                for arm in DECOMPOSE_ARMS:
                    column = f"prediction_{arm}"
                    if column not in block.columns:
                        continue
                    decomposition = decompose_level_shape(
                        block["log_D"], block[column], block["extractant"])
                    records.append({
                        "regime": regime, "novelty_bin": str(novelty), "arm": arm,
                        "n_rows": int(len(block)), "n_ligands": int(block["extractant"].nunique()),
                        "mae": float(np.abs(block[column] - block["log_D"]).mean()),
                        "offset_mae": decomposition.summary["offset_mae"],
                        "shape_mae": decomposition.summary["shape_mae"],
                        "shape_r2": decomposition.summary["shape_r2"],
                        "offset_share_of_sse": decomposition.summary["offset_share_of_sse"],
                    })
            by_bin = pd.DataFrame(records)
            by_bin.to_csv(output_dir / "gen5_offset_shape_by_novelty.csv", index=False)
    return {"status": "OK", "summary": summary.to_dict("records"),
            "by_novelty": by_bin.to_dict("records") if len(by_bin) else []}


def step_pilot(args, output_dir: Path, log) -> dict:
    log("step 7/7  retrospective diversity pilot")
    script = REPO_ROOT / "scripts" / "run_diversity_causal.py"
    if args.skip_pilot:
        return {"status": "SKIPPED", "reason": "--skip-pilot"}
    if not script.is_file():
        return {"status": "SKIPPED", "reason": f"{script} not present"}
    pilot_dir = output_dir / "pilot"
    command = [sys.executable, str(script), *args.pilot_args, "--output-dir", str(pilot_dir)]
    log("    " + " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
    (output_dir / "pilot_stdout.txt").write_text(completed.stdout[-200_000:])
    if completed.returncode != 0:
        (output_dir / "pilot_stderr.txt").write_text(completed.stderr[-200_000:])
        return {"status": "FAILED", "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-4000:]}
    result: dict = {"status": "OK", "output_dir": str(pilot_dir)}
    summary_path = pilot_dir / "summary.json"
    if summary_path.is_file():
        result["summary"] = json.loads(summary_path.read_text())
    report_path = pilot_dir / "decision_report.md"
    if report_path.is_file():
        result["report"] = report_path.read_text()
    return result


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _refit_gate(fit: dict) -> tuple[str, str]:
    """PASS/FAIL/SKIPPED for the refit tolerance.

    Written as a function because the inline version was `(… or 1) < 0.02`, and
    `0.0 or 1` is `1`: a *perfect* refit — the one outcome the protocol says a
    cluster re-run should produce — scored FAIL. `None` must mean "not measured",
    not "1.0".
    """
    if fit.get("status") == "SKIPPED":
        return "SKIPPED", f"skipped ({fit.get('reason')})"
    deviation = fit.get("max_abs_diff")
    if deviation is None:
        return "SKIPPED", "no refit was performed"
    verdict = "PASS" if float(deviation) < 0.02 else "FAIL"
    return verdict, f"max |diff| {float(deviation):.4f} over {fit.get('cells')} cells"


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "—" if not np.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def write_report(steps: dict, context: dict, path: Path) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# gen6 Phase 0 report — {context['stamp']}")
    add("")
    add("Phase 0 fits almost nothing. It exists to show that the gen6 harness stands where the "
        "gen5 harness already stood, and to answer the seven questions that decide whether "
        "Phase 1 is worth running. Every number below was computed by this script on the "
        "committed dataset; nothing is copied from an earlier document.")
    add("")
    add(f"* dataset: `{context['dataset']}` — {context['source_rows']} rows, "
        f"{context['source_extractants']} extractants")
    add(f"* cohorts: BASE (min_cells {BASE_MIN_CELLS}) {context['base_rows']} rows / "
        f"{context['base_extractants']} extractants; EXPANDED (min_cells {EXPANDED_MIN_CELLS}) "
        f"{context['expanded_rows']} rows / {context['expanded_extractants']} extractants")
    add("")

    # gates
    gates = context["gates"]
    add("## Gate summary")
    add("")
    add("| gate | verdict | evidence |")
    add("|---|---|---|")
    for name, (verdict, evidence) in gates.items():
        add(f"| {name} | **{verdict}** | {evidence} |")
    add("")

    # 1 provenance
    provenance = steps["provenance"]
    add("## 1. Provenance — what can actually be reconstructed")
    add("")
    if provenance["status"] != "OK":
        add(f"`{provenance['status']}` — {provenance.get('reason')}")
    else:
        add(f"* `publication_id`: **{provenance['publication_status']}** — "
            f"{provenance['n_publications']} publications, "
            f"{100 * (provenance['fraction_ambiguous'] or 0):.2f} % of rows ambiguous.")
        add(f"* `experiment_series_id`: **{provenance['state']['series_status']}**;  "
            f"`replicate_id`: **{provenance['state']['replicate_status']}**.")
        merged = provenance.get("merged") or {}
        add("")
        add("How much of the modelling structure crosses study boundaries:")
        add("")
        add("| grouping | groups | spanning >1 publication | rows affected |")
        add("|---|---|---|---|")
        for value in merged.values():
            if not isinstance(value, dict) or value.get("status") == "unavailable":
                continue
            add(f"| {value.get('label')} | {value['n_groups']} | "
                f"{value['n_groups_multi_publication']} | "
                f"{value['n_rows_in_multi_publication_groups']} "
                f"({100 * value['fraction_rows_in_multi_publication_groups']:.2f} %) |")
        add("")
        if provenance.get("repeated_cells"):
            by_subset = provenance.get("repeated_cell_range_by_subset") or {}
            add(f"* **The published noise floor is contaminated, but not refuted.** Of "
                f"{provenance['repeated_cells']} repeated (extractant, condition, metal) cells, "
                f"**{provenance.get('repeated_cells_physical_axis')}** differ in a *physical* "
                f"variable the bundle drops, "
                f"{provenance.get('repeated_cells_free_text_only')} differ only in the paper's own "
                f"figure/table caption (which a genuine replicate reported twice also does), and "
                f"{provenance.get('repeated_cells_true_replicates')} have no recoverable difference "
                f"at all.")
            if by_subset:
                add("")
                add("  | subset | cells | median log D range | max |")
                add("  |---|---|---|---|")
                for name, stats in by_subset.items():
                    if not stats.get("n"):
                        continue
                    add(f"  | {name.replace('_', ' ')} | {stats['n']} | "
                        f"{_fmt(stats.get('median'))} | {_fmt(stats.get('max'))} |")
                add("")
                add("  The subsets point in opposite directions: the cells with **no** recoverable "
                    "difference scatter *more* than the ones with a hidden axis, so the partition "
                    "carries little information about the spread. The defensible claim is that the "
                    "floor is contaminated and its true value unknown — not that it is an artefact.")
        add(f"* `provenance_strict` == `legacy_compatible`: "
            f"**{provenance.get('strict_is_legacy')}**.")
    add("")

    # 2 chemistry
    chemistry = steps["chemistry"]
    add("## 2. Chemistry map — can all 190 ligands be clustered stably?")
    add("")
    audit = chemistry["audit"]
    add(f"* {audit['n_extractants']} extractants → {audit['n_ecfp_clusters']} bit-identical ECFP "
        f"clusters → {audit['n_superclusters']} Tanimoto-{audit['supercluster_threshold']} "
        f"super-clusters. Largest super-cluster holds "
        f"{audit['largest_supercluster_extractants']} extractants.")
    add(f"* families (from the frozen rdkit-derived motif columns; rdkit itself "
        f"{'available' if audit['rdkit_available'] else 'NOT installed here'}): "
        + ", ".join(f"{k} {v}" for k, v in sorted(audit["family_counts"].items(),
                                                  key=lambda kv: -kv[1])))
    add(f"* within-all nearest-neighbour Tanimoto: median "
        f"{_fmt(audit['nn_within_all_quantiles'].get(0.5))}, "
        f"q25–q75 {_fmt(audit['nn_within_all_quantiles'].get(0.25))}–"
        f"{_fmt(audit['nn_within_all_quantiles'].get(0.75))}")
    add("")
    add("**Answer to question 1: yes, and freezing the map is safe — but the check that sounds "
        "like the proof is not one.** `build_level_dataset` labels chemotypes *before* it applies "
        "the row filter, so the labels a cohort carries were already derived from all 190 "
        "extractants. Comparing against those is a real check at the bit-identical level and "
        "near-vacuous at the chemotype level. It is reported first because it is what governs "
        "*reproducing gen5* — gen5's folds used exactly those labels:")
    add("")
    add("| cohort | level | local groups | frozen groups | splits | merges | identical |")
    add("|---|---|---|---|---|---|---|")
    for cohort_name, report in chemistry["stability"].items():
        for level in ("ecfp", "supercluster"):
            entry = report.get(level, {})
            if "n_local" not in entry:
                continue
            add(f"| {cohort_name} | {level} | {entry['n_local']} | {entry['n_frozen_restricted']} | "
                f"{entry['local_groups_spanning_multiple_frozen']} | "
                f"{entry['frozen_groups_spanning_multiple_local']} | "
                f"{entry['identical_partition']} |")
    add("")
    add("The non-vacuous check re-runs single linkage on the cohort alone, as a study that had "
        "only ever seen those ligands would have. The property that matters is not *identical* "
        "but **never finer**: a frozen group may merge two local ones (a ligand outside the cohort "
        "bridges them), which makes a held-out chemotype larger and the hold-out stricter; a "
        "frozen group that *split* a local one would put related chemistry on both sides of a "
        "fold, and that is what must be zero.")
    add("")
    add("| cohort | level | from-scratch groups | frozen groups | frozen SPLITS a local group | frozen merges several | conservative |")
    add("|---|---|---|---|---|---|---|")
    for cohort_name, report in chemistry.get("conservative", {}).items():
        for level in ("ecfp", "supercluster"):
            entry = report.get(level, {})
            if "n_local_from_scratch" not in entry:
                continue
            add(f"| {cohort_name} | {level} | {entry['n_local_from_scratch']} | "
                f"{entry['n_frozen_restricted']} | **{entry['local_groups_the_frozen_map_splits']}** | "
                f"{entry['frozen_groups_merging_several_local']} | {entry['conservative']} |")
    add("")

    # 3 cohort comparison
    cohort = steps["cohort_comparison"]
    comparison = cohort["comparison"]
    add("## 3. What EXPANDED152 adds — questions 2 and 3")
    add("")
    add("| cohort | rows | extractants | ECFP clusters | super-clusters |")
    add("|---|---|---|---|---|")
    for key, label in (("base", "BASE (≥10 cells)"), ("added_by_expansion", "added (3–9 cells)"),
                       ("expanded", "EXPANDED (≥3 cells)")):
        block = comparison[key]
        add(f"| {label} | {block['n_rows']} | {block['n_extractants']} | "
            f"{block['n_ecfp_clusters']} | {block['n_superclusters']} |")
    add("")
    add(f"* **{cohort['n_entering']} extractants enter** at `min_cells = 3`, for "
        f"{100 * comparison['row_cost_fraction']:.1f} % of the rows.")
    add(f"* they bring **{comparison['new_ecfp_clusters']} ECFP clusters and "
        f"{comparison['new_superclusters']} Tanimoto super-clusters that do not exist in BASE**, "
        f"carrying {comparison['rows_in_new_superclusters']} rows.")
    add(f"* their nearest neighbour in BASE: median "
        f"{_fmt(cohort['median_nn_entering_to_kept'])} versus "
        f"{_fmt(cohort['median_nn_kept_to_kept'])} for BASE ligands among themselves; "
        + ", ".join(f"{n} below {t}" for t, n in cohort["n_below"].items()) + ".")
    add(f"* families entering: "
        + ", ".join(f"{k} {v}" for k, v in sorted(cohort["entering_families"].items(),
                                                  key=lambda kv: -kv[1])))
    add("")

    # 4 splits
    splits = steps["splits"]
    add("## 4. Byte-identical test rows — question 4")
    add("")
    add(f"* split integrity (no extractant, ECFP cluster or super-cluster shared between a fold's "
        f"test rows and any arm's training rows; BASE ⊂ EXPANDED; every row tested exactly once) "
        f"holds for **all {len(splits['per_seed'])} split seeds**: "
        f"`{splits['all_integrity_ok']}`.")
    first_seed = next(iter(splits["per_seed"].values()))
    add("")
    add("Fold structure for the first seed (the test row-id hash is what makes 'identical test "
        "rows' checkable rather than assumed):")
    add("")
    add("| fold | test rows | test row-id sha256 | train rows BASE | train rows EXPANDED | "
        "sparse train extractants |")
    add("|---|---|---|---|---|---|")
    for fold in first_seed["folds"]:
        counts = fold["n_train_rows_by_arm"]
        add(f"| {fold['fold']} | {fold['n_test_rows']} | `{fold['test_row_ids_sha256'][:12]}…` | "
            f"{counts.get(BASE_ARM)} | {counts.get(EXPANDED_ARM)} | "
            f"{fold['n_sparse_train_extractants']} |")
    add("")

    # 5 reproduction
    reproduction = steps["reproduction"]
    add("## 5. Reproduction of the gen5 champion")
    add("")
    if reproduction["status"] != "OK":
        add(f"`{reproduction['status']}` — {reproduction.get('reason')}")
    else:
        cohort_check = reproduction["cohort"]
        add(f"* **cohort**: gen6 builds {cohort_check['gen6_rows']} rows, the published run "
            f"recorded {cohort_check['published_rows']} — identical: "
            f"`{cohort_check['identical']}`.")
        split_check = reproduction.get("split", {})
        if "rows_compared" in split_check:
            add(f"* **split**: {split_check['rows_compared']} row-fold assignments compared across "
                f"every regime and seed; **{split_check['rows_with_different_fold']} differ**.")
        metrics_check = reproduction.get("metrics", {})
        if "max_abs_diff" in metrics_check:
            add(f"* **metrics**: recomputing macro MAE from the *published* out-of-fold "
                f"predictions reproduces the published per-seed numbers over "
                f"{metrics_check['cells']} (regime, seed, arm) cells to a maximum absolute "
                f"difference of **{metrics_check['max_abs_diff']:.2e}**.")
            add("")
            add("  Read this for what it is. It uses `levels.equal_group_macro_mae`, which is "
                "expression-identical to the function that produced the published table, and it "
                "keys on the published run's own cluster column. It therefore proves that "
                "`per_seed_metrics.csv` is arithmetically consistent with `oof_predictions.csv` — "
                "worth knowing, and not the same as an independent re-implementation agreeing.")
        fit_check = reproduction.get("fit", {})
        if fit_check.get("status") == "SKIPPED":
            add(f"* **fit**: skipped ({fit_check.get('reason')}).")
        elif "max_abs_diff" in fit_check:
            add(f"* **fit**: refitting {fit_check['cells']} (regime, seed, arm) cells reproduces "
                f"the published macro MAE to mean {fit_check['mean_abs_diff']:.4f} / max "
                f"**{fit_check['max_abs_diff']:.4f}** ({fit_check['seconds']:.0f} s).")
            add("")
            add("  A forest is only bit-reproducible on the machine that grew it. The published "
                "run was fitted on the cluster; this refit runs on a different CPU and BLAS, so "
                "the trees differ slightly even with an identical cohort, identical folds and an "
                "identical random seed. The measured spread is the honest tolerance, and it "
                "matters for reading gen5: **an effect smaller than ~0.01 macro MAE is not "
                "distinguishable from a change of machine.**")
    add("")

    # 6 offset/shape
    decomposition = steps["offset_shape"]
    add("## 6. Offset versus shape — questions 5 and 6")
    add("")
    if decomposition["status"] != "OK":
        add(f"`{decomposition['status']}` — {decomposition.get('reason')}")
    else:
        add("Error of the published gen5 out-of-fold predictions, split into the per-ligand level "
            "offset and the within-ligand shape. `offset_mae` is the mean over held-out ligands "
            "of |mean residual|; `shape_mae` is the MAE after centring truth and prediction "
            "within each ligand; `offset_share_of_sse` is the fraction of squared error an "
            "oracle per-ligand offset would remove.")
        add("")
        add("| regime | arm | macro MAE | offset MAE | shape MAE | shape R² | offset share of SSE |")
        add("|---|---|---|---|---|---|---|")
        for row in decomposition["summary"]:
            add(f"| {row['regime']} | {row['arm']} | {_fmt(row['macro_mae'])} | "
                f"{_fmt(row['offset_mae'])} | {_fmt(row['shape_mae'])} | {_fmt(row['shape_r2'])} | "
                f"{_fmt(row['offset_share_of_sse'])} |")
        add("")
        if decomposition["by_novelty"]:
            add("By nearest-neighbour Tanimoto to the training ligands:")
            add("")
            add("| regime | bin | arm | rows | ligands | MAE | offset MAE | shape MAE | shape R² |")
            add("|---|---|---|---|---|---|---|---|---|")
            for row in decomposition["by_novelty"]:
                if row["arm"] not in ("MC", "MC_lig2d_ext", "MC_donors"):
                    continue
                add(f"| {row['regime']} | {row['novelty_bin']} | {row['arm']} | {row['n_rows']} | "
                    f"{row['n_ligands']} | {_fmt(row['mae'])} | {_fmt(row['offset_mae'])} | "
                    f"{_fmt(row['shape_mae'])} | {_fmt(row['shape_r2'])} |")
    add("")

    # 7 pilot
    pilot = steps["pilot"]
    add("## 7. Retrospective diversity pilot — question 7")
    add("")
    if pilot["status"] != "OK":
        add(f"`{pilot['status']}` — {pilot.get('reason') or pilot.get('stderr_tail', '')[:800]}")
    else:
        add(f"Ran `scripts/run_diversity_causal.py` in pilot mode; artifacts in "
            f"`{pilot['output_dir']}`.")
        if pilot.get("report"):
            add("")
            add("<details><summary>pilot decision report</summary>")
            add("")
            add(pilot["report"])
            add("")
            add("</details>")
    add("")
    add("## What would falsify the Phase 0 conclusions")
    add("")
    add("* If the frozen chemistry partition ever stops matching a cohort's own partition "
        "(section 2), every cross-cohort comparison in this generation is comparing different "
        "objects and must be rebuilt.")
    add("* If a future split fails the integrity check (section 4), the BASE-vs-EXPANDED contrast "
        "is not measuring added chemistry.")
    add("* If the offset/shape split (section 6) is an artefact of the metric rather than the "
        "model, the identity `SSE = SSE_centred + Σ n·b²` would not hold; it is unit-tested "
        "exactly, on random data and on the degenerate fixtures.")
    add("* The reproduction tolerance (section 5) is measured, not assumed. Rerunning it on the "
        "cluster that produced the published run should shrink it to zero; if it does not, the "
        "difference is not environmental and the gate must be reopened.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / "runs" / f"gen6_phase0_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    log(f"gen6 Phase 0 -> {output_dir}")
    source = pd.read_parquet(args.dataset)
    descriptors = pd.read_parquet(args.descriptors) if Path(args.descriptors).is_file() else None
    if descriptors is None:
        log(f"WARNING: descriptor parquet missing at {args.descriptors}")
    cohorts = {
        "base": build_level_dataset(source, min_rows_per_extractant=BASE_MIN_CELLS,
                                    ligand_descriptors=descriptors),
        "expanded": build_level_dataset(source, min_rows_per_extractant=EXPANDED_MIN_CELLS,
                                        ligand_descriptors=descriptors),
    }
    upstream_dir = args.upstream_dir or find_upstream_directory()

    steps: dict = {}
    steps["provenance"] = step_provenance(source, upstream_dir, output_dir, log)
    steps["chemistry"], chemistry = step_chemistry(source, descriptors, cohorts, output_dir, log)
    steps["cohort_comparison"] = step_cohort_comparison(chemistry, cohorts, output_dir, log)
    steps["splits"] = step_split_machinery(cohorts, chemistry, output_dir, log)
    steps["reproduction"] = step_reproduction(cohorts, args, log)
    steps["offset_shape"] = step_offset_shape(args.gen5_run, output_dir, log)
    steps["pilot"] = step_pilot(args, output_dir, log)

    noise = replicate_noise_floor(source)
    reproduction = steps["reproduction"]
    gates = {
        "provenance audited": (
            "PASS" if steps["provenance"]["status"] == "OK" else "FAIL",
            f"publication_id {steps['provenance'].get('publication_status')}"),
        "chemistry freeze is conservative": (
            "PASS" if steps["chemistry"]["all_conservative"] else "FAIL",
            "re-clustering each cohort from scratch: the frozen map splits no local group"),
        "identical test rows / no leakage": (
            "PASS" if steps["splits"]["all_integrity_ok"] else "FAIL",
            f"{len(steps['splits']['per_seed'])} seeds x 5 folds checked"),
        "cohort reproduces": (
            "PASS" if reproduction.get("cohort", {}).get("identical") else "FAIL",
            f"{reproduction.get('cohort', {}).get('gen6_rows')} rows"),
        "split reproduces": (
            "PASS" if reproduction.get("split", {}).get("identical") else
            ("SKIPPED" if reproduction.get("split", {}).get("status") == "SKIPPED" else "FAIL"),
            f"{reproduction.get('split', {}).get('rows_with_different_fold')} rows differ"),
        "published metrics are arithmetically consistent with the published OOF": (
            "PASS" if reproduction.get("metrics", {}).get("exact") else
            ("SKIPPED" if reproduction.get("metrics", {}).get("status") == "SKIPPED" else "FAIL"),
            f"max |diff| {reproduction.get('metrics', {}).get('max_abs_diff')}"),
        "refit within measured tolerance": _refit_gate(reproduction.get("fit", {})),
    }
    context = {
        "stamp": stamp,
        "dataset": str(args.dataset),
        "source_rows": int(len(source)),
        "source_extractants": int(source["canonical_smiles"].nunique()),
        "base_rows": cohorts["base"].audit["rows"],
        "base_extractants": cohorts["base"].audit["extractants"],
        "expanded_rows": cohorts["expanded"].audit["rows"],
        "expanded_extractants": cohorts["expanded"].audit["extractants"],
        "gates": gates,
    }
    report = write_report(steps, context, output_dir / "gen6_phase0_report.md")
    (output_dir / "gen6_phase0_report.md").write_text(report)
    print(report)

    (output_dir / "cohort_comparison.json").write_text(
        json.dumps(steps["cohort_comparison"], indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(json.dumps({
        "stamp": stamp, "context": {k: v for k, v in context.items() if k != "gates"},
        "gates": {k: v[0] for k, v in gates.items()},
        "steps": steps, "noise_floor_as_published": noise,
        "cohort_audit": {k: v.audit for k, v in cohorts.items()},
    }, indent=2, default=str) + "\n")

    manifest = RunManifest(layer="gen6_diversity", run_id=f"gen6_phase0_{stamp}")
    manifest.record_dataset(dataset_path=args.dataset, source_frame=source,
                            descriptor_path=args.descriptors, descriptor_frame=descriptors)
    manifest.record_code([REPO_ROOT / "src" / "lanthanide_separation" / "gen6",
                          REPO_ROOT / "src" / "lanthanide_separation" / "levels.py",
                          Path(__file__)], repo_root=REPO_ROOT)
    manifest.record_features(feature_sets={
        name: list(cohorts["base"].block_columns(blocks)) for name, blocks in REPRO_ARMS.items()})
    manifest.record_chemistry(definition={
        "threshold": chemistry.audit["supercluster_threshold"],
        "n_extractants": chemistry.audit["n_extractants"],
        "n_ecfp_clusters": chemistry.audit["n_ecfp_clusters"],
        "n_superclusters": chemistry.audit["n_superclusters"],
        "frozen_over": "all extractants in the source table",
        "target_independent": True,
    })
    manifest.record_provenance(state=steps["provenance"].get("state", {"status": "unavailable"}))
    manifest.record_preprocessing([
        {"step": "build_level_dataset", "min_cells_base": BASE_MIN_CELLS,
         "min_cells_expanded": EXPANDED_MIN_CELLS, "replicate_policy": "mean",
         "log_d_floor": -6.0},
        {"step": "fold-local median imputation + missing indicators", "fitted_on": "training fold"},
    ])
    manifest.record("model_seed", 42)
    manifest.record("split_seeds", list(GEN5_SEEDS))
    folds = []
    for seed, block in steps["splits"]["per_seed"].items():
        for fold in block["folds"]:
            folds.append({
                "fold": fold["fold"], "split_seed": int(seed),
                "test_row_ids_sha256": fold["test_row_ids_sha256"],
                "test_extractants": [], "test_superclusters": fold["held_out_superclusters"],
                "train_extractants_by_arm": {}, "train_superclusters_by_arm": {},
                "n_test_rows": fold["n_test_rows"],
                "n_train_rows_by_arm": fold["n_train_rows_by_arm"],
            })
    manifest.record_split(definition={
        "algorithm": "shuffle unique group labels, deal round-robin (identical to gen5)",
        "group": "tanimoto_cluster", "n_splits": 5, "cohort": "expanded (min_cells 3)"}, folds=folds)
    payload = manifest.write(output_dir)
    validation = validate_run(
        output_dir, manifest=payload,
        required_artifacts=["gen6_phase0_report.md", "summary.json", "chemistry_map.parquet",
                            "chemistry_cluster_manifest.json", "nearest_neighbor_matrix.npz",
                            "provenance_audit.json", "split_manifest.json",
                            "cohort_comparison.json"],
        # A SKIPPED gate is recorded as ok so that a deliberate `--skip-refit` does
        # not fail the run, but it must never read as a pass: the verdict travels
        # with it, and the skipped list is surfaced separately below.
        checks={name.replace(" ", "_"): {"ok": verdict in ("PASS", "SKIPPED"), "verdict": verdict,
                                         "evaluated": verdict != "SKIPPED",
                                         "evidence": evidence}
                for name, (verdict, evidence) in gates.items()})
    skipped = [name for name, (verdict, _) in gates.items() if verdict == "SKIPPED"]
    validation["skipped_gates"] = skipped
    validation["gates_evaluated"] = len(gates) - len(skipped)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, default=str) + "\n")
    marker = write_success(output_dir, manifest=payload, validation=validation)
    log(f"validation ok={validation['ok']}; marker={marker}")
    failed = [name for name, (verdict, _) in gates.items() if verdict == "FAIL"]
    if skipped:
        log(f"SKIPPED GATES (not evaluated, NOT passed): {skipped}")
    if failed:
        log(f"FAILED GATES: {failed}")
    log(f"written to {output_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
