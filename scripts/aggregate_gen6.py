"""Combine several gen6 run directories — and refuse when they are not comparable.

Two earlier generations of this repo were bitten by the same failure: numbers
from runs that used different data, a different feature registry or a different
cohort ended up in one leaderboard, and the resulting ranking was an artifact of
the mixture rather than of the arms.  Nothing in the artifacts disclosed it,
because aggregation was a convenience script that concatenated CSVs.

This aggregator inverts that.  **Refusing to aggregate is the feature.**  Before
a single metric row is read it compares the provenance fingerprints of the runs:

===========================  ==================================================
fingerprint                  why a difference makes the runs incomparable
===========================  ==================================================
``dataset_file_sha256``      different source parquet: different rows entirely
``source_table_sha256``      same file, different content-canonical frame
``feature_registry_sha256``  an arm name means a different column list
chemistry-cluster            "super-cluster 7" is a different set of ligands, so
definition                   macro-by-cluster is a different metric
cohort definition            different eligibility rule / replicate policy /
                             ``log_D`` floor: different test population
===========================  ==================================================

A disagreement on any of those exits non-zero with the offending values printed.
There is no ``--force``: if two runs really should be compared despite a
difference, the difference has to be explained in a protocol amendment, not
suppressed by a flag.

What is *tolerated*, deliberately:

* **A missing artifact.**  A run may legitimately have no leaderboard (the
  chemistry-map run fits nothing) or no bootstrap table (a pilot).  Such a run
  still appears in the inventory with its provenance checked; it just
  contributes no metric rows.
* **A missing fingerprint.**  A run that never had a cohort cannot state a
  cohort definition.  Absence is reported per run and per key; only a
  *disagreement between two stated values* is fatal.  ``--strict-keys`` turns
  absence into a failure too, for the confirmation runs where every key should
  be present.

The aggregate itself writes the standard gen6 provenance set (manifest,
``validation.json``, ``artifact_hashes.json``, ``_SUCCESS.json``) so that an
aggregate can in turn be audited, and it works with a single run directory —
which is how Phase 0 calls it, to get one uniform provenance statement over a
run that has just been produced.

Example::

    .venv/bin/python scripts/aggregate_gen6.py runs/gen6_chemistry_20260819T084720Z
    .venv/bin/python scripts/aggregate_gen6.py runs/gen6_diversity_* --output-dir runs/gen6_aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.manifest import (  # noqa: E402
    RunManifest, sha256_json, validate_run, write_success,
)

#: Metric CSVs to look for, in preference order.  The first name that exists in a
#: run directory wins and the choice is recorded, so a reader can tell which file
#: a row came from.  Different gen6 runners name their leaderboard differently
#: (Experiment A scores arms, Experiment B scores policy x budget), and an
#: aggregator that hard-codes one name silently drops the other.
LEADERBOARD_CANDIDATES: tuple[str, ...] = (
    "leaderboard.csv", "arm_metrics.csv", "overall_metrics.csv",
    "per_arm_metrics.csv", "diversity_learning_curve.csv", "per_seed_metrics.csv",
)
BOOTSTRAP_CANDIDATES: tuple[str, ...] = (
    # Experiment A writes `contrasts.csv` (per split seed) and `contrast_summary.csv`
    # (pooled over seeds); the pooled one is what an aggregate should combine.  These
    # were missing from the list, so aggregating a real Experiment A run silently
    # reported "no bootstrap CSV found" instead of its contrasts.
    "contrast_summary.csv", "contrasts.csv",
    "paired_bootstrap.csv", "bootstrap.csv", "paired_unit_bootstrap.csv",
    "chemistry_bootstrap.csv", "diversity_bootstrap.csv", "learning_curve_bootstrap.csv",
)

#: Columns that identify *what was scored* in a leaderboard row.  The first one
#: present is used as the arm axis unless ``--arm-column`` overrides it: Experiment
#: A scores ``arm``, Experiment B scores ``policy``, a feature ablation scores
#: ``feature_set``.
ARM_COLUMN_CANDIDATES: tuple[str, ...] = ("arm", "policy", "feature_set", "regime")
#: Columns that split one arm into several genuinely different measurements.  They
#: must all enter the grouping, or the aggregate averages an "overall" endpoint
#: with a "hard chemistry" one and reports a number that describes neither — the
#: exact silent-mixing failure this script exists to prevent.
FACET_COLUMNS: tuple[str, ...] = (
    "endpoint", "regime", "budget", "policy", "feature_set", "weighting", "arm",
)

#: Manifest sub-keys that, taken together, define the cohort a run scored on.
#: Used only when a run does not record an explicit ``cohort_definition``.
COHORT_SUBKEYS: tuple[str, ...] = (
    "min_cells", "eval_min_cells", "min_rows_per_extractant", "base_min_cells",
    "shared_cohort_min_cells", "replicate_policy", "drop_below_log_d", "log_d_floor",
    "cohort_rows", "cohort_extractants",
)

#: Artifacts the aggregate must have written before it may claim success.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "runs_inventory.csv", "compatibility.json", "aggregate_decision_report.md",
    "summary.json", "manifest.json",
)

CODE_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve(),
    REPO_ROOT / "src" / "lanthanide_separation" / "gen6",
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=Path,
                   help="gen6 run directories (one is enough; Phase 0 calls it with one)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--strict-keys", action="store_true",
                   help="treat a MISSING provenance fingerprint as a failure, not just a "
                        "disagreement between two stated ones")
    p.add_argument("--require-success", action="store_true",
                   help="refuse a run whose _SUCCESS.json is absent or whose validation failed")
    p.add_argument("--leaderboard-name", default=None,
                   help=f"override the leaderboard CSV name (default: first of {LEADERBOARD_CANDIDATES})")
    p.add_argument("--bootstrap-name", default=None,
                   help=f"override the bootstrap CSV name (default: first of {BOOTSTRAP_CANDIDATES})")
    p.add_argument("--rank-metric", default="macro_mae",
                   help="numeric column used to rank arms in the combined leaderboard "
                        "(lower is better); ignored when absent")
    p.add_argument("--arm-column", default=None,
                   help="column identifying the arm in the leaderboard CSVs; by default the first "
                        f"of {ARM_COLUMN_CANDIDATES} that is present")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Reading one run
# --------------------------------------------------------------------------- #

@dataclass
class LoadedRun:
    """One gen6 run directory, read defensively.

    ``problems`` collects everything that was wrong but survivable (a missing
    optional CSV, an unreadable ``summary.json``).  Fatal disagreements are not
    stored here — they are decided across runs, in :func:`compatibility_report`.
    """

    path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    success: dict[str, Any] | None = None
    leaderboard: pd.DataFrame | None = None
    leaderboard_source: str | None = None
    bootstrap: pd.DataFrame | None = None
    bootstrap_source: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id") or self.path.name)

    @property
    def layer(self) -> str:
        return str(self.manifest.get("layer") or "unknown")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"_non_object_payload": payload}


def _read_first_csv(directory: Path, candidates: Sequence[str],
                    override: str | None) -> tuple[pd.DataFrame | None, str | None]:
    names = [override] if override else list(candidates)
    for name in names:
        path = directory / name
        if path.is_file() and path.stat().st_size > 0:
            return pd.read_csv(path), name
    return None, None


def load_run(path: Path, *, leaderboard_name: str | None, bootstrap_name: str | None) -> LoadedRun:
    """Read a run directory.  Raises only when the directory is not a gen6 run at all."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    manifest = _read_json(directory / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(
            f"{directory} has no manifest.json — it is not a gen6 run directory. "
            "Aggregation is defined on the manifest, not on the CSVs, precisely so that "
            "incomparable runs cannot be combined.")
    run = LoadedRun(path=directory, manifest=manifest)
    run.summary = _read_json(directory / "summary.json")
    run.validation = _read_json(directory / "validation.json")
    run.success = _read_json(directory / "_SUCCESS.json")
    if run.summary is None:
        run.problems.append("no summary.json")
    if run.validation is None:
        run.problems.append("no validation.json")
    if run.success is None:
        run.problems.append("no _SUCCESS.json (the run did not validate, or predates the marker)")
    run.leaderboard, run.leaderboard_source = _read_first_csv(
        directory, LEADERBOARD_CANDIDATES, leaderboard_name)
    if run.leaderboard is None:
        run.problems.append("no leaderboard CSV")
    run.bootstrap, run.bootstrap_source = _read_first_csv(
        directory, BOOTSTRAP_CANDIDATES, bootstrap_name)
    if run.bootstrap is None:
        run.problems.append("no bootstrap CSV")
    return run


# --------------------------------------------------------------------------- #
# Compatibility — the part that fails closed
# --------------------------------------------------------------------------- #

def _plain(manifest: Mapping[str, Any], key: str) -> tuple[Any, str]:
    value = manifest.get(key)
    return (value, key) if value is not None else (None, "absent")


def _chemistry_fingerprint(manifest: Mapping[str, Any]) -> tuple[Any, str]:
    """Hash of the chemistry-cluster definition.

    Hashed rather than compared field-by-field because the definition is a nested
    object; the report prints the raw definitions of any two runs that disagree,
    so the hash never has to be interpreted on its own.
    """
    definition = manifest.get("chemistry_cluster_definition")
    if not definition:
        return None, "absent"
    return sha256_json(definition), "chemistry_cluster_definition"


def _cohort_fingerprint(manifest: Mapping[str, Any]) -> tuple[Any, str]:
    """Hash of whatever the run says its cohort was.

    Prefers an explicit ``cohort_definition``.  Failing that it reconstructs one
    from the cohort-shaped sub-keys of ``split_definition`` / ``preprocessing``,
    because older-shaped runners record the eligibility rule there.  A run with
    no cohort at all (the chemistry-map run) reports ``absent``, which is
    tolerated unless ``--strict-keys`` is given.
    """
    explicit = manifest.get("cohort_definition")
    if explicit:
        return sha256_json(explicit), "cohort_definition"
    harvested: dict[str, Any] = {}
    for container_key in ("split_definition", "preprocessing"):
        container = manifest.get(container_key)
        blocks: list[Mapping[str, Any]] = []
        if isinstance(container, Mapping):
            blocks = [container]
        elif isinstance(container, list):
            blocks = [b for b in container if isinstance(b, Mapping)]
        for block in blocks:
            for key in COHORT_SUBKEYS:
                if key in block and key not in harvested:
                    harvested[key] = block[key]
    if harvested:
        return sha256_json(harvested), "derived from split_definition/preprocessing"
    return None, "absent"


#: ``(label, extractor)``.  Every extractor returns ``(value_or_None, source)``.
COMPATIBILITY_KEYS: tuple[tuple[str, Callable[[Mapping[str, Any]], tuple[Any, str]]], ...] = (
    ("dataset_file_sha256", lambda m: _plain(m, "dataset_file_sha256")),
    ("source_table_sha256", lambda m: _plain(m, "source_table_sha256")),
    ("feature_registry_sha256", lambda m: _plain(m, "feature_registry_sha256")),
    ("chemistry_cluster_definition", _chemistry_fingerprint),
    ("cohort_definition", _cohort_fingerprint),
)


def compatibility_report(runs: Sequence[LoadedRun], *, strict_keys: bool) -> dict[str, Any]:
    """Compare the provenance fingerprints of every run.  ``ok`` is the gate."""
    report: dict[str, Any] = {"n_runs": len(runs), "strict_keys": bool(strict_keys),
                              "keys": {}, "fatal": [], "warnings": [], "ok": True}
    for label, extractor in COMPATIBILITY_KEYS:
        per_run: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for run in runs:
            value, source = extractor(run.manifest)
            per_run[run.run_id] = value
            sources[run.run_id] = source
        stated = {run_id: value for run_id, value in per_run.items() if value is not None}
        absent = [run_id for run_id, value in per_run.items() if value is None]
        distinct = sorted({json.dumps(v, sort_keys=True, default=str) for v in stated.values()})
        entry = {
            "per_run": per_run,
            "source_per_run": sources,
            "n_stated": len(stated),
            "n_absent": len(absent),
            "absent_in": absent,
            "n_distinct_values": len(distinct),
            "agrees": len(distinct) <= 1,
        }
        if len(distinct) > 1:
            entry["status"] = "MISMATCH"
            report["ok"] = False
            report["fatal"].append(
                f"{label}: {len(distinct)} different values across {len(stated)} runs -> "
                + "; ".join(f"{run_id}={str(value)[:24]}" for run_id, value in sorted(stated.items())))
        elif absent and strict_keys:
            entry["status"] = "ABSENT (strict)"
            report["ok"] = False
            report["fatal"].append(f"{label}: absent in {absent} and --strict-keys was given")
        elif absent:
            entry["status"] = "ABSENT (tolerated)"
            report["warnings"].append(
                f"{label}: not stated by {absent}; those runs are inventoried but their "
                "comparability on this key is unproven")
        else:
            entry["status"] = "AGREES"
        report["keys"][label] = entry
    return report


def success_gate(runs: Sequence[LoadedRun], *, require_success: bool) -> dict[str, Any]:
    """Optional gate on each input run's own validation verdict."""
    failed = [run.run_id for run in runs
              if run.success is None or not bool((run.validation or {}).get("ok", run.success is not None))]
    return {
        "required": bool(require_success),
        "runs_without_validated_success": failed,
        "ok": (not failed) if require_success else True,
    }


# --------------------------------------------------------------------------- #
# Combining
# --------------------------------------------------------------------------- #

def inventory_frame(runs: Sequence[LoadedRun]) -> pd.DataFrame:
    """One row per input run: what it is, whether it validated, what it contributed."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        validation = run.validation or {}
        rows.append({
            "run_id": run.run_id,
            "layer": run.layer,
            "path": str(run.path),
            "created_utc": run.manifest.get("created_utc"),
            "validated_ok": validation.get("ok"),
            "has_success_marker": run.success is not None,
            "dataset_file_sha256": str(run.manifest.get("dataset_file_sha256") or "")[:16],
            "feature_registry_sha256": str(run.manifest.get("feature_registry_sha256") or "")[:16],
            "n_folds": len(run.manifest.get("folds") or []),
            "leaderboard_file": run.leaderboard_source,
            "leaderboard_rows": 0 if run.leaderboard is None else int(len(run.leaderboard)),
            "bootstrap_file": run.bootstrap_source,
            "bootstrap_rows": 0 if run.bootstrap is None else int(len(run.bootstrap)),
            "problems": "; ".join(run.problems),
        })
    return pd.DataFrame(rows)


def _stamp(frame: pd.DataFrame, run: LoadedRun, source: str | None) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "source_file", source)
    out.insert(0, "run_path", str(run.path))
    out.insert(0, "run_id", run.run_id)
    return out


def combined_leaderboard(runs: Sequence[LoadedRun]) -> pd.DataFrame:
    """Every leaderboard row from every run, stamped with its provenance."""
    parts = [_stamp(run.leaderboard, run, run.leaderboard_source)
             for run in runs if run.leaderboard is not None and not run.leaderboard.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def combined_bootstrap(runs: Sequence[LoadedRun]) -> pd.DataFrame:
    """Every bootstrap row from every run, stamped with its provenance."""
    parts = [_stamp(run.bootstrap, run, run.bootstrap_source)
             for run in runs if run.bootstrap is not None and not run.bootstrap.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def resolve_arm_column(combined: pd.DataFrame, override: str | None) -> str | None:
    """Which column names the thing being scored, given what the CSVs actually have."""
    if override:
        return override if override in combined.columns else None
    return next((c for c in ARM_COLUMN_CANDIDATES if c in combined.columns), None)


def leaderboard_by_arm(combined: pd.DataFrame, *, arm_column: str | None,
                       rank_metric: str) -> pd.DataFrame:
    """Across-run view of one metric per arm.

    Deliberately reports ``n_runs`` and the spread, never a single pooled number:
    runs differ in seeds and regimes even when they are provenance-compatible, so
    the mean of a metric over runs is a summary, not an estimate.  Ranking is by
    the mean, and the sd column is what tells you whether the ranking survives.

    Every facet column present (endpoint, regime, budget, weighting…) enters the
    grouping.  Collapsing across a facet would average measurements that are not
    the same measurement.
    """
    if (combined.empty or not arm_column or arm_column not in combined.columns
            or rank_metric not in combined.columns):
        return pd.DataFrame()
    values = pd.to_numeric(combined[rank_metric], errors="coerce")
    work = combined.assign(_metric=values).dropna(subset=["_metric"])
    if work.empty:
        return pd.DataFrame()
    group_columns = [arm_column] + [c for c in FACET_COLUMNS
                                    if c in work.columns and c != arm_column]
    grouped = (work.groupby(group_columns, dropna=False)
               .agg(**{f"{rank_metric}_mean": ("_metric", "mean"),
                       f"{rank_metric}_sd": ("_metric", "std"),
                       f"{rank_metric}_min": ("_metric", "min"),
                       f"{rank_metric}_max": ("_metric", "max"),
                       "n_rows": ("_metric", "size"),
                       "n_runs": ("run_id", "nunique")})
               .reset_index()
               .sort_values(f"{rank_metric}_mean", ignore_index=True))
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped


def bootstrap_consistency(combined: pd.DataFrame) -> pd.DataFrame:
    """Does a contrast point the same way in every run, and does its CI exclude zero?

    The gen6 protocol's pass conditions are stated as "CI95 low > 0 in the paired
    chemistry-unit bootstrap"; when several runs supply the same contrast, the
    honest aggregate is *how many of them clear it*, not a re-pooled interval.
    Re-pooling would need the replicate draws, which the CSVs do not carry.
    """
    if combined.empty:
        return pd.DataFrame()
    # Experiment A's pooled table prefixes its columns with `pooled_`; accept both
    # spellings rather than silently reporting "no recognised delta/CI columns".
    delta_columns = [c for c in ("point_delta", "pooled_point_delta", "point_delta_mae", "delta")
                     if c in combined.columns]
    low_columns = [c for c in ("ci95_low", "pooled_ci95_low") if c in combined.columns]
    high_columns = [c for c in ("ci95_high", "pooled_ci95_high") if c in combined.columns]
    if not delta_columns or not low_columns:
        return pd.DataFrame()
    delta = delta_columns[0]
    combined = combined.rename(columns={low_columns[0]: "ci95_low",
                                        **({high_columns[0]: "ci95_high"} if high_columns else {})})
    keys = [c for c in ("comparison", "reference", "candidate", "statistic", "endpoint",
                        "regime", "feature_set", "weighting", "budget")
            if c in combined.columns]
    if not keys:
        return pd.DataFrame()
    work = combined.copy()
    work[delta] = pd.to_numeric(work[delta], errors="coerce")
    work["ci95_low"] = pd.to_numeric(work["ci95_low"], errors="coerce")
    if "ci95_high" in work.columns:
        work["ci95_high"] = pd.to_numeric(work["ci95_high"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for key_values, block in work.groupby(keys, dropna=False, sort=True):
        values = block[delta].dropna()
        record = dict(zip(keys, key_values if isinstance(key_values, tuple) else (key_values,)))
        record.update({
            "n_runs": int(block["run_id"].nunique()),
            "n_rows": int(len(block)),
            "mean_delta": float(values.mean()) if len(values) else np.nan,
            "min_delta": float(values.min()) if len(values) else np.nan,
            "max_delta": float(values.max()) if len(values) else np.nan,
            "n_delta_positive": int((values > 0).sum()),
            "n_ci95_low_above_zero": int((block["ci95_low"] > 0).sum()),
            "n_with_ci": int(block["ci95_low"].notna().sum()),
        })
        record["direction_consistent"] = bool(
            len(values) > 0 and (record["n_delta_positive"] in (0, len(values))))
        rows.append(record)
    return pd.DataFrame(rows).sort_values("mean_delta", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def markdown_table(frame: pd.DataFrame, *, floatfmt: str = "{:.4g}", max_rows: int = 200) -> str:
    """GitHub pipe table (``tabulate`` is not a dependency of this repo)."""
    if frame is None or frame.empty:
        return "_(no rows)_"
    shown = frame.head(max_rows)

    def cell(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, (float, np.floating)):
            return "—" if not np.isfinite(value) else floatfmt.format(float(value))
        if isinstance(value, (bool, np.bool_)):
            return "yes" if bool(value) else "**no**"
        return str(value)

    header = [str(c) for c in shown.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row.tolist()) + " |")
    if len(frame) > max_rows:
        lines.append(f"| … {len(frame) - max_rows} further rows in the CSV | " +
                     " | ".join("" for _ in header[1:]) + " |")
    return "\n".join(lines)


def render_report(
    *, stamp: str, runs: Sequence[LoadedRun], compatibility: dict[str, Any],
    success: dict[str, Any], inventory: pd.DataFrame, leaderboard: pd.DataFrame,
    by_arm: pd.DataFrame, bootstrap: pd.DataFrame, consistency: pd.DataFrame,
    rank_metric: str, arm_column: str | None,
) -> str:
    out: list[str] = []
    A = out.append
    A(f"# gen6 aggregate — {stamp}")
    A("")
    A(f"{len(runs)} run director{'y' if len(runs) == 1 else 'ies'} aggregated. "
      "An aggregate is only as meaningful as the claim that its inputs are comparable, so that "
      "claim is checked first and printed first.")
    A("")

    A("## 1. Provenance compatibility (the gate)")
    A("")
    rows = []
    for label, entry in compatibility["keys"].items():
        rows.append({
            "fingerprint": label,
            "status": entry["status"],
            "runs stating it": entry["n_stated"],
            "distinct values": entry["n_distinct_values"],
            "absent in": ", ".join(entry["absent_in"]) or "—",
        })
    A(markdown_table(pd.DataFrame(rows)))
    A("")
    if compatibility["fatal"]:
        A("**FATAL — these runs are not comparable:**")
        A("")
        for message in compatibility["fatal"]:
            A(f"* {message}")
        A("")
        A("Nothing below should be read as a comparison. The aggregate exits non-zero.")
    elif len(runs) == 1:
        A("Single input run, so there is nothing to disagree with. This section records **what** the "
          "run was built from, so a later aggregate can be checked against it.")
    else:
        A("No fingerprint disagreement. Every value that two runs both state is identical.")
    A("")
    for message in compatibility["warnings"]:
        A(f"* WARNING: {message}")
    if compatibility["warnings"]:
        A("")
        A("An absent fingerprint is tolerated (a run that fits nothing has no cohort), but it is "
          "an *unproven* compatibility, not a verified one. Re-run with `--strict-keys` where every "
          "input is expected to state every key.")
        A("")
    if success["required"]:
        A(f"`--require-success` gate: runs without a validated success marker = "
          f"{success['runs_without_validated_success'] or 'none'}.")
        A("")

    A("## 2. Runs")
    A("")
    A(markdown_table(inventory))
    A("")
    A("`problems` lists tolerated absences. A run with no leaderboard contributes provenance only.")
    A("")

    A("## 3. Combined leaderboard")
    A("")
    if leaderboard.empty:
        A("_No leaderboard CSV was found in any input run._ For a chemistry-map or provenance run "
          "this is expected: the aggregate is then a compatibility statement, not a comparison.")
    else:
        A(f"{len(leaderboard)} rows from "
          f"{leaderboard['run_id'].nunique()} run(s); full table in `combined_leaderboard.csv`.")
        A("")
        if by_arm.empty:
            A(f"_No across-run ranking was computed: arm column = `{arm_column or 'not found'}`, "
              f"rank metric = `{rank_metric}` "
              f"({'absent' if rank_metric not in leaderboard.columns else 'present but non-numeric'}). "
              "Pass `--arm-column` / `--rank-metric` with columns these leaderboards have._")
        else:
            A(f"Scored unit: `{arm_column}`; ranked by mean `{rank_metric}` (lower is better), grouped "
              "by every facet column present so that an 'overall' row is never averaged with a "
              "'hard chemistry' one. `n_runs` and the sd are part of the result: a ranking whose gaps "
              "are smaller than its spread is not a ranking.")
            A("")
            A(markdown_table(by_arm))
    A("")

    A("## 4. Bootstrap contrasts")
    A("")
    if bootstrap.empty:
        A("_No bootstrap CSV was found in any input run._")
    else:
        A(f"{len(bootstrap)} rows from {bootstrap['run_id'].nunique()} run(s); full table in "
          "`combined_bootstrap.csv`.")
        A("")
        if consistency.empty:
            A("_The bootstrap tables carry no recognised delta/CI columns, so no consistency view "
              "was computed._")
        else:
            A("Per contrast: how many runs put the delta above zero, and how many have a CI95 lower "
              "bound above zero. Intervals are **not** re-pooled — that would need the replicate "
              "draws, which the CSVs do not carry — so this counts runs, it does not combine them.")
            A("")
            A(markdown_table(consistency))
    A("")

    A("## 5. What would falsify the aggregate")
    A("")
    A("* Any of the section 1 fingerprints differing between two inputs. That is checked, and fatal.")
    A("* An arm whose rank flips between runs while its across-run sd exceeds the gap to its "
      "neighbour: the ranking is then noise, whatever the mean says.")
    A("* A contrast whose CI95 low is above zero in some runs and below in others "
      "(`n_ci95_low_above_zero` strictly between 0 and `n_runs`): the effect is not established.")
    A("* An input run whose `validated_ok` is not `yes`: its numbers were produced by a run that "
      "could not prove its own artifacts. Use `--require-success` to make that fatal.")
    A("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _fail(message: str, log: Callable[[str], None]) -> int:
    log("AGGREGATION REFUSED")
    for line in message.splitlines():
        log(f"  {line}")
    print(f"\naggregate_gen6: REFUSED\n{message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "runs" / f"gen6_aggregate_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    log(f"gen6 aggregate of {len(args.runs)} run director"
        f"{'y' if len(args.runs) == 1 else 'ies'} -> {output_dir}")

    runs: list[LoadedRun] = []
    for path in args.runs:
        try:
            run = load_run(path, leaderboard_name=args.leaderboard_name,
                           bootstrap_name=args.bootstrap_name)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            return _fail(f"cannot read run directory {path}: {error}", log)
        runs.append(run)
        log(f"  {run.run_id}: layer={run.layer} folds={len(run.manifest.get('folds') or [])} "
            f"leaderboard={run.leaderboard_source} bootstrap={run.bootstrap_source}"
            + (f" problems=[{'; '.join(run.problems)}]" if run.problems else ""))

    duplicate_ids = sorted({r.run_id for r in runs if [x.run_id for x in runs].count(r.run_id) > 1})
    if duplicate_ids:
        # Two directories claiming the same run_id would make every "n_runs" count a lie.
        return _fail(f"two input directories share a run_id {duplicate_ids}; "
                     "aggregating them would double-count their rows", log)

    compatibility = compatibility_report(runs, strict_keys=args.strict_keys)
    gate = success_gate(runs, require_success=args.require_success)
    (output_dir / "compatibility.json").write_text(
        json.dumps({"compatibility": compatibility, "success_gate": gate}, indent=2, default=str) + "\n")

    inventory = inventory_frame(runs)
    inventory.to_csv(output_dir / "runs_inventory.csv", index=False)

    leaderboard = combined_leaderboard(runs)
    bootstrap = combined_bootstrap(runs)
    arm_column = resolve_arm_column(leaderboard, args.arm_column)
    by_arm = leaderboard_by_arm(leaderboard, arm_column=arm_column, rank_metric=args.rank_metric)
    consistency = bootstrap_consistency(bootstrap)
    if not leaderboard.empty:
        leaderboard.to_csv(output_dir / "combined_leaderboard.csv", index=False)
    if not by_arm.empty:
        by_arm.to_csv(output_dir / "leaderboard_by_arm.csv", index=False)
    if not bootstrap.empty:
        bootstrap.to_csv(output_dir / "combined_bootstrap.csv", index=False)
    if not consistency.empty:
        consistency.to_csv(output_dir / "bootstrap_consistency.csv", index=False)
    log(f"  combined: {len(leaderboard)} leaderboard rows, {len(bootstrap)} bootstrap rows")

    report = render_report(
        stamp=stamp, runs=runs, compatibility=compatibility, success=gate, inventory=inventory,
        leaderboard=leaderboard, by_arm=by_arm, bootstrap=bootstrap, consistency=consistency,
        rank_metric=args.rank_metric, arm_column=arm_column)
    (output_dir / "aggregate_decision_report.md").write_text(report)

    summary = {
        "stamp": stamp,
        "layer": "gen6",
        "kind": "aggregate",
        "n_runs": len(runs),
        "run_ids": [run.run_id for run in runs],
        "run_paths": [str(run.path) for run in runs],
        "compatibility": compatibility,
        "success_gate": gate,
        "n_leaderboard_rows": int(len(leaderboard)),
        "n_bootstrap_rows": int(len(bootstrap)),
        "rank_metric": args.rank_metric,
        "arm_column": arm_column,
        "leaderboard_by_arm": by_arm.to_dict("records") if not by_arm.empty else [],
        "bootstrap_consistency": consistency.to_dict("records") if not consistency.empty else [],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    # The aggregate inherits the provenance of its inputs where they agree, and
    # says "DISAGREEMENT"/"ABSENT" where they do not, so that an aggregate of
    # aggregates cannot launder a mismatch into a single clean-looking hash.
    def inherited(label: str) -> Any:
        entry = compatibility["keys"][label]
        stated = [v for v in entry["per_run"].values() if v is not None]
        if not stated:
            return None
        return stated[0] if entry["agrees"] else "DISAGREEMENT"

    manifest = RunManifest(layer="gen6", run_id=f"gen6_aggregate_{stamp}")
    manifest.record("dataset_path", [run.manifest.get("dataset_path") for run in runs])
    manifest.record("dataset_file_sha256", inherited("dataset_file_sha256"))
    manifest.record("source_table_sha256", inherited("source_table_sha256"))
    manifest.record("feature_registry_sha256", inherited("feature_registry_sha256"))
    manifest.record_code(CODE_PATHS, repo_root=REPO_ROOT)
    manifest.record_chemistry(definition={
        "inherited_from_runs": [run.run_id for run in runs],
        "agrees_across_runs": compatibility["keys"]["chemistry_cluster_definition"]["agrees"],
        "definition": next((run.manifest.get("chemistry_cluster_definition") for run in runs
                            if run.manifest.get("chemistry_cluster_definition")), None),
    })
    manifest.record_provenance(state={
        "inherited_from_runs": [run.manifest.get("provenance_state") for run in runs],
    })
    manifest.record_split(definition={
        "kind": "aggregate",
        "reason": "an aggregate draws no split of its own; it inherits its inputs'",
        "input_split_sha256": {run.run_id: run.manifest.get("split_sha256") for run in runs},
        "input_fold_counts": {run.run_id: len(run.manifest.get("folds") or []) for run in runs},
    }, folds=[])
    manifest.record_preprocessing([
        {"step": "load run manifests", "runs": [str(run.path) for run in runs]},
        {"step": "compatibility gate", "keys": [label for label, _ in COMPATIBILITY_KEYS],
         "strict_keys": bool(args.strict_keys)},
        {"step": "concatenate leaderboard / bootstrap CSVs",
         "leaderboard_files": {run.run_id: run.leaderboard_source for run in runs},
         "bootstrap_files": {run.run_id: run.bootstrap_source for run in runs}},
    ])
    manifest.record("model_seed", [run.manifest.get("model_seed") for run in runs])
    manifest.record("split_seeds", [run.manifest.get("split_seeds") for run in runs])
    manifest.record("cohort_definition", {
        "agrees_across_runs": compatibility["keys"]["cohort_definition"]["agrees"],
        "per_run": {run.run_id: run.manifest.get("cohort_definition") for run in runs},
    })
    manifest.record("input_runs", [{"run_id": run.run_id, "path": str(run.path),
                                    "validated_ok": (run.validation or {}).get("ok")} for run in runs])
    payload = manifest.write(output_dir)

    checks = {
        "provenance_compatible": {"ok": compatibility["ok"], "fatal": compatibility["fatal"],
                                  "warnings": compatibility["warnings"]},
        "input_success_gate": gate,
        "every_run_has_a_manifest": {"ok": True, "n_runs": len(runs)},
    }
    validation = validate_run(output_dir, manifest=payload,
                              required_artifacts=REQUIRED_ARTIFACTS, checks=checks)
    write_success(output_dir, manifest=payload, validation=validation)

    print()
    print(report)
    if not compatibility["ok"]:
        return _fail("\n".join(compatibility["fatal"])
                     + f"\n(inventory and compatibility.json were still written to {output_dir})", log)
    if not validation["ok"]:
        return _fail(f"the aggregate failed its own validation: {validation['failed_checks']} "
                     f"missing={validation['missing_artifacts']}", log)
    log(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
