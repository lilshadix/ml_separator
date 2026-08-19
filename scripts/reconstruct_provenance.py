"""gen6: reconstruct experimental provenance, or prove that it cannot be reconstructed.

Five generations of this project shipped the same caveat — "condition-matched rows
may silently combine measurements from unrelated studies" — without a number
beside it.  This script produces the number, or an explicit *unavailable*.

What it does:

1. joins the bundle onto the upstream SAFE exports (``--upstream-dir``) when they
   are reachable, recovering ``DOI`` / ``entry_author`` / ``addition_date`` and the
   experimental columns the bundle drops;
2. forms ``publication_id`` / ``experiment_series_id`` / ``experiment_id`` /
   ``replicate_id``, each labelled ``reconstructed`` / ``surrogate`` / ``ambiguous``
   / ``unavailable`` — never guessed;
3. measures how much of the modelling structure crosses study boundaries: the
   level cell that gen5 averages, the series it holds out, the pair key, the
   ligand;
4. tests whether the "replicate" cells behind the published noise floor are
   replicates at all;
5. writes a frozen ``provenance_table.parquet`` so a later run — on a cluster
   without the upstream checkout — can consume the reconstruction by path instead
   of re-deriving it.

Running it without the upstream tables is a *successful* run: it reports
``unavailable`` with the reason. Only a real error exits non-zero.

Example::

    .venv/bin/python scripts/reconstruct_provenance.py
    .venv/bin/python scripts/reconstruct_provenance.py --upstream-dir /path/to/raw_data
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.manifest import (  # noqa: E402
    RunManifest, validate_run, write_success,
)
from lanthanide_separation.gen6.provenance import (  # noqa: E402
    find_upstream_directory, provenance_audit_report, reconstruct_provenance,
)

DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
#: Columns written to the frozen table.  Deliberately excludes the raw upstream
#: experimental columns: they are *candidate features* for a later phase and must
#: not leak into a run by being sitting in a provenance artifact that some future
#: script joins wholesale.  Pass --include-upstream-columns to keep them.
CORE_COLUMNS: tuple[str, ...] = (
    "safe_exp_id", "extractant", "metal_symbol", "condition_id", "cell_id", "strict_cell_id",
    "experiment_id", "experiment_id_status",
    "publication_id", "publication_references", "publication_id_status",
    "experiment_series_id", "experiment_series_id_status",
    "replicate_id", "replicate_id_status",
    "legacy_compatible", "provenance_strict",
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--upstream-dir", type=Path, default=None,
                   help="directory of raw *_SAFE.csv exports; auto-detected when omitted")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--include-upstream-columns", action="store_true",
                   help="also write the recovered raw experimental columns into the frozen table")
    p.add_argument("--include-curator", action="store_true", default=True,
                   help="include curator names in the audit's aggregate counts (default on)")
    p.add_argument("--no-include-curator", dest="include_curator", action="store_false")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / "runs" / f"gen6_provenance_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    log(f"reading {args.dataset}")
    source = pd.read_parquet(args.dataset)
    upstream_dir = args.upstream_dir or find_upstream_directory()
    if upstream_dir is None:
        log("upstream SAFE exports NOT found — publication identity will be reported as unavailable")
    else:
        log(f"upstream SAFE exports: {upstream_dir}")

    audit = reconstruct_provenance(source, upstream_directory=upstream_dir,
                                   include_curator=args.include_curator)
    log(f"publication_id: {audit.audit['publication_id_status']}; "
        f"series: {audit.audit['experiment_series_id_status']}; "
        f"replicate: {audit.audit['replicate_id_status']}")

    columns = [c for c in CORE_COLUMNS if c in audit.table.columns]
    if args.include_upstream_columns:
        columns += [c for c in audit.table.columns if c not in columns]
    table = audit.table[columns]
    table.to_parquet(output_dir / "provenance_table.parquet", index=False)
    audit.to_json(output_dir / "provenance_audit.json")
    report = provenance_audit_report(audit)
    (output_dir / "provenance_report.md").write_text(report)
    print(report)

    (output_dir / "summary.json").write_text(json.dumps({
        "stamp": stamp,
        "dataset": str(args.dataset),
        "upstream_dir": str(upstream_dir) if upstream_dir else None,
        "state": audit.state,
        "audit": audit.audit,
    }, indent=2, default=str) + "\n")

    manifest = RunManifest(layer="gen6_diversity", run_id=f"gen6_provenance_{stamp}")
    manifest.record_dataset(dataset_path=args.dataset, source_frame=source)
    manifest.record_code([REPO_ROOT / "src" / "lanthanide_separation" / "gen6",
                          Path(__file__)], repo_root=REPO_ROOT)
    manifest.record_features(feature_sets={})           # this run fits nothing
    manifest.record_chemistry(definition={"used": False, "reason": "provenance run"})
    manifest.record_provenance(state=audit.state)
    manifest.record_preprocessing([{"step": "upstream join", "key": "safe_exp_id",
                                    "matched_fraction": audit.audit.get("upstream_join_fraction")}])
    manifest.record("model_seed", None)
    manifest.record("split_seeds", [])
    manifest.record_split(definition={"algorithm": "none", "reason": "provenance run fits nothing"},
                          folds=[])
    payload = manifest.write(output_dir)
    validation = validate_run(
        output_dir, manifest=payload,
        required_artifacts=["provenance_table.parquet", "provenance_audit.json",
                            "provenance_report.md", "summary.json"],
        checks={
            "every_row_has_an_experiment_id": bool(
                audit.table["experiment_id"].notna().all()),
            "status_vocabulary_respected": {
                "ok": bool(set(audit.table["publication_id_status"]) <=
                           {"reconstructed", "surrogate", "ambiguous", "unavailable"}),
                "values": sorted(set(audit.table["publication_id_status"])),
            },
            "no_invented_publications": {
                "ok": bool(audit.audit["publication_id_status"] != "unavailable"
                           or audit.table["publication_id"].isna().all()),
                "detail": "when publication identity is unavailable the column must be empty",
            },
        })
    marker = write_success(output_dir, manifest=payload, validation=validation)
    log(f"validation ok={validation['ok']}; marker={marker}")
    log(f"written to {output_dir}")
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
