"""``scripts/g19_h3_refit_stale.py`` -- POST-HOC addendum 4 item 3: list, and delete, the H3 fold records that were
written under a SUPERSEDED ``h3`` registry entry AFTER that entry was superseded.

Addendum 2 item 5 verifies a record against the registry entry of its stage and protects a record written EARLIER:
"a later change can neither validate nor invalidate a record written earlier".  POST-HOC addendum 4 item 3 closes the
other half: while H3 was running, its module was corrected (the deployed-arm alias fix and the F6 guard), which
superseded the ``h3`` entry, and 13 ACT_PERMUTED folds were written after that point still carrying the superseded
digests.  Those records are DELETED and refitted under the current registered ``h3`` digest before any H3 delta is
scored; records written before the supersession keep their entry and verify as registered.

What this script does (it fits nothing and writes no record):

* ``--list`` (the default) prints, from ``gen19ct.evaluation.h3.stale_records``:
  - ``stale``      -- records of a SCORED transform that do not verify at their own timestamp: exactly what goes;
  - ``stale_exploratory`` -- the ACT_METAL_SHUFFLED fold(s), kept as ``exploratory_not_scored`` (addendum 4 item 1);
  - ``verifying``  -- records that match an entry of stage ``h3`` (current or superseded) at their own timestamp;
* ``--delete`` removes exactly the ``stale`` list (each ``*.json`` with its sibling ``*.parquet``, because a prediction
  without a record raises) and re-verifies afterwards: the pre-supersession records must still verify and no stale
  record may remain.  Nothing else is touched and no record is rewritten.
* the outcome is written to ``evaluation/h3/decisions/refits.json``.

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_h3_refit_stale.py [--delete]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import h3 as H3  # noqa: E402
from gen19ct.manifest import git_head, write_json  # noqa: E402

NAME = "g19_h3_refit_stale"
STAGE = "h3"


def refits_path(out_root: Path) -> Path:
    return H3.h3_decisions_dir(out_root) / "refits.json"


def run(out_root: Path, *, delete: bool = False, registry: Path | None = None) -> dict[str, Any]:
    """List (and optionally delete) the stale records, then re-verify.  Returns the record written to
    ``evaluation/h3/decisions/refits.json``."""
    before = H3.stale_records(out_root, stage=STAGE, path=registry)
    action = H3.delete_records(before["stale"], out_root=paths.REPO_ROOT, dry_run=not delete)
    after = H3.stale_records(out_root, stage=STAGE, path=registry) if delete else before
    if delete and after["stale"]:
        raise SystemExit(f"refused: {len(after['stale'])} stale record(s) remain after the deletion: "
                         f"{[r['path'] for r in after['stale'][:5]]}")
    lost = ([r["path"] for r in before["verifying"]
             if r["path"] not in {v["path"] for v in after["verifying"]}] if delete else [])
    if lost:
        raise SystemExit(f"refused: {len(lost)} record(s) that verified before the deletion no longer verify "
                         f"({lost[:5]}); POST-HOC addendum 4 item 3 deletes exactly the stale records and nothing else")
    return {"schema": H3.SCHEMA, "script": NAME, "git_head": git_head(),
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "deleted": bool(delete), "action": action,
            "stale_before": before["stale"], "n_stale_before": len(before["stale"]),
            "stale_after": after["stale"], "n_stale_after": len(after["stale"]),
            "stale_exploratory_kept": before["stale_exploratory"],
            "n_verifying_before": len(before["verifying"]), "n_verifying_after": len(after["verifying"]),
            "verifying_after": after["verifying"],
            # WHICH entry each record verifies against, and WHICH timestamp the ordering check used: no H3 record
            # verifies against the CURRENT entry, and 43 records carry no ``written_utc`` at all, so the exemption their
            # ordering rests on must be auditable from this file (task X findings numbers VH-04 / VH-05)
            "ordering": {
                "matched_entry_counts": after.get("matched_entry_counts"),
                "n_verifying_against_the_current_entry": after.get("n_verifying_against_the_current_entry"),
                # ... and WHY that count is 0, so the absence of a `current` record is a recorded decision and not a
                # silent gap (TASK F item 7)
                "why_no_record_is_current": after.get("why_no_record_is_current"),
                "written_utc_source_counts": after.get("written_utc_source_counts"),
                "operative_reading": after.get("operative_reading"),
                "per_record": sorted(({"path": r.get("path"), "written_utc": r.get("written_utc"),
                                       "written_utc_source": r.get("written_utc_source"),
                                       "ordering_checked": r.get("ordering_checked"),
                                       "matched_entry": r.get("matched_entry"),
                                       "code_digest": str(r.get("code_digest") or "")[:12],
                                       "superseded_utc_of_matched_entry": r.get("superseded_utc_of_matched_entry")}
                                      for r in [*(after.get("verifying") or []), *(after.get("stale") or []),
                                                *(after.get("stale_exploratory") or [])]),
                                     key=lambda r: str(r["path"]))},
            "rule": before["rule"], "reading": H3.READINGS["record_digest_basis"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--delete", action="store_true", help="delete exactly the stale records (default: list only)")
    ap.add_argument("--registry", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--no-write", action="store_true", help="print the outcome; write no decisions file")
    ns = ap.parse_args(argv)
    out_root = Path(ns.out_root)
    rec = run(out_root, delete=bool(ns.delete), registry=None if ns.registry is None else Path(ns.registry))
    if not ns.no_write:
        # a later --list must never erase the record of the deletion that happened: the previous outcome is kept
        prev = json.loads(refits_path(out_root).read_text(encoding="utf-8")) if refits_path(out_root).exists() else None
        if prev is not None:
            rec["previous"] = {k: v for k, v in prev.items() if k != "previous"}
            rec["previous_note"] = ("the earlier outcome of this script, kept so the DELETION that satisfied POST-HOC "
                                    "addendum 4 item 3 stays on record after a later listing run")
        rec["path"] = paths.rel(write_json(refits_path(out_root), H3.json_safe(rec)))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("verifying_after",)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
