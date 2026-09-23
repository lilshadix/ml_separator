"""``scripts/g19_power_debt.py`` -- write the section 8 ``needs_power`` inventory into ``evaluation/power/power_checks.json``
without running a single check.

POST-HOC addendum 2's reading ``needs_power`` is: "a contrast needs the check when its family is primary (H1), H1b, S1(b)
or H3 and its full R19 verdict is not PASS".  Every H3 contrast of the discovery run satisfies it (each row of
``evaluation/h3/h3_contrasts.csv`` is FAIL or UNDECIDED on the full R19 verdict) and none was checked: the four checks in
``power_checks.json`` are the H1 / H1b / S1(b) ones.  The file listed no inventory, so the unrun H3 checks were an
ABSENCE rather than a recorded debt -- nothing said which contrasts owed one (task X finding protocol VH-09).

This script only ADDS the inventory (``gen19ct.evaluation.h3.needs_power_inventory``) under the ``needs_power`` key of the
existing ``power_checks.json``, leaving every other key -- ``checks`` above all -- byte-for-byte as the power runner wrote
it.  It runs no fit, no bootstrap and no injection, and it changes no verdict: the H3 verdicts already read
``UNDECIDED (no registered power check)`` (POST-HOC addendum 4 item 4).  ``scripts/g19_run_power.py`` writes the same
inventory itself from now on, so this script is only needed for a ``power_checks.json`` written before that.

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_power_debt.py [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import h3 as H3  # noqa: E402
from gen19ct.manifest import git_head, write_json  # noqa: E402

NAME = "g19_power_debt"


def inventory(out_root: Path) -> dict[str, Any]:
    """The inventory, from the H3 contrast table and the checks ``power_checks.json`` already holds."""
    path = H3.h3_root(out_root) / "h3_contrasts.csv"
    contrasts = pd.read_csv(path) if path.exists() else None
    # the fold counts and per-fold seconds that PRICE each unrun check, so the debt is a number of refits and hours and
    # not only a list of names (TASK F item 6); both come from files the H3 run already wrote
    summary = H3.h3_root(out_root) / "h3_summary.json"
    sets = (json.loads(summary.read_text(encoding="utf-8")).get("record_sets") if summary.exists() else None) or {}
    inv = H3.needs_power_inventory(contrasts, checked=H3.power_checked_keys(out_root), record_sets=sets,
                                   per_fold_seconds=H3.h3_per_fold_seconds(out_root))
    inv.update({"source_contrasts": paths.rel(path) if path.exists() else None,
                "source_record_sets": paths.rel(summary) if summary.exists() else None,
                "written_by": NAME, "git_head": git_head(),
                "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    return inv


def run(out_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    checks = H3.power_root_checks_path(out_root)
    body = json.loads(checks.read_text(encoding="utf-8")) if checks.exists() else {}
    inv = inventory(out_root)
    body["needs_power"] = inv
    if not dry_run:
        write_json(checks, H3.json_safe(body))
    return {"path": paths.rel(checks), "written": not dry_run, "n_owed": inv["n_owed"],
            "n_owed_and_run": inv["n_owed_and_run"], "n_owed_and_not_run": inv["n_owed_and_not_run"],
            "n_checks_in_file": len(body.get("checks") or []), "keys": sorted(body)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="print the inventory; write nothing")
    ns = ap.parse_args(argv)
    print(json.dumps(run(Path(ns.out_root), dry_run=bool(ns.dry_run)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
