"""``scripts/g19_write_d02.py`` -- ``decisions/D02_factorization.md`` ALONE, before the ladder exists.

``scripts/g19_build_report.py`` writes four artefacts (``GEN19_REPORT.md``, ``SUMMARY.md``,
``decisions/D02_factorization.md``, ``tables/claims.json``) and refuses to start until the M3-M7
ladder has run to a done / skipped status (its gate 3, ``h3.ladder_complete``), because the report's
deployed predictor is 'the retained ladder configuration'.  The discovery run and the scorer are
complete, so the D02 decision file (brief section 29) is answerable now; the report is not.

This script therefore runs the D02 part and nothing else: ``gen19ct.evaluation.report.d02_text`` on
``report.Ledger`` / ``report._context``, the same generator ``g19_build_report.py`` calls, with no
edit to either module -- their combined SHA-256 is the ``report`` entry of
``manifests/digest_registry.json`` (POST-HOC addendum 2 item 5), and editing them to add a
``--d02-only`` switch would leave that registered digest describing code that no longer exists.
``g19_build_report.py`` regenerates the file, with the ladder rows and the deployed predictor filled
in, once its own gates pass.

Discipline, unchanged from the report:

* every number is read from a file and cites its path and key (``tables/d02_numbers.csv``), and is
  RE-RESOLVED from that source after the text is built (``report.verify_numbers``,
  ``tables/d02_numbers_verification.csv``); a number that does not re-resolve exits 3;
* an input that does not exist prints ``not computed (input missing: <path>)`` -- the M3-M7 rows and
  the deployed predictor do, until the ladder runs;
* the seal gate of the ``report`` stage runs first (``registry.refuse_unless_sealed``): sealed text,
  registered footer digest, 2 addenda, the registered addendum text;
* no discovery record is read, verified or re-verified: the inputs are the scorer's small outputs.

Manifest ``manifests/g19_write_d02.json`` (git HEAD, gate, inputs, outputs, verification).

Run from the repository root:
    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer .venv/Scripts/python.exe \
        generations/gen19_chem_transfer/scripts/g19_write_d02.py
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.evaluation import report as R  # noqa: E402
from gen19ct.manifest import Run, git_head, write_csv, write_text  # noqa: E402

NAME = "g19_write_d02"
#: the registry stage whose seal expectations gate this script: D02 is a report artefact
STAGE = "report"
D02_REL = "decisions/D02_factorization.md"
NUMBERS_REL = "tables/d02_numbers.csv"
VERIFICATION_REL = "tables/d02_numbers_verification.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def provenance_note(n_numbers: int, n_ok: int) -> list[str]:
    """The section appended to the generator's text: the byline at the top of the file is
    ``d02_text``'s own and names ``g19_build_report.py``, so the file says who actually ran it."""
    return ["## provenance of this file", "",
            f"*Written by `scripts/g19_write_d02.py` (manifest `manifests/{NAME}.json`), which calls the D02 generator of "
            "`scripts/g19_build_report.py` -- `gen19ct.evaluation.report.d02_text` -- and nothing else; the byline at the top "
            "is that generator's own. It was run alone because the full report refuses to start until the M3-M7 ladder has "
            "run (`g19_build_report.py` gate 3, `gen19ct.evaluation.h3.ladder_complete`), and no report-stage code was edited "
            "to make that possible: the combined digest of `gen19ct/evaluation/report.py` + `scripts/g19_build_report.py` is "
            "the `report` entry of `manifests/digest_registry.json`. `g19_build_report.py` will regenerate this file from "
            "whatever files exist when its gates pass, with the ladder rows and the deployed predictor filled in. Every "
            f"number above was re-resolved from its cited source after the text was built ({n_ok} of {n_numbers}; "
            f"`{NUMBERS_REL}`, `{VERIFICATION_REL}`). No discovery record was read, verified or re-verified: a record is "
            "checked only against `manifests/digest_registry.json` (POST-HOC addendum 2 item 5), and this script reads the "
            "scorer's small outputs.*", ""]


def build(out_root: Path, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``decisions/D02_factorization.md`` as text, with its ledger and the re-resolution of every
    number it prints."""
    root = Path(out_root)
    L = R.Ledger(root)
    ctx = R._context(L)                                   # the report's own context builder, unmodified
    body = R.d02_text(L, ctx, dict(extra or {}))
    numbers = L.frame()
    ver = R.verify_numbers(root, numbers)
    n_ok = int(ver["ok"].sum()) if len(ver) else 0
    text = body.rstrip("\n") + "\n\n" + "\n".join(provenance_note(len(ver), n_ok))
    return {"text": text, "numbers": numbers, "verification": ver, "ledger": L, "context": ctx,
            "n_numbers": int(len(ver)), "n_ok": n_ok, "all_ok": bool(ver["ok"].all()) if len(ver) else True,
            "failed": ver[~ver["ok"].astype(bool)].head(20).to_dict("records") if len(ver) else []}


def write(out_root: Path, res: Mapping[str, Any]) -> list[Path]:
    root = Path(out_root)
    return [write_text(root / D02_REL, res["text"]), write_csv(res["numbers"], root / NUMBERS_REL),
            write_csv(res["verification"], root / VERIFICATION_REL)]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="build and verify, write nothing")
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def main(argv=None, *, check: Callable[[], int] | None = None,
         digests: Callable[[], Mapping[str, Any]] | None = None) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    gate = REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=ns.expect_addenda)
    log(f"seal gate: stage {gate['stage']}, source {gate['gate_source']}, {gate['n_addenda']} addenda, "
        f"prereg {str(gate['prereg_sha256'])[:12]}...")
    res = build(out_root, extra={"git_head": git_head()})
    if not res["all_ok"]:
        log(f"ERROR: {res['n_numbers'] - res['n_ok']} number(s) do not re-resolve: {res['failed'][:5]}")
        return 3
    if ns.dry_run:
        log(f"dry run: {res['n_numbers']} numbers, all re-resolved; missing inputs "
            f"{sorted(res['ledger'].missing_inputs)}; nothing written")
        return 0
    if ns.no_manifest:
        outs = write(out_root, res)
    else:
        with Run(NAME, args=dict(vars(ns)), seed=D.PRIMARY_SEED,
                 extra={"prereg_gate": gate, "stage": STAGE,
                        "generator": "gen19ct.evaluation.report.d02_text (scripts/g19_build_report.py's D02 part)",
                        "reason": "g19_build_report.py refuses until the M3-M7 ladder is complete (h3.ladder_complete); "
                                  "the D02 decision file is answerable from the completed discovery run and the scorer",
                        "verification": {"n_numbers": res["n_numbers"], "n_ok": res["n_ok"], "all_ok": res["all_ok"]},
                        "missing_inputs": sorted(res["ledger"].missing_inputs),
                        "records_read": 0, "records_verified": 0, "confirmation_half_read": False,
                        "v6_target_rows_scored": 0, "seed": D.PRIMARY_SEED}) as run:
            outs = write(out_root, res)
            run.inputs(*[out_root / p for p, ok in res["ledger"].inputs.items() if ok])
            run.outputs(*outs)
    log(f"wrote {', '.join(paths.rel(p) for p in outs)}: {res['n_numbers']} numbers, {res['n_ok']} re-resolved; "
        f"missing inputs {sorted(res['ledger'].missing_inputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
