"""Seal gate for every round-5 scoring script (CLAUDE_CODE_PROMPT.md sections 1.2, 4.6 and 8).

    from r5 import sealcheck
    TAG = sealcheck.require_seal()      # refuses (SystemExit) unless scripts/r5_seal.py --verify passes
    ...
    ok = sealcheck.post_run(TAG, "ECT") # section 8: after every Stage-1 run, the seals still verify

``require_seal`` rebuilds the full manifest (round-4 verifier and the import-closure probe run as child processes, one
at a time) and compares it with manifests/stage0_manifest.json; it returns the 12-character seal tag that the scoring
script records next to every number it writes.

``post_run`` repeats that check after a Stage-1 run (brief 8: "After every Stage-1 run, check that r4_seal.py --verify
still reports 5bacfe4c and that the round-3 LF digests still match") without raising: it records the outcome in
results/leakage/postrun_<ITEM>.json and on an END line of RUN_RECORD.md and returns True only when the round-5 seal
(with the same tag), the round-4 verifier and the round-3 LF digests all still verify.

``refuse_if_sealed`` is the guard of every script that rewrites a sealed input (the Stage-0 scripts, --write-spec, the
EA_CHECK cache build): once manifests/stage0_seal.txt exists it refuses unless the caller passes --amend (an
AMENDMENTS.md entry and a reseal follow).
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

from . import paths

SEAL_SCRIPT = paths.R5 / "scripts" / "r5_seal.py"
SEAL_FILE = paths.R5 / "manifests" / "stage0_seal.txt"
LEAK_DIR = paths.R5 / "results" / "leakage"


def seal_module():
    """scripts/r5_seal.py imported by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("r5_seal", SEAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def require_seal(base: Path | None = None, *, full: bool = True, quiet: bool = False) -> str:
    """The 12-character tag of the verified round-5 seal; SystemExit when no seal exists or it does not verify.
    ``base`` / ``full=False`` exist for the tests on a temporary copy only."""
    ok, msg, seal = seal_module().verify(Path(base) if base is not None else paths.R5, full=full)
    if not ok:
        raise SystemExit(f"round-5 seal does not verify ({msg}; seal {seal[:12] or 'none'}): scoring refused.  Seal "
                         "Stage 0 first (scripts/r5_seal.py --seal), or record an amendment and reseal.")
    if not quiet:
        print("round-5 seal verified", seal[:12])
    return seal[:12]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def post_run(tag: str, item: str, note: str = "", *, base: Path | None = None, full: bool = True) -> bool:
    """Brief section 8 post-run check; never raises.  Writes results/leakage/postrun_<item>.json and appends
    '- <UTC> END <item> seal=<tag> post-run check ok|FAILED (<msg>) <note>' to RUN_RECORD.md; returns ok."""
    base = Path(base) if base is not None else paths.R5
    mod = None
    try:
        mod = seal_module()
        ok5, msg, s = mod.verify(base, full=full)
    except Exception as e:                       # noqa: BLE001  (recorded, never raised)
        ok5, msg, s = False, repr(e), ""
    r4 = {"verified": None, "output": "not run (light check)"}
    r3 = {"ok": None}
    if full and mod is not None:
        try:
            r4 = mod.round4_seal()
        except Exception as e:                   # noqa: BLE001
            r4 = {"verified": False, "output": repr(e)}
        try:
            r3 = mod.round3_check()
        except Exception as e:                   # noqa: BLE001
            r3 = {"ok": False, "error": repr(e)}
    ok = bool(ok5 and s[:12] == tag and (not full or (r4.get("verified") and r3.get("ok"))))
    rec = {"label": "POST-RUN SEAL CHECK (brief 8; no score)", "item": item, "seal": tag, "ok": ok,
           "round5_verify": {"ok": bool(ok5), "msg": msg, "seal": s[:12]}, "round4_output": r4.get("output"),
           "round4_verified": r4.get("verified"), "round3_ok": r3.get("ok"), "utc": _now()}
    try:
        paths.write_text_lf(base / "results" / "leakage" / f"postrun_{item}.json", json.dumps(rec, indent=1) + "\n")
    except Exception:                            # noqa: BLE001
        pass
    with open(base / "RUN_RECORD.md", "ab") as fh:
        fh.write((f"- {rec['utc']} END {item} seal={tag} post-run check {'ok' if ok else 'FAILED'} ({msg})"
                  + (f" {note}" if note else "") + "\n").encode("utf-8"))
    print(f"post-run seal check {item}: {'ok' if ok else 'FAILED'} ({msg}; round 4: {r4.get('output')}; round 3 ok: "
          f"{r3.get('ok')})")
    return ok


def refuse_if_sealed(what: str, amend: bool = False, *, seal_file: Path | None = None) -> None:
    """SystemExit when round 5 is sealed (manifests/stage0_seal.txt exists) and ``amend`` is False: a sealed input is
    changed only through an AMENDMENTS.md entry and a reseal."""
    sf = Path(seal_file) if seal_file is not None else SEAL_FILE
    if sf.exists() and not amend:
        raise SystemExit(f"{what}: round 5 is sealed; change sealed inputs via an AMENDMENTS.md entry and reseal "
                         "(--amend)")


def reduced_coverage(**given) -> list:
    """Names of leakage-coverage options that differ from the registered run (brief 8): a seed-count option given
    with a value other than len(paths.DEV_SEEDS), or a skip flag that is set.  ``given`` maps option -> value; a
    value of None means 'not given'; booleans are skip flags."""
    bad = []
    for k, v in given.items():
        if v is None or v is False:
            continue
        if v is True or int(v) != len(paths.DEV_SEEDS):
            bad.append(f"{k}={v}")
    return bad


def forbid_reduced_coverage(smoke: bool, **given) -> None:
    """Outside --smoke, refuse an option that shrinks or switches off a section-8 check: the registered run covers
    every dev seed and every check."""
    if smoke:
        return
    bad = reduced_coverage(**given)
    if bad:
        raise SystemExit(f"reduced leakage coverage is not the registered run ({', '.join(bad)}); allowed only in --smoke")


def stop(item: str, mode: str, tag: str, msg: str) -> None:
    """Append a STOP line to RUN_RECORD.md and raise SystemExit(msg) (brief 6.0 / 8: a failed check stops the item
    before any scored output is written)."""
    from . import common as CM
    CM.run_record(item, mode, tag, event="STOP", note=msg)
    raise SystemExit(msg)


def run_sealed(item: str, mode: str, tag: str, fn) -> int:
    """A sealed (non-smoke) Stage-1 run: a START line in RUN_RECORD.md, ``fn()`` -> status, then the brief-8
    post-run seal check, whose END line carries the status (or the stop / error message).  ``fn`` returns a status or
    (status, detail); the detail goes on the END line.  A failed post-run check adds 4 to the status; a SystemExit or
    error inside ``fn`` is recorded and re-raised."""
    from . import common as CM
    CM.run_record(item, mode, tag)
    try:
        res = fn()
    except SystemExit as e:
        post_run(tag, item, note=f"{mode} STOP ({e})")
        raise
    except BaseException as e:                   # noqa: BLE001  (recorded, re-raised)
        post_run(tag, item, note=f"{mode} ERROR ({type(e).__name__}: {e})")
        raise
    status, detail = (res if isinstance(res, tuple) else (res, ""))
    status = int(status or 0)
    ok = post_run(tag, item, note=f"{mode} status={status}" + (f"; {detail}" if detail else ""))
    return status if ok else (status | 4)

