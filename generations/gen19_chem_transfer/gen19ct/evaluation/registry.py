"""``evaluation/registry.py`` -- the digest registry of POST-HOC addendum 2, "What changes" item 5.

Addendum 2 item 5 (quoted; each sentence is implemented by the function named after it):

* "a digest registry (``manifests/digest_registry.json``) records, per stage, the below-footer digest and the code
  digest under which that stage's records were written (discovery: the addendum-1 digest and the discovery code digest)"
  -- :func:`register_stage`, :func:`registered`; for the discovery stage :func:`register_discovery_from_records` reads
  both digests FROM the completed run's record fields (``prereg_addenda_sha256``, ``code_digest``), never recomputed;
* "a record is verified against the registry entry of its stage, never against the live text or live code" --
  :func:`verify_record`;
* "no record is rewritten" -- nothing here writes under ``evaluation/``;
* "the report prints every registry entry" -- ``evaluation.report.reproducibility_section`` reads :func:`read_registry`;
* "Any later edit to an existing module is logged in the registry with its commit and reason and can neither validate
  nor invalidate a record written earlier" -- :func:`log_code_change` appends to ``code_changes``; stage entries are
  immutable (a differing re-registration is refused unless ``force=True``, which keeps the earlier entry under
  ``superseded``), and :func:`verify_record` still verifies a record written earlier against that earlier entry
  (:func:`superseded_entries`), so re-registering a stage under a later addendum invalidates nothing.  A POST-HOC
  addendum appended below the footer changes the below-footer digest of every stage that has not yet written its
  records; those stages are re-registered (``register --stage ... --force``) and the stages whose records exist --
  ``discovery``, ``discovery_candidates`` -- keep their entries;
* "The gate constants are replaced by the registry when this addendum is appended - a code change made after the
  discovery run has completed and before any post-discovery runner starts" -- :func:`gate_expectations`,
  :func:`below_footer_sha256`, :func:`addenda_count`, :func:`discovery_code_digest` and :func:`refuse_unless_sealed`
  PREFER the registry when ``manifests/digest_registry.json`` exists and FALL BACK to the constants of
  ``evaluation.discovery`` (``N_ADDENDA_EXPECTED``, ``REGISTERED_ADDENDA_SHA256``) and to the discovery runner's live
  ``current_code_digest()`` otherwise, so nothing changes behaviour until the registry is created.  The closure-side
  replacement (``discovery.py``, ``g19_run_discovery.py``, ``g19_score_discovery.py``) is the orchestrator's patch,
  applied after the run.

Digest rules: the below-footer digest is ``sha256(LF-normalised text below the single footer line)`` exactly as
``scripts/g19_seal_prereg.py`` splits it and ``g19_run_discovery.prereg_digests`` hashes it (:func:`below_footer_digest`);
the discovery code digest is ``g19_run_discovery.current_code_digest()`` reused by import (:func:`live_discovery_code_digest`).
The file is written with ``gen19ct.manifest.write_json`` (LF, sorted keys).

The discovery runner has TWO stages (task X finding on the patch's "Note on behaviour"): ``discovery`` -- every job of
``discovery.enumerate_plan`` up to the ``M3+`` marker EXCEPT the conditional freezing-candidate jobs -- and
``discovery_candidates`` (:data:`CANDIDATES`) -- the jobs of ``discovery.STAGES['candidates']``, which exist only once
``plan_state.freezing_candidates`` is non-empty, i.e. after ``scripts/g19_score_discovery.py`` has run, which is after
addendum 2 is appended.  Their records are written under the two-addenda text and the post-patch runner code, so they
are their own stage: registered from the tree BEFORE the runner fits them (:func:`register_stage_from_tree`), gated on
that entry, and verified against it.  :func:`job_stage` maps a job or record to its registry stage (the job's ``stage``
field, which every record's ``job`` block carries; it is a ``discovery.LABEL_FIELDS`` label, outside the resume digest);
:func:`code_digest_for` gives the code digest a record of a stage is verified with; :func:`refuse_unless_writable` is the
runner's rule that a record of a registered stage is written only under the code the stage was registered with.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gen19ct import paths
from gen19ct.evaluation import discovery as D
from gen19ct.manifest import git_head as _git_head
from gen19ct.manifest import write_json

SCHEMA = "gen19.digest_registry.v1"
REGISTRY_REL = "manifests/digest_registry.json"
#: the stages whose records carry a below-footer digest and a code digest (each post-discovery runner is one stage; the
#: discovery runner is two, see the module docstring)
STAGES: tuple[str, ...] = ("discovery", "discovery_candidates", "scorer", "ladder", "h3", "power", "process", "figures",
                           "report", "confirmation")
DISCOVERY = "discovery"
#: the freezing-candidate pass of the discovery runner (``discovery.STAGES['candidates']`` jobs, fitted after the scorer)
CANDIDATES = "discovery_candidates"
#: the two stages whose records ``scripts/g19_run_discovery.py`` writes
RUNNER_STAGES: tuple[str, ...] = (DISCOVERY, CANDIDATES)
#: the record fields compared by :func:`verify_record`
RECORD_FIELDS: dict[str, str] = {"prereg_addenda_sha256": "below_footer_sha256", "code_digest": "code_digest"}
_HEX64 = 64

READINGS: dict[str, str] = {
    "registry": "addendum 2 item 5: a record is verified against the registry entry of its stage (the below-footer digest "
                "and the code digest under which that stage's records were written), never against the live text or live "
                "code; the discovery entry is read from the completed run's record fields, not recomputed; no record is "
                "rewritten; a later code edit is logged with its commit and reason and validates or invalidates nothing. "
                "When a POST-HOC addendum is appended below the footer, every stage whose records do NOT yet exist is "
                "re-registered under the new below-footer digest (register --stage ... --force, the earlier entry kept "
                "under 'superseded') while the stages that HAVE written records keep their entries; a record written "
                "under a superseded entry of its own stage still verifies against that entry (verify_record -> "
                "matched_entry 'superseded'), so appending an addendum invalidates no record -- ordered by the "
                "record's own timestamp (registry.record_run_utc): 'earlier' means written before that entry was "
                "superseded, and a record a runner holding stale code wrote AFTER the re-registration does not verify "
                "(stale_after_supersession); a record carrying no timestamp verifies with ordering_checked false",
    "fallback": "until manifests/digest_registry.json exists every gate reads the constants of evaluation.discovery "
                "(N_ADDENDA_EXPECTED, REGISTERED_ADDENDA_SHA256) and the live discovery code digest, exactly as before",
    "candidates": "the conditional freezing-candidate jobs of the discovery runner (discovery.STAGES['candidates'], "
                  "enumerated only from plan_state.freezing_candidates, which the scorer writes) run AFTER the scorer, "
                  "hence after addendum 2 is appended and after the registry replaces the gate constants; they are the "
                  "registry stage 'discovery_candidates', registered from the tree (two-addenda text, the runner's code "
                  "digest at that time) before the runner fits them; the runner gates on that stage once "
                  "plan_state.freezing_candidates is non-empty, writes their records with that stage's digests "
                  "(fold_digest, prereg_addenda_sha256, code_digest, registry_stage) and refuses to FIT any job of a "
                  "registered stage whose live code digest is not the registered one (a complete record set is skipped, "
                  "an incomplete one is an error, never a refit); every reader resolves the code digest a record set is "
                  "verified with from the job stage its records carry (registry.job_stage / code_digest_for)",
}


def registry_path(root: Path | None = None) -> Path:
    return (paths.G19_ROOT if root is None else Path(root)) / REGISTRY_REL


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _hex(value: Any, what: str) -> str:
    s = str(value)
    if len(s) != _HEX64 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError(f"{what} is not a SHA-256 hex digest: {s[:16]!r}...")
    return s


# --------------------------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------------------------- #

def read_registry(path: Path | None = None) -> dict[str, Any] | None:
    """The registry body, or ``None`` when the file does not exist (the constants then govern)."""
    p = registry_path() if path is None else Path(path)
    if not p.exists():
        return None
    body = json.loads(p.read_text(encoding="utf-8"))
    if body.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {body.get('schema')!r}, expected {SCHEMA!r}")
    return body


def _empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "stages": {}, "superseded": [], "code_changes": [],
            "rule": READINGS["registry"], "fallback": READINGS["fallback"]}


def _write(body: Mapping[str, Any], path: Path | None) -> Path:
    return write_json(registry_path() if path is None else Path(path), dict(body))


def registered(stage: str, path: Path | None = None) -> dict[str, Any] | None:
    """The registry entry of ``stage`` (``None`` when the registry or the stage is absent)."""
    body = read_registry(path)
    if body is None:
        return None
    e = (body.get("stages") or {}).get(stage)
    return None if e is None else dict(e)


def register_stage(stage: str, *, below_footer_sha256: str, code_digest: str, git_head: str | None, addenda_count: int,
                   note: str, path: Path | None = None, force: bool = False, extra: Mapping[str, Any] | None = None
                   ) -> dict[str, Any]:
    """Write or update ``manifests/digest_registry.json`` with the entry of ``stage``: "records, per stage, the
    below-footer digest and the code digest under which that stage's records were written".  An identical
    re-registration is a no-op; a DIFFERENT digest for an existing stage is refused unless ``force=True``, which moves the
    earlier entry to ``superseded`` (nothing is silently overwritten)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; registry stages are {STAGES}")
    entry = {"stage": stage, "below_footer_sha256": _hex(below_footer_sha256, "below_footer_sha256"),
             "code_digest": _hex(code_digest, "code_digest"), "git_head": None if git_head is None else str(git_head),
             "addenda_count": int(addenda_count), "note": str(note), "registered_utc": _now(), **dict(extra or {})}
    body = read_registry(path) or _empty()
    stages = body.setdefault("stages", {})
    old = stages.get(stage)
    if old is not None:
        same = all(old.get(k) == entry[k] for k in ("below_footer_sha256", "code_digest", "addenda_count"))
        if same:
            # the digests ARE the record; the descriptive fields are not, so an identical re-registration may correct
            # them (task X finding V-REG-08: 'source: current tree' read as if the code digest came from the tree, while
            # for the scorer and the ladder stage_code_digest() deliberately gives the DISCOVERY entry's digest)
            amended = {k: v for k, v in entry.items()
                       if k not in ("registered_utc",) and old.get(k) != v}
            if not amended:
                return dict(old)
            stages[stage] = {**old, **amended, "amended_utc": _now(),
                             "amendment": "descriptive fields corrected; the below-footer digest, code digest and "
                                          "addenda count are unchanged, so no record's verification changes"}
            _write(body, path)
            return dict(stages[stage])
        if not force:
            raise ValueError(f"stage {stage!r} is already registered with other digests (below-footer "
                             f"{str(old.get('below_footer_sha256'))[:12]}..., code {str(old.get('code_digest'))[:12]}...); "
                             "a stage's records were written once -- pass force=True only to supersede deliberately")
        body.setdefault("superseded", []).append({**old, "superseded_utc": _now(), "superseded_by_note": str(note)})
    stages[stage] = entry
    _write(body, path)
    return dict(entry)


def _mismatches(record: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    mism: dict[str, Any] = {}
    for rec_key, reg_key in RECORD_FIELDS.items():
        have, want = record.get(rec_key), entry.get(reg_key)
        if have != want:
            mism[rec_key] = {"record": have, "registry": want}
    if record.get("prereg_n_addenda") is not None and int(record["prereg_n_addenda"]) != int(entry["addenda_count"]):
        mism["prereg_n_addenda"] = {"record": record.get("prereg_n_addenda"), "registry": entry["addenda_count"]}
    return mism


def superseded_entries(stage: str, path: Path | None = None) -> list[dict[str, Any]]:
    """The stage's earlier entries, newest first: what :func:`register_stage` moved to ``superseded`` when the stage was
    re-registered under a later sealed text or later code."""
    body = read_registry(path) or {}
    out = [(i, dict(e)) for i, e in enumerate(body.get("superseded") or []) if str(e.get("stage")) == stage]
    # the append position breaks a tie: `superseded_utc` has second resolution, and two re-registrations in the same
    # second must still order newest first
    return [e for _, e in sorted(out, key=lambda t: (str(t[1].get("superseded_utc") or ""), t[0]), reverse=True)]


def record_run_utc(record: Mapping[str, Any]) -> str | None:
    """WHEN a record was written, for :func:`verify_record`'s ordering check: its own ``written_utc`` when the runner
    stamped one, else the latest ``steps.<step>.date_utc`` the record carries (the run timestamp of its last step),
    else ``None`` -- a record that carries no timestamp at all cannot be ordered against a supersession."""
    stamp = record.get("written_utc")
    if stamp:
        return str(stamp)
    steps = record.get("steps")
    times = [str(s["date_utc"]) for s in (steps or {}).values()
             if isinstance(s, Mapping) and s.get("date_utc")] if isinstance(steps, Mapping) else []
    return max(times) if times else None


def record_run_utc_source(record: Mapping[str, Any]) -> str:
    """WHERE :func:`record_run_utc` took the timestamp from: ``"written_utc"``, ``"max(steps.*.date_utc)"`` or
    ``"none"``.

    The ordering check that protects a record written BEFORE a supersession rests on this stamp, and a record without
    ``written_utc`` is ordered by a DERIVED value, so the source has to be reported wherever the exemption is claimed
    rather than re-derived by the reader (task X finding numbers VH-05)."""
    if record.get("written_utc"):
        return "written_utc"
    steps = record.get("steps")
    if isinstance(steps, Mapping) and any(isinstance(s, Mapping) and s.get("date_utc") for s in steps.values()):
        return "max(steps.*.date_utc)"
    return "none"


def registered_code_digests(stage: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Every code digest a record of ``stage`` may legitimately carry, newest first: the stage's current entry, then its
    superseded entries (:func:`superseded_entries`).  Each item is ``{"code_digest", "entry", "matched_entry",
    "superseded_utc"}``.  A reader that verifies a RECORD SET written before a re-registration (addendum 2 item 5)
    resolves the set's code digest through this list instead of the live digest."""
    out: list[dict[str, Any]] = []
    cur = registered(stage, path)
    if cur is not None:
        out.append({"code_digest": str(cur["code_digest"]), "entry": cur, "matched_entry": "current",
                    "superseded_utc": None})
    for old in superseded_entries(stage, path):
        out.append({"code_digest": str(old.get("code_digest")), "entry": old, "matched_entry": "superseded",
                    "superseded_utc": old.get("superseded_utc")})
    return out


def verify_record(record: Mapping[str, Any], stage: str | None, path: Path | None = None, *,
                  raise_on_mismatch: bool = False, written_utc: str | None = None) -> dict[str, Any]:
    """"A record is verified against the registry entry of its stage, never against the live text or live code": compares
    the record's ``prereg_addenda_sha256`` and ``code_digest`` fields (and ``prereg_n_addenda`` when present) with the
    stage's entry.  ``stage=None`` takes the stage the record carries (``registry_stage``, else :func:`job_stage` of its
    ``job`` block).  Returns ``{"ok", "stage", "mismatches", "registry_entry"}``; without a registry entry ``ok`` is None
    and the reason is named (nothing is recomputed from the live tree here).

    A record that matches a SUPERSEDED entry of its own stage verifies against THAT entry (``matched_entry`` says
    which, ``superseded_match`` carries it): addendum 2 item 5 -- a later change "can neither validate nor invalidate a
    record written earlier", and re-registering a stage under a later addendum (addendum 3) is such a change.  A record
    that matches no entry of its stage, current or superseded, does not verify.

    The superseded fallback protects records written EARLIER, so it is ORDERED (task X finding V-P06): the record's own
    timestamp (:func:`record_run_utc`, or ``written_utc`` passed in) must predate the entry's ``superseded_utc``.  A
    record written AFTER the stage was re-registered but still carrying the old code digest was produced by a runner
    holding stale code: it does not verify (``stale_after_supersession`` names the entry it would otherwise have
    matched).  A record that carries NO timestamp cannot be ordered; it still verifies, and the result says so
    (``ordering_checked`` False) rather than claiming an ordering that was not checked."""
    if stage is None:
        stage = str(record.get("registry_stage") or job_stage(record))
    entry = registered(stage, path)
    if entry is None:
        out = {"ok": None, "stage": stage, "mismatches": {}, "registry_entry": None,
               "reason": f"no registry entry for stage {stage!r} ({registry_path() if path is None else Path(path)})"}
        if raise_on_mismatch:
            raise D.StaleRecordError(out["reason"])
        return out
    rule = "record fields compared to the registry entry of the stage, never to the live text or live code"
    mism = _mismatches(record, entry)
    out = {"ok": not mism, "stage": stage, "mismatches": mism, "registry_entry": entry, "rule": rule,
           "matched_entry": "current" if not mism else None}
    if mism:
        stamp = str(written_utc) if written_utc else record_run_utc(record)
        stale: list[dict[str, Any]] = []
        for old in superseded_entries(stage, path):
            if _mismatches(record, old):
                continue
            sup = str(old.get("superseded_utc") or "")
            if stamp is not None and sup and stamp > sup:
                stale.append({"superseded_utc": sup, "code_digest": old.get("code_digest"),
                              "below_footer_sha256": old.get("below_footer_sha256")})
                continue
            return {"ok": True, "stage": stage, "mismatches": {}, "registry_entry": entry,
                    "matched_entry": "superseded", "superseded_match": old,
                    "mismatches_against_current": mism, "rule": rule,
                    "ordering_checked": stamp is not None and bool(sup), "written_utc": stamp,
                    "reason": f"the record was written under an earlier entry of stage {stage!r} "
                              f"(below-footer {str(old.get('below_footer_sha256'))[:12]}..., "
                              f"{old.get('addenda_count')} addenda) and verifies against it: a later "
                              "re-registration validates or invalidates no record written earlier (addendum 2 item 5)"
                              + ("" if stamp is not None and sup else
                                 "; the record carries no timestamp, so 'earlier' could not be checked")}
        if stale:
            out.update({"ok": False, "written_utc": stamp, "ordering_checked": True,
                        "stale_after_supersession": stale,
                        "reason": f"the record carries a SUPERSEDED code / text of stage {stage!r} but was written at "
                                  f"{stamp}, after that entry was superseded ({stale[0]['superseded_utc']}): addendum 2 "
                                  "item 5 protects a record written EARLIER, not one a runner holding stale code wrote "
                                  "later; refit it under the registered code"})
        if raise_on_mismatch:
            raise D.StaleRecordError(out.get("reason") or
                                     f"record of stage {stage!r} differs from its registry entry and from every "
                                     f"superseded entry of that stage: {sorted(mism)}")
    return out


def log_code_change(files: Iterable[str | Path], commit: str | None, reason: str, path: Path | None = None,
                    *, root: Path | None = None) -> dict[str, Any]:
    """"Any later edit to an existing module is logged in the registry with its commit and reason and can neither
    validate nor invalidate a record written earlier": appends ``{files (with their LF sha256 now), commit, reason,
    logged_utc}`` to ``code_changes``.  Stage entries are untouched."""
    base = paths.REPO_ROOT if root is None else Path(root)
    recs = []
    for f in files:
        p = Path(f)
        p = p if p.is_absolute() else base / p
        try:
            rel = paths.rel(p) if root is None else p.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            rel = p.as_posix()
        sha = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if p.exists() else None
        recs.append({"path": rel, "sha256_lf": sha, "exists": p.exists()})
    change = {"files": recs, "commit": None if commit is None else str(commit), "reason": str(reason), "logged_utc": _now(),
              "effect_on_records": "none: validates or invalidates no record written earlier (addendum 2 item 5)"}
    body = read_registry(path) or _empty()
    body.setdefault("code_changes", []).append(change)
    _write(body, path)
    return change


# --------------------------------------------------------------------------------------------- #
# the two digests from the current tree (registration of a post-discovery stage) and from records
# --------------------------------------------------------------------------------------------- #

def _seal_module():
    name = "g19_seal_prereg"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, paths.G19_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _runner_module():
    """``scripts/g19_run_discovery.py`` as a module (its ``current_code_digest`` and ``refuse_unless_sealed``; imports the
    model stack, so it is loaded only when needed)."""
    name = "g19_run_discovery"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, paths.G19_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def below_footer_digest(sealed: Path | None = None, sha_file: Path | None = None) -> dict[str, Any]:
    """The sealed text's digests with the seal script's own functions -- the SAME rule as ``scripts/g19_seal_prereg.py``
    and ``g19_run_discovery.prereg_digests``: BOM dropped, CRLF -> LF, exactly one footer line, ``below_footer_sha256 =
    sha256(text below the footer)``, ``n_addenda`` = the ``## POST-HOC addendum N (date ...)`` headings."""
    sm = _seal_module()
    sealed = paths.G19_ROOT / "preregistration.md" if sealed is None else Path(sealed)
    sha_file = paths.G19_ROOT / "manifests" / "prereg_sha256.txt" if sha_file is None else Path(sha_file)
    text = sm.read_normalised(sealed)
    above, footer, below = sm.split_footer(text)
    return {"footer": sm.footer_digest(footer), "recomputed": sm.digest_of_above(above),
            "digest_file": sm.read_digest_file(sha_file), "below_footer_sha256": D.sha256_text(below),
            "n_addenda": sum(1 for ln in below.split("\n") if sm._ADDENDUM.match(ln))}


def live_discovery_code_digest() -> str:
    """The discovery runner's ``current_code_digest()`` reused by import (CODE_FILES + the runner + RUNNER_OBJECTS)."""
    return str(_runner_module().current_code_digest())


def current_digests(sealed: Path | None = None, sha_file: Path | None = None, *, with_code: bool = True) -> dict[str, Any]:
    """What the current tree would register: the below-footer digest / addenda count and (``with_code``) the live discovery
    code digest and git HEAD."""
    out = below_footer_digest(sealed, sha_file)
    out["git_head"] = _git_head()
    if with_code:
        out["discovery_code_digest"] = live_discovery_code_digest()
    return out


def job_stage(job_or_record: Any) -> str:
    """The registry stage of a discovery job or fold record: :data:`CANDIDATES` for a job of
    ``discovery.STAGES['candidates']`` (the freezing-candidate pass, fitted after the scorer), :data:`DISCOVERY` for every
    other job of the runner.  Accepts a ``JobSpec``, a record dict (its ``job`` block's ``stage``; a record that names its
    ``registry_stage`` is believed) or a job ``record()`` dict."""
    if isinstance(job_or_record, Mapping):
        named = job_or_record.get("registry_stage")
        if named in RUNNER_STAGES:
            return str(named)
        block = job_or_record.get("job")
        stage = (block or {}).get("stage") if isinstance(block, Mapping) else job_or_record.get("stage")
    else:
        stage = getattr(job_or_record, "stage", "")
    return CANDIDATES if str(stage or "") == D.STAGES["candidates"] else DISCOVERY


def plan_record_dirs(out_root: Path, *, stage: str = DISCOVERY, arms: Sequence[str] = D.FITTED_ARMS,
                     state: D.PlanState | None = None) -> list[Path] | None:
    """The record directories ``evaluation/discovery/<arm>/<design_dir>/s<seed>`` of the fit jobs of registry ``stage``
    in ``discovery.enumerate_plan`` under the plan state on disk (``decisions/plan_state.json``); ``None`` when no plan
    state exists (a tree without one holds no completed run to restrict to)."""
    root = D.discovery_root(out_root)
    sp = root / "decisions" / "plan_state.json"
    if state is None:
        if not sp.exists():
            return None
        state = D.PlanState.read(sp)
    out: list[Path] = []
    for j in D.enumerate_plan(state):
        if j.kind != "fit" or job_stage(j) != stage:
            continue
        for arm in j.writes:
            if arm in arms:
                out.append(root / arm / j.design_dir / f"s{j.seed}")
    return sorted(set(out))


def digests_from_discovery_records(out_root: Path | None = None, *, arms: Sequence[str] = D.FITTED_ARMS,
                                   stage: str = DISCOVERY, restrict_to_plan: bool = True) -> dict[str, Any]:
    """The below-footer digest, addenda count and code digest the completed discovery run's fold records CARRY
    (``prereg_addenda_sha256``, ``prereg_n_addenda``, ``code_digest`` of the ``evaluation/discovery/<arm>/**/*.json``
    records of registry ``stage``): read from the fields, never recomputed.  Only records whose job is of ``stage``
    (:func:`job_stage`; the freezing-candidate pass is another stage) count, and -- when ``decisions/plan_state.json``
    exists and ``restrict_to_plan`` -- only those in the record directories of the plan's fit jobs of that stage
    (:func:`plan_record_dirs`): a record left in a directory the current plan no longer fits (an earlier code version's
    job) is listed under ``ignored_paths`` and blocks nothing.  Raises unless every counted record agrees on one value of
    each field, naming the offending paths per value (a mixed set was written under two texts or two codes and must be
    sorted out before registration).  Reads no prediction and no verdict."""
    base = paths.G19_ROOT if out_root is None else Path(out_root)
    root = D.discovery_root(base)
    keep = plan_record_dirs(base, stage=stage, arms=arms) if restrict_to_plan else None
    keep_set = None if keep is None else {p.resolve() for p in keep}
    seen: dict[str, dict[str, list[str]]] = {"prereg_addenda_sha256": {}, "code_digest": {}, "prereg_n_addenda": {}}
    n, ignored, other_stage = 0, [], []
    for arm in arms:
        d = root / arm
        if not d.exists():
            continue
        for js in sorted(d.rglob("*.json")):
            rec = D.read_record(js)
            if rec is None or "digest" not in rec or "code_digest" not in rec:
                continue
            rel = js.relative_to(root).as_posix()
            if job_stage(rec) != stage:
                other_stage.append(rel)
                continue
            if keep_set is not None and js.parent.resolve() not in keep_set:
                ignored.append(rel)
                continue
            n += 1
            for k in seen:
                v = rec.get(k)
                if v is not None:
                    seen[k].setdefault(str(v), []).append(rel)
    if n == 0:
        raise FileNotFoundError(f"no discovery fold record of stage {stage!r} under {root} (arms {list(arms)}; "
                                f"{len(ignored)} outside the plan's directories, {len(other_stage)} of another stage)")
    bad = {k: {v: (ps[:5] + ([f"... {len(ps) - 5} more"] if len(ps) > 5 else [])) for v, ps in vals.items()}
           for k, vals in seen.items() if len(vals) != 1}
    if bad:
        raise ValueError(f"the discovery records of stage {stage!r} do not agree on one value (paths per value, relative "
                         f"to {root}): {json.dumps(bad, indent=1)}")
    return {"below_footer_sha256": next(iter(seen["prereg_addenda_sha256"])), "code_digest": next(iter(seen["code_digest"])),
            "addenda_count": int(next(iter(seen["prereg_n_addenda"]))), "n_records": n, "source": "record fields",
            "stage": stage, "restricted_to_plan": keep is not None, "n_plan_dirs": None if keep is None else len(keep),
            "ignored_paths": ignored[:50], "n_ignored": len(ignored), "n_other_stage": len(other_stage)}


def register_discovery_from_records(out_root: Path | None = None, *, note: str, path: Path | None = None,
                                    git_head: str | None = None, arms: Sequence[str] = D.FITTED_ARMS) -> dict[str, Any]:
    """Register the ``discovery`` stage "(discovery: the addendum-1 digest and the discovery code digest)" from the
    completed run's record fields (:func:`digests_from_discovery_records`); nothing is recomputed from the live tree."""
    dg = digests_from_discovery_records(out_root, arms=arms, stage=DISCOVERY)
    return register_stage(DISCOVERY, below_footer_sha256=dg["below_footer_sha256"], code_digest=dg["code_digest"],
                          git_head=_git_head() if git_head is None else git_head, addenda_count=dg["addenda_count"],
                          note=note, path=path, extra={"source": dg["source"], "n_records": dg["n_records"],
                                                       "restricted_to_plan": dg["restricted_to_plan"],
                                                       "n_ignored_records": dg["n_ignored"]})


# --------------------------------------------------------------------------------------------- #
# what the post-discovery runners read: registry first, constants otherwise
# --------------------------------------------------------------------------------------------- #

def gate_expectations(stage: str, path: Path | None = None) -> dict[str, Any]:
    """The addenda count and below-footer digest a runner of ``stage`` expects: the registry entry when the registry
    exists and holds the stage, else the constants of ``evaluation.discovery`` (``source`` says which).  A registry
    without the stage yields the constants too -- the sealed text may then refuse, which is the correct outcome until the
    orchestrator registers the stage."""
    entry = registered(stage, path)
    if entry is None:
        return {"stage": stage, "n_addenda": int(D.N_ADDENDA_EXPECTED), "below_footer_sha256": D.REGISTERED_ADDENDA_SHA256,
                "source": "constants", "registry_present": read_registry(path) is not None}
    return {"stage": stage, "n_addenda": int(entry["addenda_count"]), "below_footer_sha256": str(entry["below_footer_sha256"]),
            "source": "registry", "registry_present": True, "code_digest": entry.get("code_digest")}


def below_footer_sha256(stage: str, path: Path | None = None) -> str:
    """The below-footer digest a record of ``stage`` is written with (registry, else ``REGISTERED_ADDENDA_SHA256``)."""
    return str(gate_expectations(stage, path)["below_footer_sha256"])


def addenda_count(stage: str, path: Path | None = None) -> int:
    """The addenda count a record of ``stage`` is written with (registry, else ``N_ADDENDA_EXPECTED``)."""
    return int(gate_expectations(stage, path)["n_addenda"])


def discovery_code_digest(path: Path | None = None) -> str:
    """The code digest discovery records are verified with: the registry's ``discovery`` entry when it exists, else the
    runner's live ``current_code_digest()`` (identical until the registry is created)."""
    entry = registered(DISCOVERY, path)
    return live_discovery_code_digest() if entry is None else str(entry["code_digest"])


def code_digest_for(stage: str, *, live: str | None = None, path: Path | None = None) -> str:
    """The code digest a record of ``stage`` is verified with (and, for the runner, written with): the stage's registry
    entry when it exists, else ``live`` when given, else the runner's live ``current_code_digest()`` -- both runner stages
    (:data:`RUNNER_STAGES`) are written by the same script, so the live digest is the right fallback for each."""
    entry = registered(stage, path)
    if entry is not None:
        return str(entry["code_digest"])
    return live_discovery_code_digest() if live is None else str(live)


def record_dir_stage(directory: Path) -> str | None:
    """The registry stage of the record set in one ``evaluation/discovery/<arm>/<design_dir>/s<seed>`` directory, from
    the first readable record's ``job`` block (metadata only; no prediction is read); ``None`` when the directory holds
    no record."""
    d = Path(directory)
    if not d.exists():
        return None
    for js in sorted(d.glob("*.json")):
        rec = D.read_record(js)
        if rec is not None and "job" in rec:
            return job_stage(rec)
    return None


def record_dir_code(out_root: Path, arm: str, design_dir: str, seed: int, *, default: str | None = None,
                    path: Path | None = None) -> str | None:
    """The code digest the record set of ``(arm, design_dir, seed)`` is verified with: :func:`code_digest_for` of the
    stage its records carry (:func:`record_dir_stage`), with ``default`` as the no-registry fallback; ``default`` alone
    when the directory holds no record (the caller then reports ``missing``)."""
    stage = record_dir_stage(D.discovery_root(out_root) / arm / design_dir / f"s{seed}")
    if stage is None:
        return default
    return code_digest_for(stage, live=default, path=path)


def refuse_unless_writable(stage: str, live_code: str, path: Path | None = None) -> dict[str, Any]:
    """The runner's rule before it FITS a job of ``stage`` (addendum 2 item 5, "the code digest under which that stage's
    records were written"): without a registry nothing changes (``ok``, source ``constants``); with one, the stage must be
    registered and its ``code_digest`` must equal the live digest the record would carry -- otherwise ``SystemExit``.
    Hence, once the ``discovery`` entry exists and the closure was patched, no discovery job is ever refitted (a complete
    record set is skipped before this check; an incomplete one surfaces here), and a freezing-candidate job is fitted only
    under the code ``discovery_candidates`` was registered with."""
    body = read_registry(path)
    if body is None:
        return {"ok": True, "stage": stage, "source": "constants", "live_code": live_code}
    entry = registered(stage, path)
    if entry is None:
        raise SystemExit(f"refused: stage {stage!r} is not registered in {registry_path() if path is None else Path(path)} "
                         "and a record of it would be written; register the stage under the current sealed text and this "
                         "code first (python -m gen19ct.evaluation.registry register --stage ...; addendum 2 item 5)")
    if str(entry.get("code_digest")) != str(live_code):
        raise SystemExit(f"refused: the live code digest {str(live_code)[:12]}... is not the one stage {stage!r} was "
                         f"registered with ({str(entry.get('code_digest'))[:12]}...): a record of a registered stage is "
                         "written only under the code the stage was registered with (addendum 2 item 5). For 'discovery' "
                         "this means the run is complete and nothing is refitted (an incomplete job here is an error, not a "
                         "refit); for another stage log the change (log-change) and re-register deliberately (force)")
    return {"ok": True, "stage": stage, "source": "registry", "live_code": live_code, "registered_code": entry["code_digest"]}


def seal_check() -> int:
    """``scripts/g19_seal_prereg.py --check`` as a subprocess (its exit code), as the discovery runner runs it."""
    env = dict(**__import__("os").environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    r = subprocess.run([sys.executable, str(paths.G19_ROOT / "scripts" / "g19_seal_prereg.py"), "--check"],
                       cwd=str(paths.REPO_ROOT), env=env, capture_output=True, text=True)
    return int(r.returncode)


def refuse_unless_sealed(stage: str, check: Callable[[], int] | None = None,
                         digests: Callable[[], Mapping[str, Any]] | None = None, *, expect_addenda: int | None = None,
                         path: Path | None = None) -> dict[str, Any]:
    """The seal gate of ``g19_run_discovery.refuse_unless_sealed`` with the addenda expectations of ``stage``.

    Without a registry FILE the call is delegated to the runner's function unchanged (the constants govern; behaviour
    identical to before).  With a registry the same checks run here against the stage's expectations -- the registry
    entry, or the constants when the stage is not registered yet (the refusal then names THAT stage, not the runner's
    default): ``--check`` exits 0; footer, recomputed and digest-file digests equal ``discovery.REGISTERED_PREREG_SHA256``;
    the addenda count equals the expected one (``expect_addenda`` overrides the count, never the digest); the below-footer
    digest equals the expected one."""
    exp = gate_expectations(stage, path)
    if exp["source"] == "constants" and not exp.get("registry_present"):
        rec = dict(_runner_module().refuse_unless_sealed(check, digests, expect_addenda=expect_addenda))
        rec.update({"stage": stage, "gate_source": "constants"})
        return rec
    where = (f"the registry entry of stage {stage!r}" if exp["source"] == "registry"
             else f"the constants (stage {stage!r} is not registered in the registry)")
    rc = (check or seal_check)()
    if rc != 0:
        raise SystemExit(f"refused: scripts/g19_seal_prereg.py --check exited {rc}; nothing runs on an unsealed or "
                         "modified pre-registration (section 0)")
    try:
        dg = dict((digests or _live_digests)())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"refused: the sealed pre-registration digest could not be read ({exc})") from None
    want = D.REGISTERED_PREREG_SHA256
    bad = {k: dg.get(k) for k in ("footer", "recomputed", "digest_file") if dg.get(k) != want}
    if bad:
        raise SystemExit(f"refused: the sealed pre-registration is not the registered text {want[:12]}... "
                         f"({ {k: (str(v)[:12] + '...') if v else v for k, v in bad.items()} }); a change to the registered "
                         "analysis is a POST-HOC addendum below the footer, never a re-seal (section 0)")
    want_n = int(exp["n_addenda"]) if expect_addenda is None else int(expect_addenda)
    got_n = int(dg.get("n_addenda") or 0)
    if got_n != want_n:
        raise SystemExit(f"refused: the sealed pre-registration carries {got_n} POST-HOC addendum/addenda below the footer, "
                         f"{want_n} expected by {where}; register the stage under the current text "
                         "(gen19ct.evaluation.registry) once the code implements the addenda")
    want_add, got_add = exp["below_footer_sha256"], dg.get("addenda_sha256")
    if got_add != want_add:
        raise SystemExit(f"refused: the POST-HOC addenda below the sealed footer digest to {str(got_add)[:12]}..., not "
                         f"{where} ({str(want_add)[:12]}...): the addendum text was edited or extended after the stage was "
                         "registered; nothing runs against an addendum the registry does not hold (--expect-addenda never "
                         "passes this check)")
    return {"prereg_sha256": want, "addenda_sha256": got_add, "addenda_sha256_registered": want_add, "n_addenda": got_n,
            "n_addenda_expected": want_n, "addendum_implemented": int(exp["n_addenda"]), "seal_check_exit": rc,
            "stage": stage, "gate_source": exp["source"],
            "registry_path": str(registry_path() if path is None else Path(path))}


def _live_digests() -> dict[str, Any]:
    dg = below_footer_digest()
    return {**dg, "addenda_sha256": dg["below_footer_sha256"]}


# --------------------------------------------------------------------------------------------- #
# the code digest each post-discovery stage's records carry (for registering that stage)
# --------------------------------------------------------------------------------------------- #

def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, paths.G19_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stage_code_digest(stage: str) -> str:
    """The ``code_digest`` field the records of ``stage`` carry, computed by that stage's own runner (imported, never
    run): ladder records carry the discovery code digest; H3 / power / figures / report their script's ``code_digest()``
    combined; process its ``code_digests()['own']``; discovery and discovery_candidates the runner's
    ``current_code_digest()`` (the candidates pass is registered from the tree before it runs, module docstring)."""
    if stage in RUNNER_STAGES:
        return live_discovery_code_digest()
    if stage in ("scorer", "ladder"):
        # the scorer verifies discovery records and writes none of its own; ladder records carry the discovery digest
        return str(discovery_code_digest()) if stage == "scorer" else str(_script("g19_run_ladder").ladder_code_digest()["discovery"])
    if stage in ("h3", "power", "figures", "report"):
        name = {"h3": "g19_run_h3", "power": "g19_run_power", "figures": "g19_make_figures", "report": "g19_build_report"}[stage]
        return str(_script(name).code_digest()["combined"])
    if stage == "process":
        return str(_script("g19_run_process").code_digests()["own"])
    if stage == "confirmation":
        # the single registered confirmation run (pre-registration section 15): its records carry the combined digest of
        # the confirmation runner, gen19ct/evaluation/confirmation.py and the discovery code the arms are fitted with,
        # exactly as the h3 stage does -- scripts/g19_run_confirmation.py.code_digest()["combined"]
        return str(_script("g19_run_confirmation").code_digest()["combined"])
    raise ValueError(f"no code digest rule for stage {stage!r}")


#: where :func:`stage_code_digest` takes each stage's code digest from -- recorded in the entry as
#: ``code_digest_source``, because "source: current tree" describes the SEALED TEXT side only and the scorer's and the
#: ladder's code digest is deliberately the ``discovery`` entry's (task X finding V-REG-08)
CODE_DIGEST_SOURCE: dict[str, str] = {
    DISCOVERY: "the completed run's record fields (register_discovery_from_records)",
    CANDIDATES: "the tree: g19_run_discovery.current_code_digest()",
    "scorer": "the registry's 'discovery' entry: the scorer verifies discovery records and writes none of its own, so "
              "its entry carries the digest those records were written under, NOT a digest of the scorer's own code "
              "(editing the scorer therefore changes no entry, and the seal gate does not compare code digests)",
    "ladder": "the registry's 'discovery' entry (g19_run_ladder.ladder_code_digest()['discovery']): a ladder record "
              "carries the discovery digest beside its own 'ladder_code_digest', and the runner compares the live value "
              "with this entry, so the two agree by construction",
    "h3": "the tree: g19_run_h3.code_digest()['combined']", "power": "the tree: g19_run_power.code_digest()['combined']",
    "figures": "the tree: g19_make_figures.code_digest()['combined']",
    "report": "the tree: g19_build_report.code_digest()['combined']",
    "process": "the tree: g19_run_process.code_digests()['own']",
    "confirmation": "the tree: g19_run_confirmation.code_digest()['combined'] (the confirmation runner, "
                    "gen19ct/evaluation/confirmation.py and the discovery code its arms are fitted with)"}


def register_stage_from_tree(stage: str, *, note: str, path: Path | None = None, code_digest: str | None = None,
                             sealed: Path | None = None, sha_file: Path | None = None,
                             force: bool = False) -> dict[str, Any]:
    """Register a post-discovery stage under the CURRENT sealed text (below-footer digest and addenda count from the
    tree) and the code digest its records will carry (:func:`stage_code_digest` unless given).

    The entry records WHERE each of the two digests came from: ``source`` for the sealed text and
    ``code_digest_source`` (:data:`CODE_DIGEST_SOURCE`) for the code digest.

    ``force=True`` re-registers a stage whose digests have changed -- a new POST-HOC addendum below the footer, or an
    edit to the stage's own code before it has run -- and moves the earlier entry to ``superseded``, where
    :func:`verify_record` still finds it for any record written under it."""
    if stage == DISCOVERY:
        raise ValueError("the discovery stage is registered from its records (register_discovery_from_records)")
    dg = below_footer_digest(sealed, sha_file)
    code = stage_code_digest(stage) if code_digest is None else code_digest
    src = ("passed explicitly (--code-digest)" if code_digest is not None
           else CODE_DIGEST_SOURCE.get(stage, "the stage's own runner (stage_code_digest)"))
    return register_stage(stage, below_footer_sha256=dg["below_footer_sha256"], code_digest=code, git_head=_git_head(),
                          addenda_count=dg["n_addenda"], note=note, path=path, force=force,
                          extra={"source": "current tree (below-footer digest and addenda count)",
                                 "code_digest_source": src})


# --------------------------------------------------------------------------------------------- #
# CLI (the orchestrator's command sequence after the discovery run)
# --------------------------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name: str, help_: str) -> argparse.ArgumentParser:
        q = sub.add_parser(name, help=help_)
        q.add_argument("--registry", default=None, help=argparse.SUPPRESS)      # tests: another registry file
        return q

    p = cmd("register-discovery", "register the discovery stage from the completed run's record fields")
    p.add_argument("--out-root", default=str(paths.G19_ROOT))
    p.add_argument("--note", required=True)
    p = cmd("register", "register a post-discovery stage under the current sealed text")
    p.add_argument("--stage", required=True, choices=[s for s in STAGES if s != DISCOVERY])
    p.add_argument("--note", required=True)
    p.add_argument("--code-digest", default=None, help="override the stage's code digest (default: its runner's)")
    p.add_argument("--force", action="store_true",
                   help="re-register a stage whose digests changed (a new POST-HOC addendum, or an edit to the stage's "
                        "own code before it has run); the earlier entry is kept under 'superseded' and every record "
                        "written under it still verifies")
    p = cmd("log-change", "log an edit to an existing module (validates / invalidates nothing)")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--commit", default=None)
    p.add_argument("--reason", required=True)
    p = cmd("verify", "verify one record JSON against its stage's registry entry")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--record", required=True)
    cmd("show", "print the registry")
    cmd("current", "print the current tree's below-footer digest, addenda count and git head")
    ns = ap.parse_args(argv)
    path = None if ns.registry is None else Path(ns.registry)
    if ns.cmd == "register-discovery":
        print(json.dumps(register_discovery_from_records(Path(ns.out_root), note=ns.note, path=path), indent=2))
    elif ns.cmd == "register":
        print(json.dumps(register_stage_from_tree(ns.stage, note=ns.note, path=path, code_digest=ns.code_digest,
                                                  force=bool(ns.force)), indent=2))
    elif ns.cmd == "log-change":
        print(json.dumps(log_code_change(ns.files, ns.commit, ns.reason, path=path), indent=2))
    elif ns.cmd == "verify":
        rec = json.loads(Path(ns.record).read_text(encoding="utf-8"))
        out = verify_record(rec, ns.stage, path=path)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out["ok"] else 1
    elif ns.cmd == "show":
        print(json.dumps(read_registry(path), indent=2))
    elif ns.cmd == "current":
        print(json.dumps(current_digests(with_code=False), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
