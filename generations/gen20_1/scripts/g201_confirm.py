"""gen20.1 stage S11 (CONFIRMATION, PRE_REGISTRATION.md §11): run once, design BP, primary + COH-NODMD refit, on
the confirmation seeds committed at S4, then score with the identical procedure ((a)-(g) and (i)-(k) re-evaluated
on the confirmation seeds, (h) taken from discovery) and render the confirmation status beside every verdict.

Order of checks (any failure stops before a model is fitted):
  1. the forbidden-writer guard is installed at import; PRE_REGISTRATION.md must be sealed;
  2. RUN_ONCE: results/confirmation/RUN_ONCE.json must not exist (status "completed" -> refused for ever;
     status "started" -> refused unless --resume, which continues the same, resumable run: same seeds, same fold
     digest, never re-dealt; every resume is appended to the RUN_ONCE record);
  3. the seeds equal manifests/confirmation_seeds.json, whose sorted list hashes (sha256(json.dumps(sorted(seeds))))
     to the S4 commitment a2bb89f7..., equal to manifests/confirmation_seeds_sha256.txt;
  4. fold gates (a)-(d) on the discovery seeds (code-path check) and the confirmation outer+inner fold digest of
     design BP (form of gate (e)) equal to f6d7bd70...;
  5. the discovery report state (results/discovery/report_state.json, S10) exists: (h) is reused from it;
  6. the S8 smoke record, read by the discovery script's own rule (``g201_discovery.smoke_flags``): a learner that
     failed smoke (C, C_PAIR) is "not run" in the confirmation exactly as in discovery (identical procedure, §11);
  7. the S7 anchor gate, by the discovery script's own rule (``g201_discovery.anchors_gate``: status PASS, all four
     anchors reproduced, the run_fold cross-check and the Kish check passed), recorded in the run manifest;
  8. the prediction code digest of the imported code (``fitting.CODE_AT_IMPORT``, equal to the files on disk:
     ``fitting.assert_code_unchanged``) equals discovery's (report_state.json ``code_digest``, cross-checked against
     the discovery BP DONE markers when they exist); it is stored in the RUN_ONCE record and a resume with a
     different digest is refused (a resume never refits the run-once confirmation under other code); the scored
     confirmation tables must carry the same digest.
Then ``fitting.run_design(bench, "BP", seeds=CONF_SEEDS, label="confirmation", variants=("primary", "nodmd"))``,
the scoring (results/confirmation/), and GEN20_1_REPORT.md re-rendered with the confirmation status.

Usage (repo root):  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B generations/gen20_1/scripts/g201_confirm.py
                    [--resume]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
G201 = Path(__file__).resolve().parents[1]
if str(G201) not in sys.path:
    sys.path.insert(0, str(G201))

from gen201 import bootstrap  # noqa: E402
from gen201 import guard  # noqa: E402

guard.install()

from gen201 import report as RP  # noqa: E402
from gen201 import rules as R  # noqa: E402
from gen201.run import Run  # noqa: E402
from gen201.seal import require_sealed  # noqa: E402

LABEL = "confirmation"
CONF_SEEDS: tuple[int, ...] = (215593, 276871, 348239, 385670, 915723)
COMMITMENT_SHA256 = "a2bb89f77fdc4be9f0e7ac186517d9d92cb6e9c406bc9c207bdb2850865f23bd"
FOLD_DIGEST_BP = "f6d7bd70460f6b35da9b60d68723016e3883578800c2352a100b86c043148841"
SEEDS_JSON = bootstrap.MANIFESTS / "confirmation_seeds.json"
SEEDS_SHA_TXT = bootstrap.MANIFESTS / "confirmation_seeds_sha256.txt"
OUT = bootstrap.RESULTS / LABEL
RUN_ONCE = OUT / "RUN_ONCE.json"
DISCOVERY_STATE = bootstrap.RESULTS / "discovery" / "report_state.json"


class ConfirmationRefused(RuntimeError):
    pass


def commitment(seeds) -> str:
    return hashlib.sha256(json.dumps(sorted(int(s) for s in seeds)).encode()).hexdigest()


def check_seeds(path: Path = SEEDS_JSON, sha_txt: Path = SEEDS_SHA_TXT) -> list[int]:
    """The confirmation seeds equal the committed ones (list, commitment, text file, recorded fold digest)."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = sorted(int(s) for s in rec["seeds"])
    if seeds != sorted(CONF_SEEDS):
        raise ConfirmationRefused(f"confirmation seeds {seeds} != the sealed {sorted(CONF_SEEDS)}")
    c = commitment(seeds)
    txt = Path(sha_txt).read_text(encoding="utf-8").strip()
    if not (c == COMMITMENT_SHA256 == rec.get("commitment_sha256") == txt):
        raise ConfirmationRefused(f"seed commitment mismatch: computed {c}, json {rec.get('commitment_sha256')}, "
                                  f"txt {txt}, sealed {COMMITMENT_SHA256}")
    if rec.get("confirmation_fold_digest_BP", {}).get("sha256") != FOLD_DIGEST_BP:
        raise ConfirmationRefused("recorded confirmation fold digest differs from the S4 commitment")
    return seeds


_DISCOVERY_MODULE = None


def discovery_module():
    """scripts/g201_discovery.py loaded as a module (once): its ``smoke_flags`` and ``anchors_gate`` are the rules the
    confirmation applies, so the two stages cannot drift apart."""
    global _DISCOVERY_MODULE
    if _DISCOVERY_MODULE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("g201_discovery_script", G201 / "scripts" / "g201_discovery.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DISCOVERY_MODULE = mod
    return _DISCOVERY_MODULE


def smoke_flags(path: Path | None = None) -> dict:
    """The S8 pass flags by the discovery script's rule (refuses a missing or failed smoke record), so the
    confirmation runs exactly the arms discovery ran."""
    mod = discovery_module()
    return mod.smoke_flags(Path(path) if path is not None else mod.DEFAULT_SMOKE)


def anchors_gate(path: Path | None = None) -> dict:
    """The S7 anchor gate by the discovery script's rule (PRE_REGISTRATION.md §4.4: STOP unless status PASS, all four
    anchors reproduced, the run_fold cross-check and the Kish check passed)."""
    mod = discovery_module()
    return mod.anchors_gate(Path(path) if path is not None else mod.DEFAULT_ANCHORS)


def check_folds(bench, seeds) -> dict:
    """Gates (a)-(d) on the discovery seeds and the confirmation BP outer+inner digest."""
    from gen201 import folds as FO
    disc = FO.build(bench)
    dig = FO.gen16_digest(disc)
    gates = {"a": FO.gate_a(dig), "b": FO.gate_b(dig), "c": FO.gate_c(disc["BP"]), "d": FO.gate_d(disc)}
    conf = FO.build(bench, seeds=tuple(seeds), designs=("BP",))
    plan = FO.plan_digest(bench, conf, designs=("BP",))
    gates["confirmation_digest"] = {"pass": plan["per_design"]["BP"] == FOLD_DIGEST_BP,
                                    "sha256": plan["per_design"]["BP"], "expected": FOLD_DIGEST_BP}
    bad = [k for k, v in gates.items() if not v["pass"]]
    if bad:
        raise FO.FoldGateError(f"INCOMPARABLE_FOLDS: gates failing {bad}")
    return {k: {"pass": v["pass"]} for k, v in gates.items()}


RESUME_KEYS: tuple[str, ...] = ("seeds", "commitment_sha256", "fold_digest_BP", "code_digest")


def discovery_code_digest(disc_state: dict, *, root: Path | None = None) -> str:
    """The prediction code digest of the discovery run: report_state.json's ``code_digest`` (S10), cross-checked
    against the discovery BP DONE markers when they exist; refused when it cannot be established or disagrees."""
    stated = RP._digest_str(disc_state.get("code_digest"))
    markers = set()
    for v in ("primary", "nodmd"):
        markers |= RP.unit_digests("discovery", "BP", v, root=root)
    if len(markers) > 1 or "missing" in markers or "unreadable" in markers:
        raise ConfirmationRefused(f"the discovery BP units name several or no code digests: {sorted(markers)}")
    if stated and markers and stated not in markers:
        raise ConfirmationRefused(f"discovery report_state.json code digest {stated} differs from its DONE markers "
                                  f"{sorted(markers)}")
    dig = stated or (next(iter(markers)) if markers else None)
    if not dig:
        raise ConfirmationRefused("the discovery prediction code digest cannot be established (no code_digest in "
                                  "results/discovery/report_state.json and no discovery BP DONE marker)")
    return dig


def check_code_digest(discovery: str, current: str) -> str:
    """§11 'the identical procedure': the confirmation is fitted by the code that produced discovery."""
    if current != discovery:
        raise ConfirmationRefused(f"prediction code digest {current} differs from discovery's {discovery}: the "
                                  "confirmation must run the identical procedure (PRE_REGISTRATION §11)")
    return current


def run_once_open(path: Path = RUN_ONCE, *, resume: bool = False, record: dict | None = None) -> dict:
    """RUN_ONCE guard: refuse a completed run for ever, refuse a started run unless ``resume``; write the record.
    A resume must match the started run's seeds, commitment, fold digest and code digest (``RESUME_KEYS``)."""
    path = bootstrap.assert_under_g201(path)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") == "completed":
            raise ConfirmationRefused(f"{path.name}: the confirmation already ran ({rec.get('completed_utc')}); "
                                      "it runs exactly once")
        if not resume:
            raise ConfirmationRefused(f"{path.name}: a confirmation run was started ({rec.get('started_utc')}) and not "
                                      "completed; pass --resume to continue that same run")

        def norm(k, v):
            return RP._digest_str(v) if k == "code_digest" else v

        bad = [k for k, v in (record or {}).items() if k in RESUME_KEYS and norm(k, rec.get(k)) != norm(k, v)]
        if bad:
            raise ConfirmationRefused(f"resume refused: {', '.join(bad)} differ from the started run (seeds, "
                                      "commitment, fold digest and code digest must be identical)")
        rec.setdefault("resumes", []).append(now)
    else:
        rec = {"status": "started", "started_utc": now, **(record or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
    return rec


def run_once_close(path: Path = RUN_ONCE, **extra) -> dict:
    path = bootstrap.assert_under_g201(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec.update({"status": "completed", "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})
    path.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8", newline="\n")
    return rec


def confirmation_status(disc: dict, conf: dict) -> dict:
    """§11: H confirmed only if PASS on both seed sets (H4: the same verdict); T final only if equal on both."""
    out = {"label": R.CONFIRMATION_LABEL, "feature_set": R.FEATURE_LABEL, "H": {}, "T": {}}
    for h in ("H1", "H2", "H3", "H4"):
        out["H"][h] = R.confirm_h(h, disc["registered"][h], conf["registered"][h])
    for tid in ("T1", "T1w", "T2", "T3"):
        for tau in map(str, R.TAUS):
            dc = (((disc.get("thresholds") or {}).get(tid) or {}).get("classes") or {}).get(tau, {}).get("class")
            cc = (((conf.get("thresholds") or {}).get(tid) or {}).get("classes") or {}).get(tau, {}).get("class")
            out["T"][f"{tid}@{tau}"] = R.confirm_t(dc, cc)
    return out


def score_confirmation(disc_state: dict, *, scorer_kwargs: dict | None = None, log=print) -> dict:
    """Score the confirmation tables (design BP only) with (h) from the discovery state."""
    h_from = RP.h_points_from_json(disc_state)
    sc = RP.Scorer(LABEL, designs=("BP",), log=log, **(scorer_kwargs or {}))
    return sc.score(h_from=h_from, full=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    require_sealed(bootstrap.PREREG)
    seeds = check_seeds()
    if not DISCOVERY_STATE.exists():
        raise ConfirmationRefused("S10 first: results/discovery/report_state.json is missing")
    disc_state = json.loads(DISCOVERY_STATE.read_text(encoding="utf-8"))
    smoke = smoke_flags()
    anchors = anchors_gate()
    before = guard.snapshot()
    with Run("g201_confirm", label=LABEL, args=vars(args), seeds=seeds) as run:
        run.record("smoke", smoke)
        run.record("anchors_gate", anchors)
        from gen201 import data as D
        from gen201 import fitting
        if RUN_ONCE.exists() and json.loads(RUN_ONCE.read_text(encoding="utf-8")).get("status") == "completed":
            run_once_open(resume=False)                    # raises: the confirmation runs exactly once
        fitting.assert_code_unchanged()                    # CodeChanged when the files on disk differ from the import
        code = fitting.CODE_AT_IMPORT
        disc_digest = check_code_digest(discovery_code_digest(disc_state), code["sha256"])
        run.record("code_digest", {"sha256": code["sha256"], "source": "fitting.CODE_AT_IMPORT",
                                   "equals_discovery": True})
        bench = D.load(check=True)
        run.record("fold_gates", check_folds(bench, seeds))
        rec = {"seeds": seeds, "commitment_sha256": COMMITMENT_SHA256, "fold_digest_BP": FOLD_DIGEST_BP,
               "design": "BP", "variants": ["primary", "nodmd"], "code_digest": code["sha256"],
               "code_files": code["files"]}
        run_once_open(resume=args.resume, record=rec)
        run.record("run_design", fitting.run_design(bench, "BP", seeds=seeds, label=LABEL,
                                                    variants=("primary", "nodmd"), resume=True,
                                                    include_c=smoke["C"], include_c_pair=smoke["C_PAIR"]))
        st = score_confirmation(disc_state, log=run.log)
        if st.get("code_digest") != disc_digest:
            raise ConfirmationRefused(f"the scored confirmation tables carry code digest {st.get('code_digest')}, "
                                      f"discovery {disc_digest}")
        paths = RP.write_results(st, OUT)
        status = confirmation_status(disc_state, RP._clean(st))
        p = OUT / "confirmation_status.json"
        p.write_text(json.dumps(RP._clean(status), indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
        md = RP.render(disc_state, confirmation=RP._clean(st))
        with open(bootstrap.assert_under_g201(RP.REPORT_MD), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md)
        run.record("confirmation_status", RP._clean(status))
        run.record("outputs", [str(x) for x in paths + [p, RP.REPORT_MD]])
        changed = guard.diff(before, guard.snapshot())
        run.record("frozen_trees_changed", changed)
        if any(changed.values()):
            raise AssertionError(f"frozen trees changed during S11: { {k: v[:10] for k, v in changed.items() if v} }")
        run_once_close(status_summary={h: v["status"] for h, v in status["H"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
