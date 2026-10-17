"""gen20.1 stage S4 helpers: the confirmation seeds and the pre-registration seal.

  seeds  compute the five confirmation split seeds by the PRE_REGISTRATION §0 / plan §6 rule
         s_i = int(sha256("gen20_1-confirmation-seed-{i}")[:8], 16) % 900000 + 100000, i = 1, 2, ...
         skipping the discovery seeds, gen16's confirmation seeds, gen20's confirmation seeds
         (re-derived from gen20's string "gen20-confirmation-seed-{i}" with gen20's own exclusions,
         g20_confirm.py:34-41, never importing that script) and seeds already accepted; the first five,
         sorted.  Writes confirmation_seeds.json (seeds, rule, the label-free design-BP outer + inner fold
         digest on those seeds, gate-(e) form) and confirmation_seeds_sha256.txt (gen20's commitment form:
         sha256 of json.dumps(sorted seeds)).
  seal   S4 preconditions first (S3 passed: counts_g201.json with s3_status PASS, 0 mismatches and every gate
         (a), (b), (b) in subprocesses, (c), (d), (e) true, frozen trees unchanged; every committed confirmation
         seed written in the PRE_REGISTRATION body; the S3 reference files present), then
         gen201.seal.seal(PRE_REGISTRATION.md) (refused while a draft marker remains), and record in
         seal_g201.json the body digest, the LF-normalised digest of IMPLEMENTATION_PLAN.md (O19 default:
         the plan is frozen with the seal by digest), the confirmation-seed commitment and the sha256 of the
         S3 reference files that the S6 re-run of g201_counts.py must reproduce.

Both subcommands install the forbidden-writer guard and snapshot the repository outside generations/gen20_1
(plus the three gen20_1 documents) before and after; any change is a STOP (exit 1).  The seal step alone
exempts PRE_REGISTRATION.md from that snapshot.

Usage (repo root):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B generations/gen20_1/scripts/g201_seal.py seeds [--out DIR]
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B generations/gen20_1/scripts/g201_seal.py seal
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
sys.path.insert(0, str(G201))

from gen201 import bootstrap  # noqa: E402
from gen201 import guard  # noqa: E402
from gen201 import seal as SL  # noqa: E402

guard.install()

DISCOVERY_SEEDS = (104729, 130363, 155921, 196613, 262147)
GEN16_CONFIRMATION_SEEDS = (137171, 506411, 721632, 937310, 939478)
GEN20_CONFIRMATION_SEEDS_DOCUMENTED = (148569, 228670, 253534, 561551, 978178)   # GEN20_REPORT.md:320
GEN20_RUN_ONCE = bootstrap.G20 / "results" / "confirmation" / "RUN_ONCE.json"
RULE = 'int(sha256("gen20_1-confirmation-seed-{i}")[:8], 16) % 900000 + 100000'
N_SEEDS = 5
#: the gates counts_g201.json must report as passed before the seal (plan §4 (a)-(e); (b) also in subprocesses)
REQUIRED_GATES = ("a", "b", "b_subprocess", "c", "d", "e")
#: S3 reference files (relative to the manifests directory) whose sha256 the seal records for the S6 re-run
S3_REFERENCE_FILES = ("fold_plan_g201.json", "counts_g201.json", "row_manifests/DIGESTS.json",
                      "fold_records_g201.parquet", "confirmation_seeds.json")


def _seed(template: str, i: int) -> int:
    return int(hashlib.sha256(template.format(i=i).encode()).hexdigest()[:8], 16) % 900000 + 100000


def gen20_confirmation_seeds(n: int = 5) -> list[int]:
    """gen20's rule (g20_confirm.py:34-41), re-derived: its own exclusions were discovery + gen16."""
    seeds, i = [], 1
    while len(seeds) < n:
        s = _seed("gen20-confirmation-seed-{i}", i)
        if s not in DISCOVERY_SEEDS and s not in GEN16_CONFIRMATION_SEEDS and s not in seeds:
            seeds.append(s)
        i += 1
    return sorted(seeds)


def commitment(seeds) -> str:
    """gen20's commitment form: sha256 of json.dumps(sorted seeds)."""
    return hashlib.sha256(json.dumps(sorted(int(s) for s in seeds)).encode()).hexdigest()


def confirmation_seeds(n: int = N_SEEDS) -> tuple[list[int], list[dict]]:
    gen20 = gen20_confirmation_seeds()
    excluded = set(DISCOVERY_SEEDS) | set(GEN16_CONFIRMATION_SEEDS) | set(gen20)
    seeds, trace, i = [], [], 1
    while len(seeds) < n:
        s = _seed("gen20_1-confirmation-seed-{i}", i)
        why = ("discovery" if s in DISCOVERY_SEEDS else "gen16" if s in GEN16_CONFIRMATION_SEEDS
               else "gen20" if s in gen20 else "duplicate" if s in seeds else None)
        trace.append({"i": i, "seed": s, "skipped": why})
        if why is None:
            seeds.append(s)
        i += 1
    if set(seeds) & excluded:  # pragma: no cover - self-check
        raise AssertionError("an excluded seed was accepted")
    return sorted(seeds), trace


def confirmation_fold_digest(seeds) -> dict:
    """Label-free design-BP outer + inner fold digest on the confirmation seeds (gate-(e) form)."""
    from gen201 import data as D
    from gen201 import folds as FO
    bench = D.load(check=True)
    F = FO.build(bench, seeds=tuple(seeds), designs=("BP",))
    dig = FO.plan_digest(bench, F, designs=("BP",))
    sizes = [{"seed": int(f.seed), "fold": int(f.fold), "n_train": int(len(f.train_index)), "n_test": int(len(f.test_index))}
             for f in F["BP"]]
    skipped = [[r["seed"], r["fold"], r["inner"], r["n_inner_train"]] for r in dig["inner"]["BP"] if r["skipped"]]
    once = FO.tested_once_per_seed(F, len(bench.frame))["BP"]
    return {"design": "BP", "sha256": dig["sha256"], "per_design": dig["per_design"], "definition": dig["definition"],
            "folds": sizes, "skipped_inner_folds": skipped, "each_cell_tested_once_per_seed": once}


def cmd_seeds(out: Path) -> int:
    gen20 = gen20_confirmation_seeds()
    if tuple(gen20) != GEN20_CONFIRMATION_SEEDS_DOCUMENTED:
        print(f"STOP: re-derived gen20 seeds {gen20} != documented {GEN20_CONFIRMATION_SEEDS_DOCUMENTED}")
        return 1
    run_once = json.loads(GEN20_RUN_ONCE.read_text(encoding="utf-8"))
    gen20_commit_ok = run_once.get("seeds_commitment") == commitment(gen20)
    if not gen20_commit_ok:
        print(f"STOP: commitment of the re-derived gen20 seeds {commitment(gen20)} != RUN_ONCE.json "
              f"{run_once.get('seeds_commitment')}")
        return 1
    seeds, trace = confirmation_seeds()
    fold = confirmation_fold_digest(seeds)
    rec = {"rule": RULE, "n": N_SEEDS, "seeds": seeds, "commitment_sha256": commitment(seeds),
           "commitment_form": "sha256(json.dumps(sorted(seeds)))",
           "excluded": {"discovery": list(DISCOVERY_SEEDS), "gen16_confirmation": list(GEN16_CONFIRMATION_SEEDS),
                        "gen20_confirmation_rederived": gen20,
                        "gen20_rederived_equals_documented": True,
                        "gen20_commitment_equals_RUN_ONCE": bool(gen20_commit_ok)},
           "trace": trace, "confirmation_fold_digest_BP": fold}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "confirmation_seeds.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    with open(out / "confirmation_seeds_sha256.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(commitment(seeds) + "\n")
    print(json.dumps({"seeds": seeds, "commitment": commitment(seeds), "confirmation_BP_fold_digest": fold["sha256"],
                      "skipped_inner_folds": fold["skipped_inner_folds"], "out": str(out)}, indent=1))
    return 0


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(bootstrap.ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def s4_preconditions(out: Path, prereg: Path, seeds) -> tuple[list[str], dict]:
    """The S4 preconditions, checked before PRE_REGISTRATION.md is touched.  Returns (problems, record)."""
    problems: list[str] = []
    counts_path = out / "counts_g201.json"
    rec: dict = {"counts_g201": _rel(counts_path)}
    if not counts_path.exists():
        problems.append(f"{counts_path.name} is missing: S3 (g201_counts.py) must pass before the seal")
    else:
        counts = json.loads(counts_path.read_text(encoding="utf-8"))
        gates = counts.get("gates", {})
        rec.update({"s3_status": counts.get("s3_status"), "n_mismatches": counts.get("n_mismatches"),
                    "gates": gates, "frozen_trees_unchanged": counts.get("frozen_trees_unchanged")})
        if counts.get("s3_status") != "PASS":
            problems.append(f"S3 status is {counts.get('s3_status')!r}, not 'PASS'")
        if counts.get("n_mismatches") != 0:
            problems.append(f"S3 reported {counts.get('n_mismatches')!r} count mismatches")
        bad = [g for g in REQUIRED_GATES if gates.get(g) is not True]
        if bad:
            problems.append(f"S3 gates not passed (or not run): {bad}")
        if counts.get("frozen_trees_unchanged") is not True:
            problems.append("S3 did not prove the frozen trees unchanged")
    body = SL.lf_text(prereg)
    missing = [int(x) for x in seeds if str(int(x)) not in body]
    rec["seeds_written_in_preregistration"] = not missing
    if missing:
        problems.append(f"confirmation seeds not written in {prereg.name}: {missing}")
    absent = [n for n in S3_REFERENCE_FILES if not (out / n).is_file()]
    if absent:
        problems.append(f"S3 reference files missing in {out}: {absent}")
    return problems, rec


def cmd_seal(out: Path, prereg: Path, plan: Path) -> int:
    from gen201.env import git_state
    from gen201.run import sha256_file
    seeds_json = out / "confirmation_seeds.json"
    commit_txt = out / "confirmation_seeds_sha256.txt"
    if not seeds_json.exists() or not commit_txt.exists():
        print("STOP: run the 'seeds' subcommand first (S4 commits the confirmation seeds with the seal)")
        return 1
    seeds = json.loads(seeds_json.read_text(encoding="utf-8"))["seeds"]
    if commit_txt.read_text(encoding="utf-8").strip() != commitment(seeds):
        print("STOP: confirmation_seeds_sha256.txt does not match confirmation_seeds.json")
        return 1
    problems, pre = s4_preconditions(out, prereg, seeds)
    if problems:
        print("STOP: S4 preconditions fail; PRE_REGISTRATION.md was not touched")
        for p in problems:
            print("  -", p)
        return 1
    s3_reference = {n: sha256_file(out / n) for n in S3_REFERENCE_FILES}
    plan_state = SL.preregistration_state(plan)
    digest = SL.seal(prereg)                      # raises while a draft marker remains
    st = SL.require_sealed(prereg)
    rec = {"sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git_state(),
           "preregistration": {"path": _rel(prereg),
                               "sha256_lf_body": digest, "valid": st["valid"]},
           "implementation_plan": {"path": _rel(plan),
                                   "sha256_lf_whole_file": SL.lf_digest(plan),
                                   "markers_remaining": plan_state["markers_remaining"],
                                   "note": "O19 default: the plan is frozen with the seal by this digest; later "
                                           "changes only by dated addendum"},
           "confirmation_seeds": {"seeds": seeds, "commitment_sha256": commitment(seeds)},
           "s4_preconditions": pre,
           "s3_reference_sha256": s3_reference,
           "s3_reference_note": "sha256 of the S3 outputs (paths relative to the manifests directory); the S6 re-run "
                                "of g201_counts.py must reproduce fold_plan_g201.json, counts_g201.json, "
                                "row_manifests/DIGESTS.json (which digests every row manifest) and "
                                "fold_records_g201.parquet byte for byte"}
    with open(out / "seal_g201.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    if plan_state["markers_remaining"]:
        print(f"WARNING: IMPLEMENTATION_PLAN.md still carries draft markers {plan_state['markers_remaining']}")
    print(json.dumps(rec, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("seeds")
    s1.add_argument("--out", default=str(bootstrap.MANIFESTS))
    s2 = sub.add_parser("seal")
    s2.add_argument("--out", default=str(bootstrap.MANIFESTS))
    s2.add_argument("--prereg", default=str(bootstrap.PREREG))
    s2.add_argument("--plan", default=str(bootstrap.PLAN))
    a = ap.parse_args()
    out = bootstrap.assert_under_g201(a.out)
    prereg = bootstrap.assert_under_g201(a.prereg) if a.cmd == "seal" else None
    exempt = (prereg,) if prereg is not None else ()        # the one file the seal step may change
    before = guard.snapshot(exclude=exempt)
    rc = cmd_seeds(out) if a.cmd == "seeds" else cmd_seal(out, prereg, Path(a.plan).resolve())
    changed = guard.diff(before, guard.snapshot(exclude=exempt))
    if any(changed.values()):
        print(f"STOP: files outside generations/gen20_1 (or a gen20.1 document) changed: "
              f"{ {k: v[:10] for k, v in changed.items() if v} }")
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
