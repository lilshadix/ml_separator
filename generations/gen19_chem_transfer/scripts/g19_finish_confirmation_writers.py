"""``scripts/g19_finish_confirmation_writers.py`` -- complete the writer stage of the SPENT confirmation run, once.

POST-HOC addendum 9 item 2.  The single run of section 15 fitted every fold and wrote
``evaluation/confirmation/decisions/confirmation.json`` and ``tables/confirmation_*.csv``, then refused before
``decisions/CONFIRMATION.md`` under the leak scan's earlier any-substring reading (24 record JSONs held a seed's six
digits inside a float mantissa; addendum 9 item 1).  The run is spent, so the runner is not re-invoked; this script
runs the run's OWN writer stage (``g19_run_confirmation.write_outputs``) exactly once on the decision body as the run
wrote it, and asserts that the body differs afterwards only by the ``seed_leak_scan`` block that stage adds.

Gates, each refusing with what is missing:
* ``decisions/confirmation.json`` exists (the run finished and scored) and ``decisions/CONFIRMATION.md`` does NOT
  (this step is once-only too);
* the sealed text verifies and the registry's ``confirmation`` stage is registered at the live code
  (``registry.refuse_unless_sealed`` / ``refuse_unless_writable``);
* the seed store verifies against the section 15 commitment.

It re-scores nothing, refits nothing and adds no claim.  ``CONFIRMATION.md`` is written last and alone names the five
seeds (addendum 6 item 4); the post-write scan over this script's own manifest runs as the runner's does.

Run from the repository root, ONCE, with the store path only the user holds:
    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer .venv/Scripts/python.exe \\
        generations/gen19_chem_transfer/scripts/g19_finish_confirmation_writers.py --seed-store <PATH> --expect-addenda 9
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import confirmation as CF  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.manifest import Run, git_head  # noqa: E402

NAME = "g19_finish_confirmation_writers"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    import datetime as _dt
    print(f"[{_dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed-store", required=True, help="the withheld-seed store (outside the repository)")
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--expect-addenda", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    dec, rp = CF.decisions_path(out_root), CF.report_path(out_root)
    if not dec.exists():
        print(f"refused: {dec} does not exist -- the single run has not finished and scored")
        return 2
    if rp.exists():
        print(f"refused: {rp} already exists -- the writer stage has been completed once; this step is once-only")
        return 2
    rc = _load("g19_run_confirmation")
    seal = _load("g19_seal_prereg")
    REG.refuse_unless_sealed(CF.STAGE, None, None, expect_addenda=ns.expect_addenda)
    code = rc.code_digest()["combined"]
    REG.refuse_unless_writable(CF.STAGE, code)
    store = CF.load_seed_store(ns.seed_store, root=out_root, seal=seal,
                               prereg_paths=seal.PreregPaths(root=out_root, repo_root=paths.REPO_ROOT))
    if not store.verified or store.digest != CF.committed_digest(out_root):
        print("refused: the seed store does not verify against manifests/confirmation_seeds_sha256.txt")
        return 2
    before = paths.digests(dec)["sha256"]
    body: dict[str, Any] = json.loads(dec.read_text(encoding="utf-8"))
    original = copy.deepcopy(body)
    log(f"decision file {before[:12]}... ({len(body.get('claims') or [])} claims; s1 {body.get('s1', {}).get('verdict')}, "
        f"s2 {body.get('s2', {}).get('verdict')}); completing the writer stage under code {code[:12]}...")
    written: list[Path] = []
    with Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=None,
             extra={"git_head": git_head(), "code_sha256": code, "decision_file_sha256_before": before,
                    "reading": CF.READINGS["seed_leak_scan_token"], "addendum": "POST-HOC addendum 9 items 1-2"}) as run:
        written = rc.write_outputs(out_root, body, store=store, seal=seal)
        after_body = json.loads(dec.read_text(encoding="utf-8"))
        added = sorted(set(after_body) - set(original))
        changed = sorted(k for k in original if after_body.get(k) != original[k])
        if added != ["seed_leak_scan"] or changed:
            raise SystemExit(f"refused: the decision body changed beyond the seed_leak_scan block (added {added}, "
                             f"changed {changed}); nothing was meant to be re-scored")
        run.outputs(*written)
        run.extra.update({"decision_file_sha256_after": paths.digests(dec)["sha256"], "added_keys": added,
                          "changed_keys": changed, "outputs": [str(p) for p in written],
                          "seed_leak_scan": after_body["seed_leak_scan"]})
    mfs = [paths.MANIFESTS_DIR / f"{NAME}.json", paths.MANIFESTS_DIR / "run_info" / f"{NAME}.json", CF.started_path(out_root)]
    final = CF.scan_for_seed_leak(store, [p for p in [*mfs, *written] if Path(p).exists()], allow=[rp], require=[rp])
    if not final["ok"]:
        raise SystemExit(f"refused: a withheld seed reached {final['files_containing_a_withheld_seed']} after the writers "
                         f"(revelation incomplete: {final['revelation_incomplete']})")
    log(f"post-write scan: {final['files_scanned']} file(s) including this step's manifest; only decisions/CONFIRMATION.md "
        f"names the seeds; written {[paths.rel(p) for p in written]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
