"""``scripts/g19_h3_guard_diagnosis.py`` -- WHY the ``ACT_PERMUTED@V1`` legs of H3 are incomplete, measured.

POST-HOC addendum 4 item 2 registers the H3 design scope and says it "is run in full"; the 20 h cap is the only
registered reason a leg may stay incomplete, and it was not reached (5.78 h of 20 h).  Both ``ACT_PERMUTED@V1`` legs
stopped on ``AssertionError: inner split ... failed the isolation check`` at fold ``pub_97510df3a0``.  This script
NAMES the cause instead of forcing the fold: it rebuilds that fold's context exactly as ``g19_run_h3.run_fold`` does,
takes the inner splits the cross-fitted conformal calibration would use, and runs
``gen19ct.data.leakage.fold_isolation_check`` on each of them against BOTH frames (the registered corpus values and the
permuted ones) at BOTH value settings (the registered ``near_dup_value_tol`` = 0.005 and value-blind ``None``).

It FITS NOTHING, writes no fold record and changes no delta; its only output is
``evaluation/h3/decisions/isolation_guard_diagnosis.json``.

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_h3_guard_diagnosis.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen19ct import paths                                                        # noqa: E402
from gen19ct.data import leakage as LK                                           # noqa: E402
from gen19ct.evaluation import discovery as D                                    # noqa: E402
from gen19ct.evaluation import h3 as H3                                          # noqa: E402
from gen19ct.folds import io as FI                                               # noqa: E402
from gen19ct.manifest import git_head, write_json                                # noqa: E402

NAME = "g19_h3_guard_diagnosis"
FOLD = "pub_97510df3a0"


def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _checks(frame: pd.DataFrame, tr: pd.Index, te: pd.Index, tol: float | None) -> dict[str, Any]:
    return LK.fold_isolation_check(tr, te, frame, level="V1", near_dup_sig=FI.NEAR_DUP_SIG,
                                   near_dup_value_tol=tol, raise_on_violation=False)


def diagnose(out_root: Path, *, arm: str = "B5", fold_id: str = FOLD) -> dict[str, Any]:
    """The guard report of every inner split of ``fold_id``, on both frames and at both value settings."""
    rh = _script("g19_run_h3")
    rd = rh.runner_module()
    coext = rd.coextractant_ids()
    state = D.PlanState.read(rd.plan_state_path(out_root))
    corpus = rd.load_corpus(coext)
    code = rh.code_digest()["combined"]
    entry = [e for e in rh.plan_jobs((arm,), state, designs=("V1",))
             if e["transform"] == "ACT_PERMUTED" and e["design"] == "V1"][0]
    job = entry["job"]
    hit = [(f, o) for f, o in corpus.fittable(job) if f.fold_id == fold_id]
    if not hit:
        raise SystemExit(f"refused: {fold_id} is not a fittable fold of {job.key}")
    fold, ordinal = hit[0]
    fc, info = rh.fold_context(entry, fold, ordinal, corpus, out_root, state, code=code)
    wrec = rh.with_record(out_root, entry, fold)
    cal, cplan = H3.frozen_runner(arm).calibration(fc, wrec["arm_record"])
    if cal is None:
        raise SystemExit(f"refused: no calibration object for {job.key}/{fold_id} ({cplan})")
    table, mask, ctx = fc.corpus.table, fc.mask, fc.ctx
    splits = cal.splitter.splits(table, mask, ctx)
    pg_perm = FI.publication_group_frame(fc.corpus.slim)            # the PERMUTED training values
    pg_reg = FI.publication_group_frame(corpus.slim)                # the REGISTERED values

    rows: list[dict[str, Any]] = []
    for sp in splits:
        tri = table.index[np.flatnonzero(sp.train_mask)]
        tei = table.index[np.sort(sp.cal_positions)]
        r: dict[str, Any] = {"inner_split": str(sp.unit), "n_train": int(len(tri)), "n_calibration": int(len(tei))}
        for label, frame in (("permuted", pg_perm), ("registered", pg_reg)):
            for tname, tol in (("near_dup_value_tol_0.005", FI.NEAR_DUP_VALUE_TOL), ("value_blind", None)):
                rep = _checks(frame, tri, tei, tol)
                r[f"{label}/{tname}"] = {"ok": bool(rep["ok"]),
                                         "violations": {k: int(v) for k, v in rep["violations"].items() if v},
                                         "examples": {k: v for k, v in rep["examples"].items() if v}}
        rows.append(r)

    failing = [r for r in rows if not r["permuted/near_dup_value_tol_0.005"]["ok"]]
    flagged: list[dict[str, Any]] = []
    if failing:
        sp = [s for s in splits if str(s.unit) == failing[0]["inner_split"]][0]
        tri = table.index[np.flatnonzero(sp.train_mask)]
        tei = table.index[np.sort(sp.cal_positions)]
        named = set(sum(failing[0]["permuted/near_dup_value_tol_0.005"]["examples"].values(), []))
        kt = LK.near_duplicate_key(pg_perm.loc[tri], sig=FI.NEAR_DUP_SIG)
        ke = LK.near_duplicate_key(pg_perm.loc[tei], sig=FI.NEAR_DUP_SIG)
        for key in sorted(set(kt) & set(ke)):
            kid = LK.key_id([key]).iloc[0]
            if kid not in named:
                continue
            ti, ei = kt.index[kt == key], ke.index[ke == key]
            perm = pd.to_numeric(pg_perm.loc[ti, "log_D"], errors="coerce").to_numpy(float)
            reg = pd.to_numeric(pg_reg.loc[ti, "log_D"], errors="coerce").to_numpy(float)
            cal_p = pd.to_numeric(pg_perm.loc[ei, "log_D"], errors="coerce").to_numpy(float)
            cal_r = pd.to_numeric(pg_reg.loc[ei, "log_D"], errors="coerce").to_numpy(float)
            flagged.append({"key_id": kid, "n_train_rows": int(len(ti)), "n_calibration_rows": int(len(ei)),
                            "train_metal_state": sorted({str(x) for x in pg_perm.loc[ti, "g19_metal_state"]}),
                            "registered_train_logD": [float(x) for x in reg],
                            "permuted_train_logD": [float(x) for x in perm],
                            "min_abs_delta_registered": float(np.min(np.abs(np.subtract.outer(reg, cal_r)))),
                            "min_abs_delta_permuted": float(np.min(np.abs(np.subtract.outer(perm, cal_p)))),
                            "train_logD_changed_by_the_permutation": bool(np.any(perm != reg))})
    return {"schema": H3.SCHEMA, "script": NAME, "git_head": git_head(),
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "job": job.key, "arm": arm, "transform": "ACT_PERMUTED", "design": "V1", "fold_id": fold_id,
            "fold_context": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool)) or v is None},
            "guard": {"level": "V1", "near_dup_sig": FI.NEAR_DUP_SIG,
                      "near_dup_value_tol_registered": FI.NEAR_DUP_VALUE_TOL,
                      "value_independent_levels": ["publication group (a stored column)",
                                                   "archive duplicate_group_id (a stored column)",
                                                   "near_duplicate_key at 6 significant figures (never reads log_D)"]},
            "n_inner_splits": len(rows), "inner_splits": rows,
            "n_failing_permuted_tol": len(failing),
            "n_failing_registered_tol": len([r for r in rows if not r["registered/near_dup_value_tol_0.005"]["ok"]]),
            "n_failing_registered_value_blind": len([r for r in rows if not r["registered/value_blind"]["ok"]]),
            "n_failing_permuted_value_blind": len([r for r in rows if not r["permuted/value_blind"]["ok"]]),
            "flagged_keys_of_the_first_failing_split": flagged,
            "conclusion": H3.GUARD_FAILURE_DIAGNOSIS,
            "action_taken": ("the fold is NOT forced: both ACT_PERMUTED@V1 legs stay unfitted and carry "
                             f"{H3.INCOMPLETE_GUARD_FAILURE} with no verdict until a POST-HOC addendum states how the "
                             "guard reads a value-permuted control arm")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--arm", default="B5")
    ap.add_argument("--fold", default=FOLD)
    ns = ap.parse_args(argv)
    out_root = Path(ns.out_root)
    body = diagnose(out_root, arm=ns.arm, fold_id=ns.fold)
    p = write_json(H3.h3_decisions_dir(out_root) / "isolation_guard_diagnosis.json", H3.json_safe(body))
    print(f"wrote {p}")
    print(f"permuted/tol failing {body['n_failing_permuted_tol']} of {body['n_inner_splits']}; "
          f"registered/tol failing {body['n_failing_registered_tol']}; "
          f"registered/value-blind failing {body['n_failing_registered_value_blind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
