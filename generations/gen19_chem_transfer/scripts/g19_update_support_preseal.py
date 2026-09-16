"""``scripts/g19_update_support_preseal.py`` -- recompute the pre-seal ``support_score`` under the section 13 s4 reading
resolved by the orchestrator on 2026-09-15 (pre-registration section 13, s4 row).

Why
---
The pre-seal run (``scripts/g19_run_preseal.py``) wrote ``support_score`` only for V5-primary and V2: on V1 and V0 the
registration was silent on whether the query's own system counts in s4, so it wrote "not computed" with the ambiguous-row
counts.  The resolution -- the own system counts when it has training rows of the query's metal state; a system with an
undefined ``d_desc`` is skipped -- is ``gen19ct.evaluation.support.s4_registered``.

What it does (support features only: no model is fitted, fitted or scored, nothing runs on V6)
------------------------------------------------------------------------------------------------
For every support job (V5__primary, V2__element, V1__copy, V0__rows) and every outer fold of its pre-seal support file:
the fold's training mask is rebuilt from the registered fold file (MODEL rows minus the fold's hidden rows, as the
pre-seal run built it), the ``d_desc`` standardisation over the fold's training systems is taken from the same engine
function the pre-seal run used (``models.baselines.LookupEngine._d_desc_scale``; descriptor tables only, no target), s4
is recomputed with ``support.s4_registered`` for every scored (metal state, system), and ``support_score`` is the mean
of the stored s1-s3, s5-s8 components and the recomputed s4 (``support.support_score``).  Checks: the recomputed s4
equals the stored pre-seal s4 on every row; V5-primary and V2 ``support_score`` values equal the pre-seal
``support_score_candidate`` (under V5 the own system never counts; any change is reported, not hidden); the recounted
domain-status table equals the existing one.

Since task X (2026-09-15, finding VR-01) the update is also a library step of ``scripts/g19_run_preseal.py``:
:func:`apply_support_update` takes the pre-seal run's as-run status and domain-status counts and returns the resolved
ones, so a ``--skip-compute`` or full rerun of the pre-seal script reproduces these outputs instead of overwriting them
with the as-run "not computed" labels.  Run on its own, this script is idempotent: applied to its own outputs it
rewrites them byte-identically (the support_score columns of the counts table are replaced, not duplicated), and its
manifest then carries the earlier run's ``pre_update_sha256`` (:func:`pre_update_record`) and the current code digests.

Outputs (``generations/gen19_chem_transfer/``)
    evaluation/preseal/support/<job>__support_score.parquet   per (fold, scored row): support_s4, support_score, flags
    evaluation/preseal/support_status.json                     support_score computed for every job, with the checks
    evaluation/preseal/domain_status_counts.csv                the same counts plus support_score mean / median
    manifests/g19_update_support_preseal.json (+ run_info)

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_update_support_preseal.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.data import load  # noqa: E402
from gen19ct.evaluation import support as ES  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json  # noqa: E402
from gen19ct.models import baselines as B  # noqa: E402
from gen19ct.models import interface as I  # noqa: E402

NAME = "g19_update_support_preseal"
#: support job -> registered fold file stem (the pre-seal run's jobs with support=True)
JOB_STEMS: dict[str, str] = {"V5__primary": "V5__primary__exact", "V1__copy": "V1__copy__exact",
                             "V0__rows": "V0__rows__random5", "V2__element": "V2__element__exact"}
#: jobs whose support_score the pre-seal run already computed (must be unchanged)
PRESEAL_COMPUTED = ("V5__primary", "V2__element")
COMPONENTS = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")
HALF_LABEL = {"S": "selection", "C": "confirmation", "NA": "none"}
SCORE_COUNT_COLUMNS = ("support_score_mean", "support_score_median")
READING = ("support_score v1 (section 13) computed on every support job under the s4 reading resolved by the "
           "orchestrator on 2026-09-15 (gen19ct.evaluation.support.s4_registered): the query's own system counts when it "
           "has training rows of the query's metal state; a system with an undefined d_desc (or fingerprint) is skipped. "
           "Recomputed by scripts/g19_update_support_preseal.py from the registered folds and the pre-seal support "
           "components (no model fitted). The n_rows_s4_ambiguous_* counts are the pre-seal run's: rows where the two "
           "unresolved readings differed. domain_status is unchanged (it does not read s4).")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return paths.rel(Path(p))


def score_file(ev: Path, job: str) -> Path:
    """Where the per-row registered support_score of a job is written."""
    return Path(ev) / "support" / f"{job}__support_score.parquet"


def row_table() -> tuple[pd.DataFrame, I.RowTable, pd.Series, pd.DataFrame, pd.DataFrame]:
    """``(MODEL rows, RowTable, row-id -> label map, systems, components)`` as the pre-seal workers build them."""
    model = load.load_model_rows()
    fr = I.prepare_frame(model)
    systems, comps = I.load_descriptor_tables()
    t = I.RowTable(fr, systems=systems, components=comps)
    idmap = pd.Series(fr.index, index=fr[I.ID_COL].astype(str))
    return model, t, idmap, systems, comps


def recompute_job(job: str, t: I.RowTable, idmap: pd.Series, ev: Path, systems: pd.DataFrame,
                  comps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per scored row of the job's pre-seal support file: the registered s4 and support_score, with the checks."""
    t0 = time.perf_counter()
    sp = pd.read_parquet(ev / "support" / f"{job}.parquet")
    folds = {f.fold_id: f for f in FI.read_design(JOB_STEMS[job])}
    missing = sorted(set(sp["fold_id"]) - set(folds))
    if missing:
        raise AssertionError(f"{job}: support folds absent from {JOB_STEMS[job]}: {missing[:5]}")
    out = []
    for fold_id, rows in sp.groupby("fold_id", sort=True):
        f = folds[fold_id]
        hid = pd.Index(idmap.loc[list(f.hidden_row_ids)].to_numpy())
        mask = np.ones(t.n, dtype=bool)
        mask[t.positions(hid)] = False
        pos = t.positions(pd.Index(idmap.loc[rows["row_id"].astype(str)].to_numpy()))
        if mask[pos].any():
            raise AssertionError(f"{job}/{fold_id}: a scored row is a training row")
        ctx = I.FitContext(systems=systems, components=comps, table=t, hidden_index=hid)
        mu, sd = B.make_engine(t, mask, ctx)._d_desc_scale(B._BASE)
        cache: dict[tuple[str, str], tuple[float, bool, float]] = {}
        s4 = np.empty(len(rows))
        own = np.zeros(len(rows), dtype=bool)
        s4_excl = np.empty(len(rows))
        for k, p in enumerate(pos):
            m, s = t.state_labels[t.state[p]], t.sys_labels[t.sys[p]]
            if (m, s) not in cache:
                mc = t.state_code[m]
                cands = [(t.sys_labels[c], t.sys_fp[c], (t.sys_desc[c] - mu) / sd, int(mask[pp].sum()))
                         for c, pp in t.systems_by_state.get(mc, [])]
                st = t.static(s)
                qz = (st["desc"] - mu) / sd
                v = ES.s4_registered(s, st["fp"], qz, cands)
                v_excl = ES.s4_registered(s, st["fp"], qz, [c for c in cands if c[0] != s])
                own_counts = any(c[0] == s and c[3] > 0 for c in cands)
                cache[(m, s)] = (v, own_counts, v_excl)
            s4[k], own[k], s4_excl[k] = cache[(m, s)]
        comp = rows[[f"support_{c}" for c in COMPONENTS]].to_numpy(dtype=float).copy()
        comp[:, COMPONENTS.index("s4")] = s4
        score = np.array([ES.support_score(dict(zip(COMPONENTS, r))) for r in comp])
        out.append(pd.DataFrame({"row_id": rows["row_id"].to_numpy(), "fold_id": fold_id,
                                 "seed": rows["seed"].to_numpy(), "support_s4": s4, "support_score": score,
                                 "s4_query_system_counts": own, "s4_changed_by_query_system": s4 != s4_excl,
                                 "preseal_support_s4": rows["support_s4"].to_numpy(dtype=float),
                                 "preseal_support_score_candidate": rows["support_score_candidate"].to_numpy(dtype=float)}))
    res = pd.concat(out, ignore_index=True).sort_values(["fold_id", "row_id"], kind="mergesort").reset_index(drop=True)
    d_s4 = np.abs(res["support_s4"] - res["preseal_support_s4"]).to_numpy()
    d_sc = np.abs(res["support_score"] - res["preseal_support_score_candidate"]).to_numpy()
    amb = sp.set_index(["fold_id", "row_id"])["support_s4_ambiguous_query_system_inclusion"]
    changed = res.set_index(["fold_id", "row_id"])["s4_changed_by_query_system"]
    checks = {
        "n_rows": int(len(res)), "n_folds": int(res["fold_id"].nunique()),
        "s4_recomputed_equals_preseal_s4": bool((d_s4 == 0).all()), "max_abs_s4_difference": float(d_s4.max()),
        "support_score_equals_preseal_candidate": bool((d_sc == 0).all()),
        "max_abs_support_score_difference": float(d_sc.max()), "n_rows_support_score_changed": int((d_sc != 0).sum()),
        "n_rows_query_system_counts_in_s4": int(res["s4_query_system_counts"].sum()),
        "n_rows_s4_changed_by_query_system_inclusion": int(res["s4_changed_by_query_system"].sum()),
        "query_system_flags_equal_preseal_ambiguity_flags": bool(
            changed.sort_index().to_numpy().tolist() == amb.reindex(changed.sort_index().index).to_numpy().tolist()),
        "support_score_nonfinite_rows": int((~np.isfinite(res["support_score"])).sum()),
        "support_score_summary": {"mean": float(res["support_score"].mean()), "median": float(res["support_score"].median()),
                                  "min": float(res["support_score"].min()), "max": float(res["support_score"].max())},
        "runtime_s": round(time.perf_counter() - t0, 1)}
    return res, checks


def apply_support_update(status: Mapping, counts: pd.DataFrame, jobs: Sequence[str], *, ev: Path, t: I.RowTable,
                         idmap: pd.Series, systems: pd.DataFrame, comps: pd.DataFrame, model: pd.DataFrame,
                         preds: Mapping[str, pd.DataFrame] | None = None,
                         say: Callable[[str], None] = log) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """The resolved support status of ``jobs``: ``(status, domain-status counts, per-job score frames, checks)``.

    ``status`` / ``counts`` are the pre-seal run's as-run ``support_status.json`` content and ``domain_status_counts.csv``
    table (or this function's own earlier outputs: the update is idempotent).  ``ev`` holds the stored pre-seal support
    parquets; ``preds`` (job -> prediction frame with ``fold_id, row_id, arm, half``) defaults to the stored predictions.
    Nothing is written here."""
    status = json.loads(json.dumps(status))
    base_counts = counts.drop(columns=[c for c in SCORE_COUNT_COLUMNS if c in counts.columns])
    scores, all_checks = {}, {}
    for job in jobs:
        say(f"{job}: recomputing s4 and support_score")
        res, checks = recompute_job(job, t, idmap, ev, systems, comps)
        if job in PRESEAL_COMPUTED and not checks["support_score_equals_preseal_candidate"]:
            say(f"{job}: support_score CHANGED on {checks['n_rows_support_score_changed']} rows "
                f"(max |diff| {checks['max_abs_support_score_difference']:.3g}) -- reported")
        if not checks["s4_recomputed_equals_preseal_s4"]:
            say(f"{job}: recomputed s4 differs from the pre-seal s4 (max |diff| {checks['max_abs_s4_difference']:.3g})")
        if checks["support_score_nonfinite_rows"]:
            raise AssertionError(f"{job}: non-finite support_score")
        scores[job] = res
        all_checks[job] = {k: v for k, v in checks.items() if k != "runtime_s"}      # the manifest is deterministic
        rec = status["jobs"][job]
        rec.update({"support_score": "computed (s4 reading resolved 2026-09-15, section 13)",
                    "support_score_file": rel(score_file(paths.G19_ROOT / "evaluation" / "preseal", job)),
                    **{k: v for k, v in checks.items() if k != "runtime_s"}})
        say(f"{job}: {checks['n_rows']} rows, {checks['n_folds']} folds, s4 equal {checks['s4_recomputed_equals_preseal_s4']}, "
            f"score equal to candidate {checks['support_score_equals_preseal_candidate']}, {checks['runtime_s']} s")
    status["readings"] = READING
    status["s4_reading"] = ES.S4_READING
    status["updated_by"] = f"scripts/{NAME}.py"
    # domain-status counts: recount exactly as the pre-seal run, check equality, add support_score mean / median
    parts = []
    cell = model.set_index(model[I.ID_COL].astype(str))[["g19_metal_state", "extractant_system_key"]]
    for job in jobs:
        sp = pd.read_parquet(ev / "support" / f"{job}.parquet", columns=["fold_id", "row_id", "domain_status_candidate"])
        pr = (preds[job][["fold_id", "row_id", "arm", "half"]] if preds is not None and job in preds
              else pd.read_parquet(ev / "predictions" / f"{job}.parquet", columns=["fold_id", "row_id", "arm", "half"]))
        h = pr[pr["arm"] == "B0"][["fold_id", "row_id", "half"]]
        d = sp.merge(h, on=["fold_id", "row_id"], how="left").merge(
            scores[job][["fold_id", "row_id", "support_score"]], on=["fold_id", "row_id"], how="left")
        if d["half"].isna().any() or d["support_score"].isna().any():
            raise AssertionError(f"{job}: support rows without a half or a support_score")
        d = d.join(cell, on="row_id")
        g = d.groupby(["half", "domain_status_candidate"])
        c = g.agg(n_row_entries=("row_id", "size"), support_score_mean=("support_score", "mean"),
                  support_score_median=("support_score", "median")).reset_index()
        c["n_cells"] = [int(sub[["g19_metal_state", "extractant_system_key"]].drop_duplicates().shape[0])
                        for _, sub in g]
        c.insert(0, "job", job)
        c["half"] = c["half"].map(HALF_LABEL)
        parts.append(c.rename(columns={"domain_status_candidate": "domain_status"}))
    new = pd.concat(parts, ignore_index=True)
    key = ["job", "half", "domain_status"]
    merged = base_counts[base_counts["job"].isin(jobs)].merge(new, on=key, how="outer", suffixes=("", "_recount"),
                                                              indicator=True)
    same = bool((merged["_merge"] == "both").all()
                and (merged["n_row_entries"] == merged["n_row_entries_recount"]).all()
                and (merged["n_cells"] == merged["n_cells_recount"]).all())
    if not same:
        raise AssertionError("recounted domain-status table differs from the pre-seal one")
    out_counts = base_counts.merge(new[key + list(SCORE_COUNT_COLUMNS)], on=key, how="left")
    return status, out_counts, scores, all_checks


def pre_update_record(manifest_path: Path, on_disk: Mapping[str, str]) -> dict:
    """``pre_update_sha256`` = the digests of the as-run ``support_status.json`` / ``domain_status_counts.csv`` this update
    replaced (``g19_run_preseal.py`` compares them with its as-run rendering: ``support_update.
    as_run_status_equals_update_manifest_pre_update_sha256``).  When the files on disk are this script's own recorded
    outputs (an idempotent re-run, e.g. to record the current code digest), the earlier manifest's ``pre_update_sha256``
    is carried, so the link to the as-run status survives, and the on-disk digests are recorded beside it."""
    if Path(manifest_path).exists():
        prev = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        outs = {o["path"]: o.get("sha256") for o in prev.get("outputs", [])}
        carried = prev.get("pre_update_sha256")
        if carried and set(carried) == set(on_disk) and all(outs.get(k) == v for k, v in on_disk.items()):
            return {"pre_update_sha256": carried, "pre_update_sha256_carried_from_earlier_run": True,
                    "on_disk_sha256_before_this_run": dict(on_disk)}
    return {"pre_update_sha256": dict(on_disk), "pre_update_sha256_carried_from_earlier_run": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", default=",".join(JOB_STEMS), help="comma list of support jobs")
    ns = ap.parse_args(argv)
    jobs = [j for j in ns.jobs.split(",") if j]
    bad = [j for j in jobs if j not in JOB_STEMS]
    if bad:
        raise SystemExit(f"unknown jobs {bad}")
    t_start = time.perf_counter()
    ev = paths.G19_ROOT / "evaluation" / "preseal"
    status_p, counts_p = ev / "support_status.json", ev / "domain_status_counts.csv"
    pre_sha = {rel(p): sha256(p) for p in (status_p, counts_p)}
    code_sha = {rel(p): sha256(p) for p in (Path(__file__), Path(ES.__file__))}     # taken before the run, as imported
    pre = pre_update_record(paths.MANIFESTS_DIR / f"{NAME}.json", pre_sha)
    with Run(NAME, args=vars(ns), seed=None, extra={**pre, "s4_reading": ES.S4_READING}) as run:
        run.inputs(paths.ARCHIVE_MASTER, FI.PUB_COMPONENTS_CSV, paths.DESCRIPTORS_DIR / "extractant_systems.csv",
                   paths.DESCRIPTORS_DIR / "extractant_components.csv",
                   *[paths.FOLDS_DIR / f"{JOB_STEMS[j]}.{e}" for j in jobs for e in ("json", "parquet")],
                   *[ev / "support" / f"{j}.parquet" for j in jobs], *[ev / "predictions" / f"{j}.parquet" for j in jobs])
        log("MODEL rows, row table")
        model, t, idmap, systems, comps = row_table()
        status = json.loads(status_p.read_text(encoding="utf-8"))
        old_counts = pd.read_csv(counts_p)
        status, out_counts, scores, all_checks = apply_support_update(
            status, old_counts, jobs, ev=ev, t=t, idmap=idmap, systems=systems, comps=comps, model=model)
        outputs = []
        for job in jobs:
            p = score_file(ev, job)
            scores[job].to_parquet(p, index=False, compression="zstd")
            outputs.append(p)
        write_csv(out_counts, counts_p)
        write_json(status_p, status)
        outputs += [status_p, counts_p]
        run.outputs(*outputs)
        run.extra.update({"checks": all_checks, "domain_status_counts_recount_identical": True,
                          "learned_arms_fitted_or_scored": [], "models_fitted": [], "code_sha256": code_sha})
    log(f"done in {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
