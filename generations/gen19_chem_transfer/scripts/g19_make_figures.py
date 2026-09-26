"""``scripts/g19_make_figures.py`` -- brief section 25 figures 7-13 (``gen19ct.evaluation.figures``) from the discovery,
ladder, power and confirmation FILES, after discovery is complete.

Gate (refuses to start unless ALL hold; ``--check-only`` prints the verdict):

1. ``scripts/g19_seal_prereg.py --check`` exits 0 AND the sealed text is the registered digest with POST-HOC addendum 1
   (``g19_run_discovery.refuse_unless_sealed``);
2. the discovery run is **COMPLETE** (``gen19ct.evaluation.h3.discovery_complete``: ``evaluation/discovery/decisions/
   wall_clock.json`` records an invocation whose ``stages_done`` reached the final plan stage ``10_not_implemented``, every
   stage of the current plan is done, and every ``fit`` job of ``discovery.enumerate_plan(plan_state)`` has a verified
   COMPLETE record set -- digest, fold hash and fold set exactly what the current code, fold files and plan state produce);
3. the scorer's decision files exist and ``decisions.json`` is newer than every fold record
   (``h3.scorer_decisions_present``): the deployed predictor is read from them (``h3.deployed_configuration``).

Frames (assembled here, drawn by ``figures.py``; every figure reports the files it used):

* F07 / F09: the deployed predictor and the V5 lookup comparator B3i on V5-primary, V1 and V2 (selection half, seed
  104729) through the scorer's verified ``Store`` (``g19_score_discovery.Store``; stale records raise);
* F08: per-cell MAE of the deployed predictor (and B3i) on V5-primary joined to the runner's ``_support`` files (domain
  status, registered ``support_score``), with the S1(e) statistic (Spearman, system-cluster bootstrap, 10,000 resamples,
  seed 19) written to ``tables/s1e_error_vs_support.csv`` for the report;
* F10 / F11: the learned ``e_m`` / ``e_l`` tables.  The registered source is ``evaluation/power/embeddings/`` (written by
  ``g19_run_power.py --include-learned``, which is NOT_IMPLEMENTED: ``evaluation/power/reliability.json`` reports the
  embeddings' section 8 reliability NOT_RUN).  Until it exists the tables are REPRODUCED from the verified M1 / M2
  discovery records (``refit_embeddings_from_records``): every V5-primary fold record of the arm (seed 104729) is
  refitted on that fold's training rows at its retained configuration, stopping count and model seed, accepted only when
  the refit's ``model_state_digest`` and ``feature_state_digest`` equal the record's (``arm_record.fit_record``), and
  its ``FactorisedArm.metal_embeddings()`` / ``system_embeddings()`` are written under
  ``evaluation/figures/embeddings/<arm>/``.  The reference table is the lowest-ordinal verified fold; the other verified
  folds are the Procrustes replicates (``power.embedding_stability``) -- **fold replicates (leave-cells-out perturbations),
  not a bootstrap**, labelled so on the figure and in ``stability.json``.  No test row is predicted, no confirmation-half
  row is scored and no V6 row is read by the refits: they reproduce fits discovery already made.  ``--no-embedding-refits``
  reads the tables already on disk instead;
* F12: the confirmation run's ``evaluation/confirmation/v6_rows.csv`` and ``v6_pairs.csv`` only (V6 is touched once);
  while the confirmation run has not run the figure is SKIPPED with the reason ``confirmation run not run``;
* F13: V5-PAIR pair-level direction from the verified M2 V5-PAIR record and ``folds/V5PAIR__primary__batched__pairs.parquet``;
  the V6 panel comes from the confirmation files and is skipped with the same reason while the run has not run.

Outputs (new directories only): ``figures/F07..F13_*.png``, ``figures/data/F*.csv``, ``tables/s1e_error_vs_support.csv``,
``evaluation/figures/embeddings/<arm>/*``, ``evaluation/figures/figures_index.json``; manifest
``manifests/g19_make_figures.json`` (git HEAD, prereg + addendum digests, the code digest of this script and ``figures.py``,
seeds, runtime, the embedding refit verification).  Nothing reads the confirmation half of any design as a test set
except the confirmation files themselves, and nothing touches V6 outside them.

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_make_figures.py --check-only
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_make_figures.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import figures as FG  # noqa: E402
from gen19ct.evaluation import h3 as H3  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.evaluation import transfer as ET  # noqa: E402
from gen19ct.manifest import Run, git_head, write_csv, write_json  # noqa: E402

NAME = "g19_make_figures"
#: the registry stage of this runner (addendum 2 item 5: the seal gate and the discovery code digest prefer
#: manifests/digest_registry.json when it exists and fall back to the constants of evaluation.discovery otherwise)
STAGE = "figures"
CODE_FILES: tuple[Path, ...] = (paths.G19_ROOT / "gen19ct" / "evaluation" / "figures.py", Path(__file__).resolve())
EMBEDDINGS_DIR = "evaluation/power/embeddings"
#: the reproduced tables (module docstring): ``<EMBEDDINGS_REFIT_DIR>/<arm>/{metal,system}_embeddings.csv``,
#: ``*_replicate_<fold>.csv``, ``stability.json``, ``verification.json``
EMBEDDINGS_REFIT_DIR = "evaluation/figures/embeddings"
#: the arms whose records hold a learned ``e_m`` / ``e_l`` (``models.neural.FactorisedArm``); M2 is the H1 candidate and is
#: the arm F10 / F11 draw, M1 is written as data beside it.  The deployed M0 (= B5, CatBoost) has no embedding.
EMBEDDING_ARMS: tuple[str, ...] = ("M2", "M1")
EMBEDDING_REPLICATE_LABEL = "fold-replicate"
CONFIRMATION_DIR = "evaluation/confirmation"
CONFIRMATION_NOT_RUN = "confirmation run not run"
PAIRS_PARQUET = "folds/V5PAIR__primary__batched__pairs.parquet"
ALL_FIGURES: tuple[str, ...] = ("F07", "F08", "F09", "F10", "F11", "F12", "F13")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def runner_module():
    return _load("g19_run_discovery")


def scorer_module():
    return _load("g19_score_discovery")


def code_digest() -> dict[str, Any]:
    return D.code_digest(CODE_FILES)


# ============================================================================================= #
# the gate
# ============================================================================================= #

def refuse_unless_ready(out_root: Path, *, check: Callable[[], int] | None = None,
                        digests: Callable[[], Mapping[str, Any]] | None = None, expect_addenda: int | None = None,
                        excluded_ids: Sequence[str] = (), folds_dir: Path | None = None,
                        discovery_code: str | None = None, runners: Mapping[str, Any] | None = None,
                        jobs: Sequence[D.JobSpec] | None = None, state: D.PlanState | None = None,
                        require_scorer: bool = True, require_ladder: bool = True) -> dict[str, Any]:
    """The gates of the module docstring; raises ``SystemExit`` naming what is missing.  Gate 4 (``require_ladder``):
    the ladder has run every step M3-M7 to a done / skipped status (``h3.ladder_complete``), because the deployed
    predictor the figures are drawn for is 'the retained ladder configuration' (task X finding V-01)."""
    rd = runner_module()
    prereg = REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=expect_addenda)
    st = state if state is not None else D.PlanState.read(rd.plan_state_path(out_root))
    code = discovery_code if discovery_code is not None else REG.discovery_code_digest()
    done = H3.discovery_complete(out_root, code=code, state=st, excluded_ids=excluded_ids, folds_dir=folds_dir,
                                 runners=runners, jobs=jobs)
    if not done["complete"]:
        raise SystemExit("refused: the discovery run is not COMPLETE (reached final stage "
                         f"{done['final_stage']}: {done['reached_final_stage']}; missing stages {done['missing_stages']}; "
                         f"{done['n_incomplete']} of {done['n_fit_record_sets']} fit record sets incomplete). Figures 7-9 and 13 "
                         "read the discovery records, so nothing is drawn while a job is still running")
    sc = H3.scorer_decisions_present(out_root)
    if require_scorer and not sc["ok"]:
        raise SystemExit("refused: the scorer's decision files are missing or older than a fold record "
                         f"(present: {sc['present']}); the deployed predictor is read from decisions.json. Run "
                         "scripts/g19_score_discovery.py, then rerun")
    lc = H3.ladder_complete(H3.read_ladder_state(out_root))
    if require_ladder and not lc["complete"]:
        raise SystemExit("refused: the ladder (M3-M7) is not complete -- the deployed predictor is 'the retained ladder "
                         f"configuration' ({H3.LADDER_STATE_FILE} {'absent' if not lc['present'] else 'steps not done: ' + str(lc['not_done'])}). "
                         "Run scripts/g19_run_ladder.py to completion, then rerun")
    return {"prereg_gate": prereg, "discovery_complete": done, "scorer_decisions": sc, "ladder_complete": lc,
            "plan_state": st.record(), "discovery_code_sha256": code}


# ============================================================================================= #
# frames from files
# ============================================================================================= #

def deployed_arm(out_root: Path) -> dict[str, Any]:
    body = D.read_record(D.discovery_root(out_root) / "decisions" / "decisions.json")
    if body is None:
        return {"arm": None, "basis": "decisions.json absent"}
    return H3.deployed_configuration(body, H3.read_ladder_state(out_root))


def _frame_with_class(store, arm: str, design: str, seed: int | None) -> pd.DataFrame | None:
    fr = store.frame(arm, design, "primary", seed)
    if fr is None:
        return None
    out = fr[["fold_id", "mean_logD", EM.Y_COL]].copy()
    out["log_D"] = pd.to_numeric(out[EM.Y_COL], errors="coerce")
    out["metal_class"] = fr["metal_class"].astype(str) if "metal_class" in fr.columns else "other"
    out["unit"] = fr["unit"].astype(str) if "unit" in fr.columns else fr["fold_id"].astype(str)
    for lvl in (0.5, 0.8, 0.95):
        lo, hi = EM.interval_columns(lvl)
        out[lo] = pd.to_numeric(fr[lo], errors="coerce") if lo in fr.columns else np.nan
        out[hi] = pd.to_numeric(fr[hi], errors="coerce") if hi in fr.columns else np.nan
    out["row_id"] = fr.index.astype(str)
    return out


def prediction_frames(store, arms: Sequence[str], *, designs: Sequence[str] = ("V5", "V1", "V2")
                      ) -> tuple[dict[str, dict[str, pd.DataFrame]], list[str]]:
    """``frames[design][arm]`` of the selection half (learned arms seed 104729) and the record directories read."""
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    inputs: list[str] = []
    for design in designs:
        for arm in arms:
            seed = None if arm in scorer_module().DETERMINISTIC_ARMS else D.PRIMARY_SEED
            fr = _frame_with_class(store, arm, design, seed)
            if fr is None:
                continue
            frames.setdefault(design, {})[arm] = fr
            if arm in scorer_module().DETERMINISTIC_ARMS:
                job = scorer_module().PRESEAL_JOB.get((design, "primary"))
                if job:
                    inputs.append(f"evaluation/preseal/predictions/{job[0]}.parquet")
            else:
                dd = store.design_dir(arm, design, "primary")
                inputs.append(f"evaluation/discovery/{arm}/{dd}/s{seed}/")
    return frames, inputs


def support_labels(store, out_root: Path, arm: str) -> tuple[pd.DataFrame | None, str | None]:
    """The runner's per-fold ``_support`` files of the arm's V5-primary job (domain status, support_score), verified."""
    SD = scorer_module()
    dd = store.design_dir(arm, "V5", "primary")
    seed = None if arm in SD.DETERMINISTIC_ARMS else D.PRIMARY_SEED
    if seed is None:                                   # closed-form comparator: the deployed arm's support files apply
        return None, None
    sup_dir = D.discovery_root(out_root) / "_support" / str(dd) / f"s{seed}"
    if not sup_dir.exists() or not SD._support_files_current(store, arm, str(dd), seed, sup_dir):
        return None, None
    sup = pd.concat([pd.read_parquet(p, columns=["row_id", "fold_id", "domain_status", "domain_status_ambiguous",
                                                 "support_score"]) for p in sorted(sup_dir.glob("*.parquet"))],
                    ignore_index=True)
    return sup, f"evaluation/discovery/_support/{dd}/s{seed}/"


def cell_table(fr: pd.DataFrame, sup: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Per hidden cell: MAE, the cell's support_score (mean over its rows), its domain status (majority), metal class.

    The labels are joined on (fold_id, row_id); when that key matches nothing -- the closed-form comparator B3i is scored
    on the pre-seal EXACT folds while the ``_support`` files are the deployed arm's batched folds (``support_labels``:
    "the deployed arm's support files apply") -- they are joined on row_id alone, which is unique in a V5 design (every
    row is hidden in exactly one fold), and ``support_join`` records which key was used."""
    key = sup.set_index(["fold_id", "row_id"])
    idx = pd.MultiIndex.from_arrays([fr["fold_id"].astype(str), fr["row_id"].astype(str)])
    lab = key.reindex(idx)
    join = "fold_id, row_id"
    if not pd.to_numeric(lab["support_score"], errors="coerce").notna().any():
        by_row = sup.assign(row_id=sup["row_id"].astype(str)).drop_duplicates("row_id").set_index("row_id")
        lab = by_row.reindex(fr["row_id"].astype(str))
        join = "row_id (the deployed arm's support files, closed-form comparator on other folds)"
    f = fr.assign(domain_status=lab["domain_status"].to_numpy(), support_score=pd.to_numeric(lab["support_score"], errors="coerce").to_numpy(),
                  ambiguous=lab["domain_status_ambiguous"].astype("boolean").fillna(False).to_numpy())
    f = f[~f["ambiguous"].astype(bool)]
    f["err"] = (f["mean_logD"].astype(float) - f["log_D"].astype(float)).abs()
    sysk = attrs[EM.SYSTEM_COL].reindex(f["row_id"].to_numpy()).astype(str).to_numpy()
    f["system"] = sysk
    # observed=True: ``unit`` may be categorical, and an unused category would otherwise give an EMPTY group whose
    # majority status has no argmax (the first real run failed here); a unit without a status is labelled, not dropped
    g = f.groupby(f["unit"].astype(str).to_numpy(), observed=True)
    out = g.agg(mae=("err", "mean"), support_score=("support_score", "mean"), n_rows=("err", "size"),
                system=("system", "first"), metal_class=("metal_class", "first"))
    out["domain_status"] = g["domain_status"].agg(
        lambda s: s.dropna().astype(str).value_counts().idxmax() if s.notna().any() else "UNKNOWN")
    out.index.name = "unit"
    out = out.reset_index()
    out.attrs["support_join"] = join
    return out


def v5pair_pairs(store, attrs: pd.DataFrame, out_root: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Pair-level observed / predicted logSF of M2 on the seed-104729 batched V5-PAIR folds (selection half rows)."""
    pq = paths.G19_ROOT / PAIRS_PARQUET if Path(out_root).resolve() == paths.G19_ROOT.resolve() else Path(out_root) / PAIRS_PARQUET
    if not pq.exists():
        return None, []
    pred = store.verified("M2", "V5PAIR__primary_batched", D.PRIMARY_SEED)
    if pred is None:
        return None, []
    bp = pd.read_parquet(pq)
    key = pred.set_index([pred["fold_id"].astype(str), pred["row_id"].astype(str)])["mean_logD"]
    ia = pd.MultiIndex.from_arrays([bp["fold_id"].astype(str), bp["row_id_a"].astype(str)])
    ib = pd.MultiIndex.from_arrays([bp["fold_id"].astype(str), bp["row_id_b"].astype(str)])
    pa, pb = key.reindex(ia).to_numpy(dtype=float), key.reindex(ib).to_numpy(dtype=float)
    y = pd.to_numeric(attrs[EM.Y_COL], errors="coerce")
    ya, yb = y.reindex(bp["row_id_a"].astype(str)).to_numpy(dtype=float), y.reindex(bp["row_id_b"].astype(str)).to_numpy(dtype=float)
    ok = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(ya) & np.isfinite(yb)
    pairs = pd.DataFrame({"fold_id": bp["fold_id"].astype(str), "system": bp[EM.SYSTEM_COL].astype(str) if EM.SYSTEM_COL in bp.columns
                          else attrs[EM.SYSTEM_COL].reindex(bp["row_id_a"].astype(str)).astype(str).to_numpy(),
                          "observed_logsf": ya - yb, "predicted_logsf": pa - pb,
                          "category_class": bp["category_class"].astype(str) if "category_class" in bp.columns else ""})[ok]
    return pairs.reset_index(drop=True), [PAIRS_PARQUET, f"evaluation/discovery/M2/V5PAIR__primary_batched/s{D.PRIMARY_SEED}/"]


def _read_csv(p: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(p, **kw) if p.exists() else None


def embedding_inputs(out_root: Path, arm: str | None = None) -> dict[str, Any]:
    """The ``e_m`` / ``e_l`` tables F10 / F11 draw: the registered ``evaluation/power/embeddings/`` when it exists, else the
    tables reproduced from ``arm``'s verified discovery records under ``evaluation/figures/embeddings/<arm>/``
    (:func:`refit_embeddings_from_records`), with their replicates and stability record.  ``source`` names which."""
    d = Path(out_root) / EMBEDDINGS_DIR
    source = "registered (evaluation/power/embeddings)"
    if not (d / "metal_embeddings.csv").exists() and arm is not None:
        d = Path(out_root) / EMBEDDINGS_REFIT_DIR / arm
        source = f"reproduced from the verified {arm} discovery records ({EMBEDDINGS_REFIT_DIR}/{arm})"
    metal = _read_csv(d / "metal_embeddings.csv", index_col=0)
    system = _read_csv(d / "system_embeddings.csv", index_col=0)
    reps = [pd.read_csv(p, index_col=0) for p in sorted(d.glob("metal_embeddings_replicate_*.csv"))]
    stability = None
    rel = D.read_record(Path(out_root) / "evaluation" / "power" / "reliability.json")
    if rel:
        for r in rel.get("records") or []:
            if r.get("quantity") == "embeddings" and r.get("status") == "computed":
                stability = r
    local = D.read_record(d / "stability.json")
    if stability is None and local and (local.get("metal") or {}).get("status") == "computed":
        stability = dict(local["metal"])
    try:
        rel_dir = d.relative_to(Path(out_root)).as_posix()
    except ValueError:
        rel_dir = str(d)
    used = [f"{rel_dir}/{p.name}" for p in sorted(d.glob("*.csv")) + sorted(d.glob("*.json"))] if d.exists() else []
    if rel:
        used.append("evaluation/power/reliability.json")
    return {"metal": metal, "system": system, "replicates": reps, "stability": stability, "inputs": used, "source": source,
            "dir": d}


def _embedding_columns(table: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in table.columns if str(c).startswith(prefix)]


def refit_embeddings_from_records(out_root: Path, store, state: D.PlanState, *, arms: Sequence[str] = EMBEDDING_ARMS,
                                  code: str | None = None, corpus=None) -> dict[str, Any]:
    """Reproduce the learned embeddings of ``arms`` from their verified V5-primary discovery records (module docstring).

    Per arm: the record set must verify (``Store.verified``; a stale record raises, an incomplete set is skipped); every
    fittable fold's record gives the retained configuration (``arm_record.selected``), stopping count (``n_epochs``) and
    ``model_seed``; the fold's outer training rows come from discovery's own ``prepare_fold`` (section 2 guards, the
    hidden cells removed); the refit is accepted only when its ``model_state_digest`` and ``feature_state_digest`` equal
    the record's.  Writes ``metal_embeddings.csv`` / ``system_embeddings.csv`` (the lowest-ordinal verified fold),
    ``*_replicate_<fold>.csv`` for the other verified folds, ``stability.json`` (``power.embedding_stability`` of the
    ``e_m_*`` and ``e_l_*`` columns; fold replicates, not a bootstrap) and ``verification.json``.  Nothing is predicted."""
    from gen19ct.evaluation import power as PW
    from gen19ct.models import neural as NN

    rd = runner_module()
    code = code or REG.discovery_code_digest()
    if corpus is None:
        log("corpus for the embedding refits (discovery's load_corpus)")
        corpus = rd.load_corpus(rd.coextractant_ids(), with_cv=True)
    fam = None
    if corpus.systems is not None and "system_family" in corpus.systems.columns:
        fam = corpus.systems["system_family"].astype(str)
    out: dict[str, Any] = {"schema": "gen19.figures.embedding_refits.v1", "arms": {}, "outputs": [],
                           "rows_predicted": 0, "confirmation_half_scored": False, "v6_target_rows_scored": 0,
                           "replicates": f"{EMBEDDING_REPLICATE_LABEL}: the other verified V5-primary outer-fold refits of "
                                         "the same arm and seed (leave-cells-out perturbations of the training set), NOT a "
                                         "bootstrap; the registered section 8 embedding reliability "
                                         "(g19_run_power.py --include-learned) is NOT_RUN",
                           "acceptance": "a refit enters the tables only when its model_state_digest and "
                                         "feature_state_digest equal the record's arm_record.fit_record"}
    for arm in arms:
        dd = store.design_dir(arm, "V5", "primary")
        rec_dir = D.discovery_root(out_root) / arm / str(dd) / f"s{D.PRIMARY_SEED}" if dd else None
        summary: dict[str, Any] = {"arm": arm, "design_dir": dd, "seed": D.PRIMARY_SEED, "folds": []}
        out["arms"][arm] = summary
        if dd is None or not rec_dir.exists():
            summary["status"] = f"skipped: no {arm} V5-primary record directory"
            continue
        if store.verified(arm, dd, D.PRIMARY_SEED) is None:
            summary["status"] = "skipped: record set incomplete"
            continue
        recs = {}
        for js in sorted(rec_dir.glob("*.json")):
            rec = D.read_record(js)
            if rec and "fold_id" in rec:
                recs[str(rec["fold_id"])] = rec
        job = D.job_from_record(next(iter(recs.values())))
        summary["job"] = job.key
        tables: list[tuple[int, str, pd.DataFrame, pd.DataFrame]] = []
        t_arm = time.perf_counter()
        for fold, k in corpus.fittable(job):
            rec = recs.get(fold.fold_id)
            row: dict[str, Any] = {"fold_id": fold.fold_id, "ordinal": int(k)}
            if rec is None:
                row["status"] = "record missing"
                summary["folds"].append(row)
                continue
            fc, info = rd.prepare_fold(job, fold, k, corpus, out_root, state, code=code)
            rows = corpus.frame.loc[corpus.table.index[fc.mask]]
            ar = rec["arm_record"]
            sel = ar["selected"]
            cfg = NN.NeuralConfig(int(sel["emb_dim"]), float(sel["weight_decay"]), int(sel["rank"]))
            t0 = time.perf_counter()
            a = NN.FactorisedArm(cfg, n_epochs=int(ar["n_epochs"]), model_seed=int(rec["model_seed"]), rows=corpus.frame,
                                 condition_vectors=corpus.cv).fit(rows)
            got, want = a.fit_record(), ar["fit_record"]
            ok = (got["model_state_digest"] == want["model_state_digest"]
                  and got["feature_state_digest"] == want["feature_state_digest"])
            row.update({"status": "verified" if ok else "digest mismatch (not used)", "n_train": int(info["n_train"]),
                        "n_train_record": rec.get("n_train"), "config": cfg.label(), "n_epochs": int(ar["n_epochs"]),
                        "model_seed": int(rec["model_seed"]), "model_state_digest": got["model_state_digest"],
                        "model_state_digest_record": want["model_state_digest"],
                        "feature_state_digest_matches": got["feature_state_digest"] == want["feature_state_digest"],
                        "refit_seconds": round(time.perf_counter() - t0, 2), "record_digest": rec.get("digest")})
            summary["folds"].append(row)
            log(f"{arm} {fold.fold_id}: {row['status']} ({row['refit_seconds']} s)")
            if ok:
                tables.append((int(k), fold.fold_id, a.metal_embeddings(), a.system_embeddings()))
        summary["seconds"] = round(time.perf_counter() - t_arm, 1)
        summary["n_folds"] = len(summary["folds"])
        summary["n_verified"] = len(tables)
        if not tables:
            summary["status"] = "skipped: no fold refit reproduced its record digest"
            continue
        tables.sort(key=lambda t: t[0])
        odir = Path(out_root) / EMBEDDINGS_REFIT_DIR / arm
        k0, f0, m0, s0 = tables[0]
        if fam is not None:
            s0 = s0.join(fam.rename("family"), how="left")
        m0 = m0.assign(fold_id=f0, fold_ordinal=k0)
        s0 = s0.assign(fold_id=f0, fold_ordinal=k0)
        # write_csv drops the index, so the state / system key becomes the first column (embedding_inputs reads index_col=0)
        outs = [write_csv(m0.reset_index(), odir / "metal_embeddings.csv"),
                write_csv(s0.reset_index(), odir / "system_embeddings.csv")]
        mreps, sreps = [], []
        for k, fid, mt, st in tables[1:]:
            name = D.safe_fold_name(fid)
            outs.append(write_csv(mt.assign(fold_id=fid, fold_ordinal=k).reset_index(),
                                  odir / f"metal_embeddings_replicate_{name}.csv"))
            outs.append(write_csv(st.assign(fold_id=fid, fold_ordinal=k).reset_index(),
                                  odir / f"system_embeddings_replicate_{name}.csv"))
            mreps.append(mt[_embedding_columns(mt, "e_m_")])
            sreps.append(st[_embedding_columns(st, "e_l_")])
        stab = {"metal": PW.embedding_stability(m0[_embedding_columns(m0, "e_m_")], mreps),
                "system": PW.embedding_stability(s0[_embedding_columns(s0, "e_l_")], sreps),
                "replicate_label": EMBEDDING_REPLICATE_LABEL, "replicates": out["replicates"], "arm": arm,
                "reference_fold": f0, "reference_fold_ordinal": k0, "n_replicates": len(mreps),
                "registered_reliability": "NOT_RUN (evaluation/power/reliability.json, quantity embeddings): this record is "
                                          "descriptive and enters no registered rule"}
        for k in ("metal", "system"):
            stab[k]["registered_reliability"] = stab["registered_reliability"]
            stab[k]["replicate_label"] = EMBEDDING_REPLICATE_LABEL
        outs.append(write_json(odir / "stability.json", H3.json_safe(stab)))
        summary.update({"status": "written", "reference_fold": f0, "n_replicates": len(mreps),
                        "n_replicates_aligned_metal": stab["metal"].get("n_replicates"),
                        "metal_stability": stab["metal"].get("stability"),
                        "metal_null_stability": stab["metal"].get("null_stability"), "metal_gate": stab["metal"].get("gate"),
                        "system_stability": stab["system"].get("stability"),
                        "system_null_stability": stab["system"].get("null_stability"),
                        "system_gate": stab["system"].get("gate"), "dir": f"{EMBEDDINGS_REFIT_DIR}/{arm}"})
        outs.append(write_json(odir / "verification.json", H3.json_safe({**summary, "acceptance": out["acceptance"]})))
        out["outputs"] += [str(o) for o in outs]
    return out


def confirmation_inputs(out_root: Path) -> dict[str, Any]:
    """The confirmation run's V6 files, and whether the single run has run at all (``confirmation.decisions_path`` /
    ``started_path``): while it has not, F12 and the V6 panel of F13 are skipped with ``CONFIRMATION_NOT_RUN``."""
    from gen19ct.evaluation import confirmation as CF

    d = Path(out_root) / CONFIRMATION_DIR
    rows, pairs = _read_csv(d / "v6_rows.csv"), _read_csv(d / "v6_pairs.csv")
    # the run's own per-system medians and sign verdicts (scripts/g19_export_confirmation_views.py): F12's right panel
    systems = _read_csv(d / "v6_systems.csv")
    used = [f"{CONFIRMATION_DIR}/{n}" for n, x in (("v6_rows.csv", rows), ("v6_pairs.csv", pairs), ("v6_systems.csv", systems))
            if x is not None]
    decided, started = CF.decisions_path(out_root).exists(), CF.started_path(out_root).exists()
    missing = [f"{CONFIRMATION_DIR}/{n}" for n, x in (("v6_rows.csv", rows), ("v6_pairs.csv", pairs)) if x is None]
    if not decided and not started:
        reason = (f"{CONFIRMATION_NOT_RUN} (input missing: {', '.join(missing)}; the single V6 run of section 3.4 happens "
                  "only inside the confirmation run, whose once-only lock is unspent)") if missing else ""
    else:
        reason = f"input missing: {', '.join(missing)}" if missing else ""
    return {"rows": rows, "pairs": pairs, "systems": systems, "inputs": used, "run_decided": decided,
            "run_started": started, "skip_reason": reason}


# ============================================================================================= #
# main
# ============================================================================================= #

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--only", default=None, help="comma list of figures (F07,...,F13)")
    ap.add_argument("--check-only", action="store_true", help="run the gate and exit")
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--n-resamples", type=int, default=ET.N_RESAMPLES, help="S1(e) bootstrap resamples (registered 10,000)")
    ap.add_argument("--no-embedding-refits", action="store_true",
                    help="F10 / F11 read the embedding tables already on disk instead of reproducing them from the records")
    ap.add_argument("--embedding-arms", default=",".join(EMBEDDING_ARMS), help="arms whose embeddings are reproduced")
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def make_all(out_root: Path, *, only: Sequence[str] | None = None, n_resamples: int = ET.N_RESAMPLES,
             refit_embeddings: bool = True, embedding_arms: Sequence[str] = EMBEDDING_ARMS
             ) -> tuple[list[FG.FigureResult], list[Path], dict[str, Any]]:
    """Every figure of ``only`` (default all); returns the results, the files written and the side records (the
    embedding refit verification)."""
    rd, SD = runner_module(), scorer_module()
    want = set(only or ALL_FIGURES)
    figures_dir = Path(out_root) / "figures"
    tables = Path(out_root) / "tables"
    dep = deployed_arm(out_root)
    side: dict[str, Any] = {}
    # the records, prediction frames and _support files of the deployed configuration are written under the name
    # discovery.ARM_ALIASES gives it (M0 -> B5), so every LOOKUP below takes arm_alias; dep["arm"] stays the name a
    # figure caption and the skip reasons print (task X finding V-P02)
    arm, shown_arm = dep.get("arm_alias") or dep.get("arm"), dep.get("arm")
    results: list[FG.FigureResult] = []
    outs: list[Path] = []
    need_store = (bool(want & {"F07", "F08", "F09", "F13"}) and arm is not None) or \
        (bool(want & {"F10", "F11"}) and refit_embeddings)
    store = attrs = None
    if need_store:
        log("row attributes and the verified prediction store")
        attrs = SD.build_attrs()
        v6 = attrs["v6_target_row"].astype(bool)
        crossings = pd.read_csv(SD.CROSSINGS_CSV, dtype={"row_id": str, "partner_id": str})
        state = D.PlanState.read(rd.plan_state_path(out_root))
        store = SD.Store(out_root, attrs, v6, crossings, state)
    elif arm is None and want & {"F07", "F08", "F09", "F13"}:
        for f in sorted(want & {"F07", "F08", "F09", "F13"}):
            results.append(FG.skipped(f, f"deployed predictor undecidable ({dep.get('basis')})", ["evaluation/discovery/decisions/decisions.json"]))
    if store is not None:
        arms = [arm, "B3i"] if arm != "B3i" else ["B3i"]
        frames, inputs = prediction_frames(store, arms)
        inputs.append("evaluation/discovery/decisions/decisions.json")
        if "F07" in want:
            results.append(FG.fig07_pred_vs_measured(frames, figures_dir, inputs=inputs, deployed=arm))
        if "F09" in want:
            rows_by_design: dict[str, pd.DataFrame] = {}
            f09_inputs = list(inputs)
            for design, fr in ((d, frames.get(d, {}).get(arm)) for d in ("V5", "V1", "V2")):
                if fr is None or not np.isfinite(fr["lower_80"].to_numpy(dtype=float)).all():
                    continue
                if design == "V5":
                    sup, sp = support_labels(store, out_root, arm)
                    if sup is not None:
                        key = sup.set_index(["fold_id", "row_id"])["domain_status"]
                        idx = pd.MultiIndex.from_arrays([fr["fold_id"].astype(str), fr["row_id"].astype(str)])
                        fr = fr.assign(domain_status=key.reindex(idx).astype(str).to_numpy())
                        f09_inputs.append(sp)
                rows_by_design[design] = fr
            results.append(FG.fig09_calibration(rows_by_design, figures_dir, inputs=f09_inputs, arm=arm))
        if "F08" in want:
            cells_by_arm: dict[str, pd.DataFrame] = {}
            stats: dict[str, dict[str, Any]] = {}
            f08_inputs = list(inputs)
            sup, sp = support_labels(store, out_root, arm)
            if sup is not None:
                f08_inputs.append(sp)
                for a in arms:
                    fr = frames.get("V5", {}).get(a)
                    if fr is None:
                        continue
                    cells = cell_table(fr, sup, attrs)
                    cells_by_arm[a] = cells
                    stats[a] = FG.error_vs_support_stats(cells, n_resamples=n_resamples)
                rows = [{"arm": a, "design": "V5", "half": "selection", "seed": D.PRIMARY_SEED if a not in SD.DETERMINISTIC_ARMS else "deterministic",
                         **st} for a, st in stats.items()]
                outs.append(write_csv(pd.DataFrame(rows), tables / "s1e_error_vs_support.csv"))
                results.append(FG.fig08_error_vs_support(cells_by_arm, figures_dir, inputs=f08_inputs, stats=stats))
            else:
                results.append(FG.skipped("F08", f"no current _support files for the deployed {shown_arm} (records "
                                                 f"under {arm}) on V5-primary: input missing "
                                                 f"evaluation/discovery/_support/<V5-primary>/s{D.PRIMARY_SEED}/",
                                          f08_inputs))
        if "F13" in want:
            pairs_by_design: dict[str, pd.DataFrame] = {}
            f13_inputs = ["evaluation/discovery/decisions/decisions.json"]
            pairs, used = v5pair_pairs(store, attrs, out_root)
            if pairs is not None:
                pairs_by_design["V5-PAIR"] = pairs
                f13_inputs += used
            conf = confirmation_inputs(out_root)
            note = ""
            if conf["pairs"] is not None and {"observed_logsf", "predicted_logsf"} <= set(conf["pairs"].columns):
                pairs_by_design["V6"] = conf["pairs"]
                f13_inputs += conf["inputs"]
            else:
                note = "V6 panel skipped: " + (conf["skip_reason"] or f"input missing: {CONFIRMATION_DIR}/v6_pairs.csv")
            if pairs_by_design:
                results.append(FG.fig13_direction_confusion(pairs_by_design, figures_dir, inputs=f13_inputs, arm="M2",
                                                            skipped_note=note))
            else:
                results.append(FG.skipped("F13", "no V5-PAIR pair-level predictions (verified M2 V5-PAIR record and "
                                                 f"{PAIRS_PARQUET}); " + note, f13_inputs))
    if want & {"F10", "F11"}:
        emb_arm = embedding_arms[0] if embedding_arms else EMBEDDING_ARMS[0]
        if refit_embeddings and store is not None:
            state = D.PlanState.read(rd.plan_state_path(out_root))
            side["embedding_refits"] = refit_embeddings_from_records(out_root, store, state, arms=embedding_arms)
            outs += [Path(p) for p in side["embedding_refits"]["outputs"]]
        emb = embedding_inputs(out_root, emb_arm)
        label = EMBEDDING_REPLICATE_LABEL if "reproduced" in emb["source"] else "bootstrap"
        if "F10" in want:
            r10 = FG.fig10_metal_embedding(emb["metal"], figures_dir, inputs=emb["inputs"], arm=emb_arm,
                                           replicates=emb["replicates"], stability=emb["stability"], replicate_label=label)
            r10.stats["source"] = emb["source"]
            if r10.status == "skipped":
                r10.reason += (f"; not computed (input missing: {EMBEDDINGS_DIR}/metal_embeddings.csv and "
                               f"{EMBEDDINGS_REFIT_DIR}/{emb_arm}/metal_embeddings.csv)")
            results.append(r10)
        if "F11" in want:
            r11 = FG.fig11_extractant_embedding(emb["system"], figures_dir, inputs=emb["inputs"], arm=emb_arm)
            r11.stats["source"] = emb["source"]
            if r11.status == "skipped":
                r11.reason += (f"; not computed (input missing: {EMBEDDINGS_DIR}/system_embeddings.csv and "
                               f"{EMBEDDINGS_REFIT_DIR}/{emb_arm}/system_embeddings.csv)")
            results.append(r11)
    if "F12" in want:
        conf = confirmation_inputs(out_root)
        if conf["rows"] is None and conf["pairs"] is None and conf.get("systems") is None:
            results.append(FG.skipped("F12", conf["skip_reason"] or f"input missing: {CONFIRMATION_DIR}/v6_rows.csv, "
                                                                     f"{CONFIRMATION_DIR}/v6_pairs.csv",
                                      [f"{CONFIRMATION_DIR}/v6_rows.csv", f"{CONFIRMATION_DIR}/v6_pairs.csv"]))
        else:
            results.append(FG.fig12_prnd_reconstruction(conf["rows"], conf["pairs"], figures_dir, inputs=conf["inputs"],
                                                        arm=arm or "deployed", systems=conf.get("systems")))
    for r in results:
        if r.status == "written":
            outs.append(resolve_figure_output(out_root, r.path))
            if r.data_path:
                outs.append(resolve_figure_output(out_root, r.data_path))
        log(f"{r.figure}: {r.status}" + (f" -> {r.path}" if r.status == "written" else "")
            + (f" ({r.reason})" if r.reason else ""))
    return results, outs, side


def resolve_figure_output(out_root: Path, p: str | Path) -> Path:
    """The absolute path of a figure output.

    A ``FigureResult.path`` / ``data_path`` is REPOSITORY-relative (``figures._rel`` -> ``paths.rel``), with an absolute
    path only where the file lies outside the repository.  Joining it to ``out_root`` gave
    ``<out_root>/generations/gen19_chem_transfer/figures/F07_*.png``, which exists nowhere, and ``Run.outputs``' own
    ``exists()`` filter then dropped it silently: the figures manifest listed 2 outputs (``figures_index.json`` and
    ``tables/s1e_error_vs_support.csv``, both joined correctly elsewhere) instead of every PNG and data CSV, so no figure
    carried a registered digest and ``verify_manifests`` could not check one.  The repository-relative path is joined to
    ``paths.REPO_ROOT``, which is the root ``paths.rel`` measured it against.
    """
    q = Path(p)
    if q.is_absolute():
        return q
    cand = paths.REPO_ROOT / q
    return cand if cand.exists() else Path(out_root) / q


def index_path(out_root: Path) -> Path:
    return Path(out_root) / "evaluation" / "figures" / "figures_index.json"


def carry_over_unselected(out_root: Path, index: Mapping[str, Any], *, only: Sequence[str] | None) -> dict[str, Any]:
    """With ``--only``, the entries of the figures this invocation did NOT make are carried over from the index on disk.

    ``figures_index`` is built from ``results``, which under ``--only`` holds just the selected figures, so writing it
    unchanged replaced a seven-figure index with a one-figure one: a ``--only F12`` run to add the confirmation figure
    silently deleted F07-F11 and F13 from the index the report reads.  A carried entry keeps its own record and is marked
    with the invocation that produced it (its ``git_head`` / ``code_sha256``), because this invocation's header describes
    only the figures it made; ``n_written`` / ``n_skipped`` are recomputed over the merged list.
    """
    if not only:
        return dict(index)
    made = {str(f.get("figure")) for f in index.get("figures", [])}
    prev = D.read_record(index_path(out_root)) or {}
    src_ = {k: prev.get(k) for k in ("git_head", "code_sha256") if prev.get(k)}
    kept = [{**f, "carried_over_from_earlier_invocation": src_ or True}
            for f in prev.get("figures", []) if str(f.get("figure")) not in made]
    figs = sorted([*index.get("figures", []), *kept], key=lambda f: str(f.get("figure")))
    return {**dict(index), "figures": figs,
            "n_written": sum(f.get("status") == "written" for f in figs),
            "n_skipped": sum(f.get("status") == "skipped" for f in figs),
            "only": list(only), "n_carried_over": len(kept),
            "carried_over": ("this invocation made only %s; every other entry is the one an earlier invocation wrote "
                             "(carried_over_from_earlier_invocation), and the header fields git_head / code_sha256 "
                             "describe THIS invocation" % ", ".join(sorted(only))) if kept else "none"}


def main(argv=None, *, check: Callable[[], int] | None = None, digests: Callable[[], Mapping[str, Any]] | None = None
         ) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    rd = runner_module()
    REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=ns.expect_addenda)
    H3.refuse_unless_cheap_complete(out_root)                   # before any heavy load (task X finding VL2-04)
    log("coextractant ids")
    coext = rd.coextractant_ids()
    gate = refuse_unless_ready(out_root, check=check, digests=digests, expect_addenda=ns.expect_addenda, excluded_ids=coext)
    if ns.check_only:
        print(json.dumps({"discovery_complete": gate["discovery_complete"]["complete"],
                          "scorer_decisions_ok": gate["scorer_decisions"]["ok"], "ladder_complete": gate["ladder_complete"],
                          "deployed": deployed_arm(out_root)}, indent=2, default=str))
        return 0
    code = code_digest()
    only = [s.strip() for s in ns.only.split(",") if s.strip()] if ns.only else None
    with (Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=D.PRIMARY_SEED,
              extra={"prereg_gate": gate["prereg_gate"], "gate": {k: v for k, v in gate.items() if k != "prereg_gate"},
                     "code_sha256": code["combined"], "code_parts": code["parts"], "git_head": git_head(),
                     "bootstrap_seed": ET.BOOTSTRAP_SEED, "n_resamples": ns.n_resamples,
                     "discovery_seeds": list(D.DISCOVERY_SEEDS), "seed": D.PRIMARY_SEED})
          if not ns.no_manifest else _Null()) as run:
        t0 = time.perf_counter()
        arms = [a.strip() for a in str(ns.embedding_arms).split(",") if a.strip()]
        results, outs, side = make_all(out_root, only=only, n_resamples=ns.n_resamples,
                                       refit_embeddings=not ns.no_embedding_refits, embedding_arms=arms)
        conf = confirmation_inputs(out_root)
        ran = bool(conf["run_decided"] or conf["run_started"])
        refits = (side.get("embedding_refits") or {}).get("arms", {})
        if not refits and ns.no_embedding_refits:
            # the tables on disk were written by an earlier invocation: its verification record is what the index cites
            for a in arms:
                v = D.read_record(Path(out_root) / EMBEDDINGS_REFIT_DIR / a / "verification.json")
                if v:
                    refits[a] = {**v, "source": f"{EMBEDDINGS_REFIT_DIR}/{a}/verification.json (written by an earlier "
                                                "invocation; --no-embedding-refits)"}
        index = FG.figures_index(results, {"git_head": git_head(), "code_sha256": code["combined"],
                                           "deployed": deployed_arm(out_root), "runtime_s": round(time.perf_counter() - t0, 1),
                                           "confirmation_half_read": False, "v6_read_outside_confirmation_files": False,
                                           "confirmation_run": {"decided": conf["run_decided"], "started": conf["run_started"],
                                                                "status": "run" if ran else CONFIRMATION_NOT_RUN},
                                           "embedding_refits": {a: {k: v for k, v in s_.items() if k != "folds"}
                                                                for a, s_ in refits.items()}})
        index = carry_over_unselected(out_root, index, only=only)
        outs.append(write_json(index_path(out_root), H3.json_safe(index)))
        if run is not None:
            run.outputs(*[p for p in outs if Path(p).exists()])
            run.extra.update({"figures": [r.record() for r in results], "confirmation_half_read": False,
                              "v6_target_rows_scored": 0, "embedding_refits": H3.json_safe(side.get("embedding_refits")),
                              "confirmation_run": index["confirmation_run"]})
    log(f"figures: {sum(r.status == 'written' for r in results)} written, {sum(r.status == 'skipped' for r in results)} skipped")
    return 0


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
