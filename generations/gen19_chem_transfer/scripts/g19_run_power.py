"""``scripts/g19_run_power.py`` -- the section 8 signal-injection power check and the section 8 "reliability before
correlation" report, after discovery.

Refuses to start unless, in this order:

1. ``scripts/g19_seal_prereg.py --check`` exits 0 AND the sealed text is the registered digest with POST-HOC addendum 1
   (``g19_run_discovery.refuse_unless_sealed``);
2. the discovery run is **COMPLETE** (``h3.discovery_complete``: the final plan stage ``10_not_implemented`` recorded in
   ``evaluation/discovery/decisions/wall_clock.json``, every stage of the current plan done, and every ``fit`` job's
   record set verified COMPLETE against the digests the current code, fold files and plan state produce);
3. the scorer's decision AND contrast files exist and ``decisions.json`` is newer than every fold record
   (``h3.scorer_decisions_present``), because the contrasts that need the check are read from them.

Power check (section 8 "Signal-injection power check"; POST-HOC addendum 1: X(?) rows dropped from the injected refits)

* every failed **H1 / H1b / S1(b) / H3** contrast of the scorer's tables (``power.contrasts_needing_power``) is re-run on
  ``y' = y + kappa s``, ``kappa in {0.1, 0.25, 0.5, 1.0}``;
* ``s_row = u_m v_s`` seeded and standardised (``transfer.injected_signal``), and for an H3 contrast every An(III) state
  shares the ``u`` of its nearest-CN8-radius Ln(III) (``power.h3_u_share``);
* the injection is applied to training and test rows alike, **X(?) rows are dropped from the injected refits of every arm
  in the contrast**, and the scored rows stay exactly the un-injected run's (``transfer.InjectedRun``, reused);
* every arm is refitted at the hyperparameters its discovery record SELECTED on the same fold
  (``h3.selected_hyperparameters`` / ``h3.frozen_runner``), never re-tuned; a closed-form comparator (B3i, B0, ...) is
  refitted on the exact leave-one-cell-out folds of section 3.1 as in discovery, never on the heavy arm's batched folds,
  and both arms must score exactly the un-injected run's rows (task X finding V-03);
* the report gives ``kappa_min``, the smallest kappa at which R19 passes; a null with ``kappa_min > 0.25`` is reported
  UNDECIDED (underpowered);
* **injected values are never persisted**: no prediction parquet and no target is written -- only metrics, R19 rows and
  kappa_min (``power.assert_no_injected_values`` checks every output frame; brief section 33).

Reliability report (section 8 "Reliability before correlation"): B7 slopes ``n`` / ``p_eff``, per-system logSF amplitudes
and support-score components by default (no learned fit); the factor loadings ``u_m`` / ``v_s`` and the learned embeddings
with ``--include-learned``.  PER UNIT (task X finding V-02): split-half by publication group (20 seeded halves of the
unit's own groups, Spearman-Brown) for every unit with >= 4 groups, the delete-one-publication jackknife for every unit
with 2-3 groups, single-group units counted; the quantity's reliability is the minimum over the computed branches and the
0.3 floor makes its correlations UNDECIDED (unreliable).  No ``V6_TARGET_ROWS`` row is read by any reliability estimate
and the observed-logSF amplitude is estimated on the selection half only (task X finding VL2-06).

Outputs (new directories only): ``evaluation/power/`` and ``tables/power_*.csv`` / ``tables/reliability_*.csv``.

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_power.py [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import h3 as H3  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.evaluation import power as PW  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.evaluation import transfer as ET  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json  # noqa: E402
from gen19ct.models import interface as I  # noqa: E402

NAME = "g19_run_power"
#: the registry stage of this runner (addendum 2 item 5: the seal gate prefers manifests/digest_registry.json when it
#: exists and falls back to the constants of evaluation.discovery otherwise)
STAGE = "power"
CODE_FILES: tuple[Path, ...] = (paths.G19_ROOT / "gen19ct" / "evaluation" / "power.py",
                                paths.G19_ROOT / "gen19ct" / "evaluation" / "h3.py", Path(__file__).resolve())
CODE_OBJECTS: tuple[Any, ...] = (PW.injected_run, PW.fold_inputs, PW.h3_u_share, PW.kappa_min,
                                 PW.split_half_reliability, PW.jackknife_reliability, PW.embedding_stability,
                                 H3.frozen_runner, H3.selected_hyperparameters)
#: the scorer tables the failed contrasts are read from
CONTRAST_FILES: tuple[str, ...] = ("contrasts_registered.csv",)
H3_CONTRAST_FILE = "h3_contrasts.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def runner_module():
    return _script("g19_run_discovery")


def h3_module():
    return _script("g19_run_h3")


def scorer_module():
    return _script("g19_score_discovery")


_CROSSINGS: pd.DataFrame | None = None


def crossings() -> pd.DataFrame:
    """``folds/wildcard_copy_crossings.csv`` as the scorer reads it (``g19_score_discovery.CROSSINGS_CSV``), cached.

    The wildcard-copy crossings depend on the FOLDS only -- which fold hid which row, and which rows are copies of one
    another -- never on an arm or on a target, so the injected run's frames are annotated from exactly the table the
    un-injected run used."""
    global _CROSSINGS
    if _CROSSINGS is None:
        sc = scorer_module()
        _CROSSINGS = pd.read_csv(sc.CROSSINGS_CSV, dtype={"row_id": str, "partner_id": str})
    return _CROSSINGS


def code_digest() -> dict[str, Any]:
    rd = runner_module()
    return D.code_digest(rd.CODE_FILES + (Path(rd.__file__).resolve(),) + CODE_FILES, rd.RUNNER_OBJECTS + CODE_OBJECTS)


# ============================================================================================= #
# the gate
# ============================================================================================= #

def refuse_unless_ready(out_root: Path, *, check: Callable[[], int] | None = None,
                        digests: Callable[[], Mapping[str, Any]] | None = None, expect_addenda: int | None = None,
                        excluded_ids: Sequence[str] = (), folds_dir: Path | None = None,
                        discovery_code: str | None = None, runners: Mapping[str, Any] | None = None,
                        jobs: Sequence[D.JobSpec] | None = None, state: D.PlanState | None = None,
                        prereg_gate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The H3 runner's three gates plus the contrast files the power check reads.

    The H3 gate's ``prereg_gate`` is the H3 stage's, and this run's records are the POWER stage's: addendum 2 item 5
    says a record is verified "against the registry entry of its stage", so what is RECORDED here is
    :data:`STAGE`'s own gate -- ``prereg_gate``, the dict ``main`` already took from
    ``registry.refuse_unless_sealed(STAGE, ...)`` and used to discard -- with H3's kept beside it under
    ``h3_prereg_gate`` (task X finding V-P05).  Passing no ``prereg_gate`` takes it here instead."""
    gate = h3_module().refuse_unless_ready(out_root, check=check, digests=digests, expect_addenda=expect_addenda,
                                          excluded_ids=excluded_ids, folds_dir=folds_dir,
                                          discovery_code=discovery_code, runners=runners, jobs=jobs, state=state)
    gate["h3_prereg_gate"] = gate.get("prereg_gate")
    gate["prereg_gate"] = dict(prereg_gate) if prereg_gate is not None else \
        REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=expect_addenda)
    root = D.discovery_root(out_root)
    present = {f: (root / f).exists() for f in CONTRAST_FILES}
    if not all(present.values()):
        raise SystemExit("refused: the scorer's contrast files are missing, and section 8 reads the failed H1 / H1b / H3 "
                         f"contrasts from them ({present}). Run scripts/g19_score_discovery.py, then rerun")
    gate["contrast_files"] = present
    return gate


# ============================================================================================= #
# arm specs and the injected fits
# ============================================================================================= #

def parse_arm(spec: str) -> tuple[str, str]:
    """``"M2"`` -> ``("M2", "WITH")``; ``"M2:WITHOUT"`` -> ``("M2", "WITHOUT")`` (an H3 contrast's endpoints)."""
    s = str(spec)
    arm, _, transform = s.partition(":")
    arm = D.ARM_ALIASES.get(arm, arm)
    t = transform or "WITH"
    if t not in H3.ALL_ARMS:
        raise ValueError(f"arm spec {spec!r}: {t!r} is not an H3 transform")
    return arm, t


def injected_corpus(corpus: Any, run: ET.InjectedRun) -> Any:
    """A :class:`g19_run_discovery.Corpus` over the injected rows: the X(?) rows the addendum drops are gone and the
    target is ``y + kappa s`` (training and test alike).  Everything keyed by row (V6 mask, co-extractant flags,
    registered halves, condition vectors) is reindexed; the descriptor tables and fold files are reused."""
    rd = runner_module()
    frame = run.frame
    kept = frame.index
    table = I.RowTable(frame, systems=corpus.systems, components=corpus.comps)
    return rd.Corpus(frame=frame, slim=FI.slim_frame(frame), table=table, v6=corpus.v6.reindex(kept),
                     coext=corpus.coext.reindex(kept), systems=corpus.systems, comps=corpus.comps,
                     cv=None if corpus.cv is None else corpus.cv.reindex(kept),
                     pmap=corpus.pmap, row_half={k: v.reindex(kept) for k, v in corpus.row_half.items()},
                     folds_dir=corpus.folds_dir, index_json=corpus.index_json)


def injected_fold_context(job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Any, icorpus: Any, run: ET.InjectedRun,
                          out_root: Path, state: D.PlanState, *, code: str, transform: str = "WITH") -> tuple[Any, dict]:
    """The fold context of one injected refit.

    Training is ``InjectedRun.fold_inputs``: the input rows minus the fold's hidden rows minus the X(?) rows -- so a
    hidden-but-unscored row can never be refitted -- and the scored rows must be hidden rows of the fold and scored rows
    of the run, identical to the un-injected run's.  The fold builders' outer guard is NOT re-run (its universe is the
    un-injected corpus, which still holds the dropped X(?) rows); instead every invariant it protects is asserted here:
    the scored rows are selection-half, known-state, not Sr(III), not ``V6_TARGET_ROWS``, hidden in the fold and absent
    from the training rows (:data:`gen19ct.evaluation.power.READINGS` ``scored_rows``)."""
    rd = runner_module()
    what = f"{job.key}/{fold.fold_id}"
    if fold.half == D.CONFIRMATION:
        raise AssertionError(f"{what}: a confirmation-half fold is never scored in discovery")
    sel = list(D.selection_scored_ids(fold))
    sc_ids = [r for r in sel if not bool(corpus.coext_by_id.get(r, False))]
    if not sc_ids:
        raise ValueError(f"{what}: no selection-half scored row")
    D.assert_selection_rows(sc_ids, corpus.half_by_id[job.design], what)
    hidden_full = corpus.labels_of(fold.hidden_row_ids)
    sc_labels = corpus.labels_of(sc_ids)
    train_frame, scored = run.fold_inputs(hidden_full, sc_labels)
    if transform != "WITH":
        train_frame = D.h3_training_rows(train_frame, transform, seed=int(job.seed))
    table = icorpus.table
    mask = np.zeros(table.n, dtype=bool)
    mask[table.positions(train_frame.index)] = True
    pos = table.positions(scored)
    if mask[pos].any():
        raise AssertionError(f"{what}: a scored row is an injected training row")
    st = icorpus.frame.loc[scored, SG.METAL_COL]
    if st.isna().any() or (st == I.SR_III).any():
        raise AssertionError(f"{what}: an X(?) or Sr(III) row would be scored")
    D.assert_v6_clean(scored, icorpus.v6, what)
    hidden_kept = hidden_full.intersection(icorpus.frame.index)
    ctx = I.FitContext(systems=icorpus.systems, components=icorpus.comps, table=table,
                       hidden_index=hidden_kept, v6_mask=icorpus.v6, exclude_from_scoring=icorpus.coext,
                       isolation_check=rd.inner_isolation_check(job, icorpus),
                       guard_cache=icorpus.guard_cache.setdefault((job.design, job.variant, job.drop_sr), {}),
                       seed=int(job.seed))
    fc = rd.FoldContext(job=job, fold=fold, ordinal=ordinal, corpus=icorpus, mask=mask, hidden=hidden_kept,
                        sc_ids=sc_ids, sc_labels=pd.Index(scored), positions=pos, ctx=ctx, out_root=out_root,
                        guard_mode=rd.guard_mode_for(job, state), batching_label=rd.batching_label(job, state),
                        code=code, state=state, design_hash=icorpus.design_hash(job.stem))
    info = {"n_train": int(mask.sum()), "n_scored_selection": len(sc_ids), "transform": transform,
            "n_hidden_in_injected_corpus": int(len(hidden_kept)),
            "n_rows_dropped_unknown_state": int(len(run.dropped_unknown_state))}
    return fc, info


def injected_predictions(arm_spec: str, entry: Mapping[str, Any], corpus: Any, icorpus: Any, run: ET.InjectedRun,
                         out_root: Path, state: D.PlanState, *, code: str) -> pd.DataFrame:
    """One arm's predictions of every fittable fold of the contrast's design under the injected targets, refitted at the
    hyperparameters its discovery record SELECTED (never re-tuned).  In memory only: nothing is written."""
    h3m = h3_module()
    arm, transform = parse_arm(arm_spec)
    base = H3.with_job(arm, entry["design"], state, seed=int(entry["seed"]))
    job = base if transform == "WITH" else H3.h3_job(base, arm, transform)
    runner = H3.frozen_runner(arm)
    frames = []
    for fold, ordinal in corpus.fittable(job):
        wrec = h3m.with_record(out_root, {"model_arm": arm, "with_job": base}, fold)
        fc, _ = injected_fold_context(job, fold, ordinal, corpus, icorpus, run, out_root, state, code=code,
                                      transform=transform)
        outputs = runner.point(fc, wrec)
        o = next(iter(outputs.values()))
        frames.append(D.prediction_frame(o.pred, job=job, arm=arm_spec, fold=fold, ordinal=ordinal, row_ids=fc.sc_ids,
                                         selected_config=o.selected_config, model_seed=o.model_seed,
                                         intervals_status="pending", batching_label=fc.batching_label,
                                         fit_seconds=o.seconds))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def injected_attrs(attrs: pd.DataFrame, run: ET.InjectedRun, corpus: Any) -> pd.DataFrame:
    """``attrs`` with the injected target on every row (the scored rows are scored against ``y' = y + kappa s``, section
    8: "the injection is applied to training and test rows alike").  In memory only."""
    ids = corpus.frame[FI.ROW_ID].astype(str)
    y = pd.Series(run.frame[I.TARGET_COL].to_numpy(dtype=float), index=ids.reindex(run.frame.index).to_numpy())
    out = attrs.copy()
    out[EM.Y_COL] = pd.to_numeric(y.reindex(out.index), errors="coerce").to_numpy(dtype=float)
    return out


# ============================================================================================= #
# one contrast
# ============================================================================================= #

def contrast_entry(row: Mapping[str, Any], *, seed: int = D.PRIMARY_SEED) -> dict[str, Any]:
    """The power-check entry of one failed contrast row: its family, design, arms and the injection's u-sharing."""
    design = str(row.get("design") or "V5")
    design_token = {"V5": "V5", "V5-P": "V5", "V1": "V1", "V2": "V2", "V5-PAIR": H3.PAIR_DESIGN}.get(design, design)
    return {"key": str(row.get("key") or row.get("contrast")), "contrast": str(row.get("contrast")),
            "family": str(row.get("family")), "design": design_token, "design_label": design,
            "candidate": str(row.get("candidate")), "comparator": str(row.get("comparator")),
            "margin": float(row["margin"]) if row.get("margin") is not None and np.isfinite(float(row["margin"]))
            else float(ET.MARGIN_FLOOR), "seed": int(seed),
            "uninjected": {k: row.get(k) for k in ("point", "r19_verdict_full", f"verdict_{H3.VERDICT_SCOPE}",
                                                   "p_two_sided", "percentile_low", "percentile_high") if k in row}}


def run_contrast(entry: Mapping[str, Any], corpus: Any, attrs: pd.DataFrame, out_root: Path, state: D.PlanState, *,
                 code: str, kappas: Sequence[float] = PW.KAPPAS, n_resamples: int = ET.N_RESAMPLES,
                 v1_scheme: str | None = None, comp_v1_scheme: str | None = None,
                 remainder_groups: Sequence[str] = ()) -> dict[str, Any]:
    """The power check of one contrast: one injected run per kappa, both arms refitted at their selected
    hyperparameters (a closed-form comparator on the exact folds; ``h3.with_design_dir``), R19 on the injected scored
    rows, then ``kappa_min``.  ``v1_scheme`` / ``comp_v1_scheme`` default to each arm's own V1 scoring scheme."""
    design, label = entry["design"], entry["design_label"]
    seed = int(entry["seed"])
    v6 = attrs["v6_target_row"].astype(bool)
    states = corpus.frame[SG.METAL_COL].dropna().astype(str).unique()
    u_share = PW.u_share_for(entry["family"], states)
    scored_index = _scored_index(entry, corpus, state)
    scored_row_ids = _scored_row_ids(corpus, scored_index)
    results: list[PW.KappaResult] = []
    unit_rows: list[pd.DataFrame] = []
    n_dropped = 0
    for kappa in kappas:
        run = PW.injected_run(corpus.frame, kappa=float(kappa), seed=seed, scored_index=scored_index, u_share=u_share)
        n_dropped = int(len(run.dropped_unknown_state))
        icorpus = injected_corpus(corpus, run)
        iattrs = injected_attrs(attrs, run, corpus)
        preds = {}
        for spec in (entry["candidate"], entry["comparator"]):
            preds[spec] = injected_predictions(spec, entry, corpus, icorpus, run, out_root, state, code=code)
        frames = {}
        sc = scorer_module()
        for spec, pred in preds.items():
            # the scoring-filter sensitivities of R19 item 6 read wildcard_copy_partner_in_training, which
            # discovery.prediction_frame does not carry: the scorer adds it from the crossings table per design stem
            # (g19_score_discovery.Store.frame), and the injected run is annotated the same way on the same folds.
            stem = FI.design_stem(str(pred["design"].iloc[0]), str(pred["variant"].iloc[0]), str(pred["scheme"].iloc[0]))
            pred = sc.wildcard_flags(pred, stem, crossings())
            hc = f"registered_half_{'V5' if label in ('V5', 'V5-P', 'V5-PAIR') else design}"
            frames[spec] = D.scoring_frame(pred, iattrs, design=label, v6_mask=v6,
                                          what=f"injected {spec}@{label} kappa={kappa:g}", half_col=hc)
        cand, comp = frames[entry["candidate"]], frames[entry["comparator"]]
        ET.assert_same_scored_rows(scored_row_ids, cand.index)
        ET.assert_same_scored_rows(scored_row_ids, comp.index)              # the exact-fold comparator too (V-03)
        cs = v1_scheme or H3.v1_scheme_of(parse_arm(entry["candidate"])[0], design, state)
        ks = comp_v1_scheme or H3.v1_scheme_of(parse_arm(entry["comparator"])[0], design, state)
        pu = D.paired_units(cand, comp, label, candidate=entry["candidate"], comparator=entry["comparator"], v6_mask=v6,
                            cand_v1_scheme=cs, comp_v1_scheme=ks, remainder_groups=list(remainder_groups) or None)
        learned = not (H3.is_deterministic(parse_arm(entry["candidate"])[0])
                       and H3.is_deterministic(parse_arm(entry["comparator"])[0]))
        reg = ET.REGISTERED_SENSITIVITIES[label]
        sens: dict[str, Any] = {}
        reduced = []
        for name in D.SCORING_FILTER_SENSITIVITIES:
            if name not in reg:
                continue
            c, k = D.filtered_pair(cand, comp, name)
            sens[name] = ET.UNTESTABLE if c.empty else D.paired_units(
                c, k, label, candidate=entry["candidate"], comparator=entry["comparator"], v6_mask=v6,
                cand_v1_scheme=cs, comp_v1_scheme=ks, remainder_groups=list(remainder_groups) or None).delta
            reduced.append(name)
        for name in reg:
            sens.setdefault(name, ET.UNTESTABLE)
        res = D.evaluate_contrast(name=f"{entry['contrast']} (injected kappa={kappa:g})", family=entry["family"],
                                  design=label, primary=pu, margin=float(entry["margin"]),
                                  seed_deltas={seed: pu.delta}, sensitivities=sens, deterministic=not learned,
                                  learned=learned, reduced_sensitivities=reduced if learned else None,
                                  n_resamples=n_resamples)
        results.append(PW.KappaResult(kappa=float(kappa), passed=res["scopes"][H3.VERDICT_SCOPE]["verdict"] == "PASS",
                                      scope_verdict=res["scopes"][H3.VERDICT_SCOPE]["verdict"],
                                      full_verdict=res["r19"].verdict, point=res["point"], margin=res["margin"],
                                      n_units=res["n_units"], n_rows=res["n_rows"], contrast=res))
        # the bootstrap input of this kappa, as metrics, so kappa_min stays auditable without a refit (finding V-L2)
        unit_rows.append(PW.per_unit_rows(pu, contrast=entry["key"], kappa=float(kappa), design=label,
                                          candidate=entry["candidate"], comparator=entry["comparator"]))
        log(f"  kappa={kappa:g}: Delta={res['point']:.4f}, {H3.VERDICT_SCOPE} {results[-1].scope_verdict}")
    rec = PW.power_record(entry["key"], family=entry["family"], design=label,
                          arms=[entry["candidate"], entry["comparator"]], results=results, seed=seed,
                          u_share=u_share, n_dropped_unknown_state=n_dropped, uninjected=entry.get("uninjected"))
    return {**rec, "per_unit_frame": pd.concat(unit_rows, ignore_index=True) if unit_rows else pd.DataFrame()}


def _scored_index(entry: Mapping[str, Any], corpus: Any, state: D.PlanState) -> pd.Index:
    """The un-injected run's scored rows of the contrast's design: the selection-half scored rows of every fittable fold
    (minus the acidic co-extractant rows), as labels of the full corpus."""
    arm, _ = parse_arm(entry["candidate"])
    job = H3.with_job(arm, entry["design"], state, seed=int(entry["seed"]))
    ids: list[str] = []
    for fold, _ordinal in corpus.fittable(job):
        ids += [r for r in D.selection_scored_ids(fold) if not bool(corpus.coext_by_id.get(r, False))]
    return corpus.labels_of(sorted(set(ids)))


def _scored_row_ids(corpus: Any, scored_index: Iterable[Any]) -> pd.Index:
    """The ``row_id`` labels of the un-injected run's scored rows.

    :func:`_scored_index` returns the CORPUS FRAME's own labels (``g19_run_discovery.Corpus.labels_of`` maps a row id
    through ``idmap`` to ``frame.index``), which is the space the injection machinery works in
    (``transfer.prepare_injected_run``, ``InjectedRun.fold_inputs``).  ``discovery.scoring_frame`` instead indexes its
    frame by ``row_id``.  Finding V-03 ("both arms score exactly the un-injected run's rows") is therefore checked in
    row_id space: comparing the two label spaces directly can never pass, because the corpus index is not the row id.
    """
    return pd.Index(corpus.frame.loc[pd.Index(list(scored_index)), FI.ROW_ID].astype(str).to_numpy())


def kappa_rows(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per (contrast, kappa) -- metrics only, no injected value."""
    rows = []
    for rec in records:
        for per in rec["per_kappa"]:
            rows.append({"contrast": rec["contrast"], "family": rec["family"], "design": rec["design"],
                         "arms": " vs ".join(rec["arms"]), **per, "kappa_min": rec.get("kappa_min"),
                         "power_verdict": rec.get("verdict"), "injection_seed": rec["injection_seed"],
                         "n_u_shared_states": rec["n_u_shared_states"],
                         "n_rows_dropped_unknown_state": rec["n_rows_dropped_unknown_state"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        PW.assert_no_injected_values(out, "power kappa table")
    return out


# ============================================================================================= #
# the reliability report
# ============================================================================================= #

def reliability_report(corpus: Any, attrs: pd.DataFrame, out_root: Path, *, include_learned: bool = False,
                       n_halves: int = PW.N_SPLIT_HALVES, seed: int = PW.SPLIT_HALF_SEED) -> dict[str, Any]:
    """Section 8 reliability of the derived quantities that need no learned fit (B7 slopes, per-system logSF amplitudes,
    support-score components), PER UNIT (``power.reliability_of``; task X finding V-02), and with ``include_learned`` the
    factor loadings and embeddings as well (:data:`gen19ct.evaluation.power.READINGS` ``reliability_not_fitted``).

    Rows read (task X findings VL2-06 and V-P03): no ``V6_TARGET_ROWS`` row enters any estimate, and every REPORTED
    quantity reads the selection half only -- the per-system |logSF| amplitude (an observed-target statistic) as
    before, and now the B7 slopes too.  Section 15 says the confirmation half "contributed to no ladder decision,
    claim or preferred-model choice", and a section 8 reliability gate IS claim-relevant (below the 0.3 floor a
    quantity "may neither support nor close a correlation-based claim"), so the both-halves B7 fit is kept only as a
    labelled diagnostic beside it.  ``confirmation_half_read`` is COMPUTED from the halves each estimator read
    (``halves_read`` per record), never asserted."""
    from gen19ct.evaluation import pairs as EP
    from gen19ct.models import mass_action as MA

    frame = corpus.frame
    v6 = corpus.v6.reindex(frame.index).fillna(False).astype(bool).to_numpy()
    half = corpus.row_half["V5"].reindex(frame.index).astype(str).to_numpy()
    base_idx = frame.index[~v6]
    sel_idx = frame.index[~v6 & (half == D.SELECTION)]
    groups = frame[I.PUB_GROUP_COL].astype(str)
    half_of = pd.Series(half, index=frame.index)

    def halves_of(idx: pd.Index) -> list[str]:
        """The registered halves the rows of ``idx`` come from -- what a quantity ACTUALLY read (task X finding V-P03:
        the flag is computed from the rows each estimator read, never asserted)."""
        return sorted({str(x) for x in half_of.reindex(pd.Index(idx)).dropna().unique()})

    reads = {"n_rows": int(len(frame)), "n_v6_target_rows_excluded": int(v6.sum()), "n_rows_after_v6": int(len(base_idx)),
             "n_rows_selection_half_after_v6": int(len(sel_idx)),
             "b7_slopes": "selection half (registered_half_V5 == S) minus V6_TARGET_ROWS: section 15 -- the "
                          "confirmation half contributes to no claim, and a section 8 reliability gate is "
                          "claim-relevant (a quantity below the 0.3 floor may neither support nor close a "
                          "correlation-based claim). The both-halves fit is reported beside as a diagnostic only",
             "logsf_amplitudes": "selection half (registered_half_V5 == S) minus V6_TARGET_ROWS: observed logSF of the "
                                 "confirmation half and of the V6 systems is never read",
             "support_score_components": "the discovery _support files (selection-half scored rows) minus V6_TARGET_ROWS"}
    records: list[dict[str, Any]] = []

    def fit_b7(idx: pd.Index) -> pd.DataFrame:
        rows = frame.loc[idx]
        if rows.empty:
            return pd.DataFrame()
        table = I.RowTable(rows, systems=corpus.systems, components=corpus.comps)
        ctx = I.FitContext(systems=corpus.systems, components=corpus.comps, table=table,
                           v6_mask=corpus.v6.reindex(rows.index), seed=D.PRIMARY_SEED)
        return MA.B7MassAction().fit_table(table, np.ones(table.n, dtype=bool), ctx).slopes_table()

    anion = pd.Series([I.NA_ANION if pd.isna(v) else str(v) for v in frame[SG.ACID_ANION_COL].to_numpy(dtype=object)],
                      index=frame.index, dtype=object)
    unit_b7 = (frame[SG.SYSTEM_COL].astype(object).astype(str) + " | " + anion).where(frame[SG.SYSTEM_COL].notna())
    sel_b7, both_b7 = fit_b7(sel_idx), fit_b7(base_idx)
    for col in ("n", "p_eff"):
        def est(idx: pd.Index, c=col) -> pd.Series:
            return PW.b7_slope_series(fit_b7(idx), c)
        rec = PW.reliability_of("b7_slopes", estimate_rows=est, full=PW.b7_slope_series(sel_b7, col),
                                unit_of_row=unit_b7.loc[sel_idx], group_of_row=groups.loc[sel_idx],
                                n_halves=n_halves, seed=seed)
        diag = PW.reliability_of("b7_slopes", estimate_rows=est, full=PW.b7_slope_series(both_b7, col),
                                 unit_of_row=unit_b7.loc[base_idx], group_of_row=groups.loc[base_idx],
                                 n_halves=n_halves, seed=seed)
        records.append({**rec, "unit": f"B7 {col} per (system, anion)", "rows_read": reads["b7_slopes"],
                        "halves_read": halves_of(sel_idx), "n_rows_read": int(len(sel_idx)),
                        "both_halves_diagnostic": {"reliability": diag.get("reliability"), "status": diag.get("status"),
                                                   "halves_read": halves_of(base_idx), "n_rows_read": int(len(base_idx)),
                                                   "note": "not reported: it reads confirmation-half observed log D "
                                                           "(section 15), and the gate it feeds is claim-relevant"}})

    pair_rows = pd.DataFrame({EM.PUB_GROUP_COL: groups.to_numpy(), EM.SYSTEM_COL: frame[SG.SYSTEM_COL].to_numpy(),
                              EM.CONDITION_KEY_COL: attrs[EM.CONDITION_KEY_COL].reindex(
                                  frame[FI.ROW_ID].astype(str)).to_numpy(),
                              EM.METAL_STATE_COL: frame[SG.METAL_COL].to_numpy(),
                              EM.Y_COL: pd.to_numeric(frame[I.TARGET_COL], errors="coerce").to_numpy(dtype=float)},
                             index=frame.index).loc[sel_idx]
    pair_rows["fold"] = "all"

    def amplitudes(idx: pd.Index) -> pd.Series:
        sub = pair_rows.loc[pair_rows.index.intersection(idx)]
        sub = sub[sub[EM.METAL_STATE_COL].notna() & np.isfinite(sub[EM.Y_COL].to_numpy(dtype=float))]
        if sub.empty:
            return pd.Series(dtype=float)
        pr = EP.comparable_pairs(sub, key_cols=(EM.PUB_GROUP_COL, EM.SYSTEM_COL, EM.CONDITION_KEY_COL), fold_col="fold")
        return PW.logsf_amplitudes(pr)
    full_amp = amplitudes(sel_idx)
    unit_sys = frame[SG.SYSTEM_COL].astype(object).astype(str).where(frame[SG.SYSTEM_COL].notna())
    records.append({**PW.reliability_of("logsf_amplitudes", estimate_rows=amplitudes, full=full_amp,
                                        unit_of_row=unit_sys.loc[sel_idx], group_of_row=groups.loc[sel_idx],
                                        n_halves=n_halves, seed=seed),
                    "unit": "per-system |logSF| amplitude", "rows_read": reads["logsf_amplitudes"],
                    "halves_read": halves_of(sel_idx), "n_rows_read": int(len(sel_idx))})

    sup_dir = D.discovery_root(out_root) / "_support"
    sup_files = sorted(sup_dir.rglob("*.parquet")) if sup_dir.exists() else []
    if sup_files:
        sup = pd.concat([pd.read_parquet(p) for p in sup_files], ignore_index=True)
        ids = frame[FI.ROW_ID].astype(str)
        state_of = pd.Series(frame[SG.METAL_COL].astype(object).to_numpy(), index=ids.to_numpy())
        sys_of = pd.Series(frame[SG.SYSTEM_COL].astype(object).to_numpy(), index=ids.to_numpy())
        v6_of = pd.Series(v6, index=ids.to_numpy())
        sup[EM.METAL_STATE_COL] = sup["row_id"].astype(str).map(state_of)
        sup[EM.SYSTEM_COL] = sup["row_id"].astype(str).map(sys_of)
        sup[EM.PUB_GROUP_COL] = sup["row_id"].astype(str).map(pd.Series(groups.to_numpy(), index=ids.to_numpy()))
        n_sup_v6 = int(sup["row_id"].astype(str).map(v6_of).fillna(False).astype(bool).sum())
        sup = sup[~sup["row_id"].astype(str).map(v6_of).fillna(False).astype(bool).to_numpy()].reset_index(drop=True)
        reads["n_support_rows_v6_excluded"] = n_sup_v6
        unit_cell = sup[list(EM.CELL_COLS)].astype(str).agg(EM.UNIT_KEY_SEP.join, axis=1)
        half_by_id = pd.Series(half, index=ids.to_numpy())
        sup_halves = sorted({str(x) for x in half_by_id.reindex(sup["row_id"].astype(str)).dropna().unique()})
        reads["n_support_rows"] = int(len(sup))
        comps = [c for c in sup.columns if str(c).startswith("support_s")]
        for comp in comps:
            def est(idx: pd.Index, c=comp) -> pd.Series:
                return PW.support_component_series(sup.loc[idx], c)
            full = PW.support_component_series(sup, comp)
            records.append({**PW.reliability_of("support_score_components", estimate_rows=est, full=full,
                                                unit_of_row=unit_cell, group_of_row=sup[EM.PUB_GROUP_COL],
                                                n_halves=n_halves, seed=seed),
                            "unit": f"{comp} per cell", "rows_read": reads["support_score_components"],
                            "halves_read": sup_halves, "n_rows_read": int(len(sup))})
    else:
        records.append({"quantity": "support_score_components", "unit": "per cell", "method": "none",
                        "status": "NOT_RUN", "reliability": float("nan"), "gate": ET.reliability_gate(float("nan")),
                        "detail": "no per-fold support file under evaluation/discovery/_support"})
    for q in ("factor_loadings", "embeddings"):
        records.append({"quantity": q, "unit": "metal state / system", "method": "none",
                        "status": "NOT_RUN" if not include_learned else "NOT_IMPLEMENTED",
                        "reliability": float("nan"), "gate": ET.reliability_gate(float("nan")),
                        "detail": PW.READINGS["reliability_not_fitted"]})
    table = PW.reliability_table(records)
    # computed from the halves each estimator actually read, never asserted (task X finding V-P03)
    read_halves = sorted({h for r in records for h in (r.get("halves_read") or ())})
    conf_read = [f"{r.get('quantity')} / {r.get('unit')}" for r in records
                 if D.CONFIRMATION in (r.get("halves_read") or ())]
    return {"schema": PW.SCHEMA, "records": records, "table": table,
            "floor": PW.RELIABILITY_FLOOR, "n_publication_groups": int(groups.loc[base_idx].nunique()),
            "rows_read": reads, "confirmation_half_read": bool(conf_read),
            "confirmation_half_read_by": conf_read, "halves_read": read_halves,
            "confirmation_half_rule": "section 15: the confirmation half contributes to no ladder decision, claim or "
                                      "preferred-model choice, and a section 8 reliability gate IS claim-relevant (a "
                                      "quantity below the 0.3 floor may neither support nor close a correlation-based "
                                      "claim), so every reported quantity here reads the selection half only; the "
                                      "flag is computed from halves_read of each record",
            "v6_target_rows_read": 0,
            "quantities": list(PW.RELIABILITY_QUANTITIES), "readings": PW.READINGS,
            "rule": "section 8: every derived per-unit quantity reports its reliability BEFORE any correlation of it is "
                    "read, per unit (split-half >= 4 groups, jackknife 2-3 groups); below the 0.3 floor its correlations "
                    "are UNDECIDED (unreliable)"}


# ============================================================================================= #
# main
# ============================================================================================= #

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--only", default=None, help="comma list of contrast keys to check")
    ap.add_argument("--dry-run", action="store_true", help="list the contrasts that need the check; fit nothing")
    ap.add_argument("--reliability-only", action="store_true")
    ap.add_argument("--no-reliability", action="store_true")
    ap.add_argument("--include-learned", action="store_true",
                    help="also report the reliability of the factor loadings and learned embeddings")
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def main(argv=None, *, check: Callable[[], int] | None = None, digests: Callable[[], Mapping[str, Any]] | None = None
         ) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    rd = runner_module()
    # the POWER stage's own seal gate: recorded, not discarded (task X finding V-P05)
    prereg_gate = REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=ns.expect_addenda)
    H3.refuse_unless_cheap_complete(out_root)                   # before any heavy load (task X finding VL2-04)
    log("coextractant ids")
    coext = rd.coextractant_ids()
    gate = refuse_unless_ready(out_root, check=check, digests=digests, expect_addenda=ns.expect_addenda,
                               excluded_ids=coext, prereg_gate=prereg_gate)
    state = D.PlanState.read(rd.plan_state_path(out_root))
    root = D.discovery_root(out_root)
    contrasts = PW.read_contrast_files([root / f for f in CONTRAST_FILES]
                                       + [H3.h3_root(out_root) / H3_CONTRAST_FILE])
    need = PW.contrasts_needing_power(contrasts)
    need = need[need["needs_power_check"].astype(bool)]
    if ns.only:
        keys = {k.strip() for k in ns.only.split(",") if k.strip()}
        need = need[need["contrast"].astype(str).isin(keys) | need.get("key", pd.Series("", index=need.index)).astype(str).isin(keys)]
    if ns.dry_run:
        print(need.to_string(index=False) if not need.empty else "no failed H1 / H1b / H3 contrast needs the check")
        print(json.dumps({"n_contrasts": int(len(need)), "kappas": list(PW.KAPPAS),
                          "kappa_min_informative": ET.KAPPA_MIN_INFORMATIVE,
                          "reliability_quantities": list(PW.RELIABILITY_QUANTITIES)}, indent=2))
        return 0
    code = code_digest()
    log("corpus")
    corpus = rd.load_corpus(coext)
    attrs = scorer_module().build_attrs()
    rem = sorted(g for g, n in attrs[EM.PUB_GROUP_COL].astype(str).value_counts().items() if n < 20)
    with (Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=D.PRIMARY_SEED,
              extra={"prereg_gate": gate["prereg_gate"], "gate": {k: v for k, v in gate.items() if k != "prereg_gate"},
                     "code_sha256": code["combined"], "code_parts": code["parts"], "kappas": list(PW.KAPPAS),
                     "readings": PW.READINGS, "seed": D.PRIMARY_SEED,
                     "injected_values_persisted": False})
          if not ns.no_manifest else _Null()) as run:
        records = []
        if not ns.reliability_only:
            for _, row in need.iterrows():
                entry = contrast_entry(row)
                log(f"power check: {entry['key']} ({entry['family']}, {entry['design_label']})")
                v1s = H3.v1_scheme_of(parse_arm(entry["candidate"])[0], entry["design"], state)
                records.append(run_contrast(entry, corpus, attrs, out_root, state, code=code["combined"],
                                            v1_scheme=v1s, remainder_groups=rem))
        rel = None if ns.no_reliability else reliability_report(corpus, attrs, out_root,
                                                               include_learned=ns.include_learned)
        outs = write_outputs(out_root, records, rel, gate=gate, contrasts=contrasts)
        if run is not None:
            run.outputs(*outs)
            run.extra.update({"n_contrasts_checked": len(records),
                              "kappa_min": {r["contrast"]: r.get("kappa_min") for r in records},
                              "power_verdicts": {r["contrast"]: r.get("verdict") for r in records},
                              # computed, never asserted (task X finding V-P03): the reliability pass reports which
                              # halves each estimator read; the contrasts themselves score selection-half rows only
                              "confirmation_half_read": bool((rel or {}).get("confirmation_half_read")),
                              "confirmation_half_read_by": list((rel or {}).get("confirmation_half_read_by") or ()),
                              "halves_read": list((rel or {}).get("halves_read") or ()),
                              "v6_target_rows_scored": 0})
    log(f"power check: {len(records)} contrast(s); " + ", ".join(f"{r['contrast']}: kappa_min={r.get('kappa_min')}"
                                                                for r in records))
    return 0


def write_outputs(out_root: Path, records: Sequence[Mapping[str, Any]], reliability: Mapping[str, Any] | None, *,
                  gate: Mapping[str, Any] | None = None, contrasts: pd.DataFrame | None = None) -> list[Path]:
    root = PW.power_root(out_root)
    tables = Path(out_root) / "tables"
    outs: list[Path] = []
    if records:
        kr = kappa_rows(records)
        outs.append(write_csv(kr, root / "power_kappa.csv"))
        outs.append(write_csv(kr, tables / "power_kappa.csv"))
        units = [r["per_unit_frame"] for r in records
                 if isinstance(r.get("per_unit_frame"), pd.DataFrame) and not r["per_unit_frame"].empty]
        if units:
            pu = pd.concat(units, ignore_index=True)
            PW.assert_no_injected_values(pu, "power per-unit metrics table")
            outs.append(write_csv(pu, root / "per_unit_mae.csv"))
            outs.append(write_csv(pu, tables / "power_per_unit_mae.csv"))
        drop = ("per_kappa_contrasts", "per_unit_frame")
        outs.append(write_json(root / "power_checks.json", PW.json_safe(
            {"schema": PW.SCHEMA, "checks": [{k: v for k, v in r.items() if k not in drop} for r in records],
             # which H3 contrasts OWE the check and which have one: the debt is inventoried in the file, so an unrun
             # check is a recorded debt rather than an absence (task X finding protocol VH-09)
             "needs_power": H3.needs_power_inventory(contrasts, checked=[str(r.get("contrast")) for r in records]),
             "gate": dict(gate or {}), "injected_values_persisted": False,
             # the bootstrap INPUT is persisted as metrics (per-unit MAE + cluster labels), never an injected value
             "per_unit_metrics_persisted": bool(units), "per_unit_metrics_path": PW.PER_UNIT_REL,
             "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})))
    if reliability is not None:
        tab = reliability["table"]
        if not tab.empty:
            outs.append(write_csv(tab, root / "reliability.csv"))
            outs.append(write_csv(tab, tables / "reliability_before_correlation.csv"))
        outs.append(write_json(root / "reliability.json", PW.json_safe(
            {k: v for k, v in reliability.items() if k != "table"})))
    return outs


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
