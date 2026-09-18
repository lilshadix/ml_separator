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
* F10 / F11: ``evaluation/power/embeddings/metal_embeddings.csv`` / ``system_embeddings.csv`` (+ ``metal_embeddings_
  replicate_*.csv``; written by ``g19_run_power.py --include-learned`` once implemented) -- skipped with the reason until
  they exist;
* F12: the confirmation run's ``evaluation/confirmation/v6_rows.csv`` and ``v6_pairs.csv`` only (V6 is touched once);
* F13: V5-PAIR pair-level direction from the verified M2 V5-PAIR record and ``folds/V5PAIR__primary__batched__pairs.parquet``;
  V6 pairs from the confirmation files.

Outputs (new directories only): ``figures/F07..F13_*.png``, ``figures/data/F*.csv``, ``tables/s1e_error_vs_support.csv``,
``evaluation/figures/figures_index.json``; manifest ``manifests/g19_make_figures.json`` (git HEAD, prereg + addendum digests,
the code digest of this script and ``figures.py``, seeds, runtime).  Nothing reads the confirmation half of any design
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
from gen19ct.evaluation import transfer as ET  # noqa: E402
from gen19ct.manifest import Run, git_head, write_csv, write_json  # noqa: E402

NAME = "g19_make_figures"
CODE_FILES: tuple[Path, ...] = (paths.G19_ROOT / "gen19ct" / "evaluation" / "figures.py", Path(__file__).resolve())
EMBEDDINGS_DIR = "evaluation/power/embeddings"
CONFIRMATION_DIR = "evaluation/confirmation"
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
    prereg = rd.refuse_unless_sealed(check, digests, expect_addenda=expect_addenda)
    st = state if state is not None else D.PlanState.read(rd.plan_state_path(out_root))
    code = discovery_code if discovery_code is not None else rd.current_code_digest()
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
    """Per hidden cell: MAE, the cell's support_score (mean over its rows), its domain status (majority), metal class."""
    key = sup.set_index(["fold_id", "row_id"])
    idx = pd.MultiIndex.from_arrays([fr["fold_id"].astype(str), fr["row_id"].astype(str)])
    lab = key.reindex(idx)
    f = fr.assign(domain_status=lab["domain_status"].to_numpy(), support_score=pd.to_numeric(lab["support_score"], errors="coerce").to_numpy(),
                  ambiguous=lab["domain_status_ambiguous"].astype("boolean").fillna(False).to_numpy())
    f = f[~f["ambiguous"].astype(bool)]
    f["err"] = (f["mean_logD"].astype(float) - f["log_D"].astype(float)).abs()
    sysk = attrs[EM.SYSTEM_COL].reindex(f["row_id"].to_numpy()).astype(str).to_numpy()
    f["system"] = sysk
    g = f.groupby(f["unit"].astype(str))
    out = g.agg(mae=("err", "mean"), support_score=("support_score", "mean"), n_rows=("err", "size"),
                system=("system", "first"), metal_class=("metal_class", "first"))
    out["domain_status"] = g["domain_status"].agg(lambda s: s.astype(str).value_counts().idxmax())
    out.index.name = "unit"
    return out.reset_index()


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


def embedding_inputs(out_root: Path) -> dict[str, Any]:
    d = Path(out_root) / EMBEDDINGS_DIR
    metal = _read_csv(d / "metal_embeddings.csv", index_col=0)
    system = _read_csv(d / "system_embeddings.csv", index_col=0)
    reps = [pd.read_csv(p, index_col=0) for p in sorted(d.glob("metal_embeddings_replicate_*.csv"))]
    stability = None
    rel = D.read_record(Path(out_root) / "evaluation" / "power" / "reliability.json")
    if rel:
        for r in rel.get("records") or []:
            if r.get("quantity") == "embeddings" and r.get("status") == "computed":
                stability = r
    used = [f"{EMBEDDINGS_DIR}/{p.name}" for p in sorted(d.glob("*.csv"))] if d.exists() else []
    if rel:
        used.append("evaluation/power/reliability.json")
    return {"metal": metal, "system": system, "replicates": reps, "stability": stability, "inputs": used}


def confirmation_inputs(out_root: Path) -> dict[str, Any]:
    d = Path(out_root) / CONFIRMATION_DIR
    rows, pairs = _read_csv(d / "v6_rows.csv"), _read_csv(d / "v6_pairs.csv")
    used = [f"{CONFIRMATION_DIR}/{n}" for n, x in (("v6_rows.csv", rows), ("v6_pairs.csv", pairs)) if x is not None]
    return {"rows": rows, "pairs": pairs, "inputs": used}


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
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def make_all(out_root: Path, *, only: Sequence[str] | None = None, n_resamples: int = ET.N_RESAMPLES
             ) -> tuple[list[FG.FigureResult], list[Path]]:
    rd, SD = runner_module(), scorer_module()
    want = set(only or ALL_FIGURES)
    figures_dir = Path(out_root) / "figures"
    tables = Path(out_root) / "tables"
    dep = deployed_arm(out_root)
    arm = dep.get("arm")
    results: list[FG.FigureResult] = []
    outs: list[Path] = []
    need_store = bool(want & {"F07", "F08", "F09", "F13"}) and arm is not None
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
                results.append(FG.skipped("F08", f"no current _support files for {arm} on V5-primary", f08_inputs))
        if "F13" in want:
            pairs_by_design: dict[str, pd.DataFrame] = {}
            f13_inputs = ["evaluation/discovery/decisions/decisions.json"]
            pairs, used = v5pair_pairs(store, attrs, out_root)
            if pairs is not None:
                pairs_by_design["V5-PAIR"] = pairs
                f13_inputs += used
            conf = confirmation_inputs(out_root)
            if conf["pairs"] is not None and {"observed_logsf", "predicted_logsf"} <= set(conf["pairs"].columns):
                pairs_by_design["V6"] = conf["pairs"]
                f13_inputs += conf["inputs"]
            results.append(FG.fig13_direction_confusion(pairs_by_design, figures_dir, inputs=f13_inputs, arm="M2"))
    if want & {"F10", "F11"}:
        emb = embedding_inputs(out_root)
        if "F10" in want:
            results.append(FG.fig10_metal_embedding(emb["metal"], figures_dir, inputs=emb["inputs"], arm=arm or "M2",
                                                    replicates=emb["replicates"], stability=emb["stability"]))
        if "F11" in want:
            results.append(FG.fig11_extractant_embedding(emb["system"], figures_dir, inputs=emb["inputs"], arm=arm or "M2"))
    if "F12" in want:
        conf = confirmation_inputs(out_root)
        results.append(FG.fig12_prnd_reconstruction(conf["rows"], conf["pairs"], figures_dir, inputs=conf["inputs"],
                                                    arm=arm or "deployed"))
    for r in results:
        if r.status == "written":
            outs.append(Path(out_root) / r.path if not Path(r.path).is_absolute() else Path(r.path))
            if r.data_path:
                outs.append(Path(out_root) / r.data_path if not Path(r.data_path).is_absolute() else Path(r.data_path))
        log(f"{r.figure}: {r.status}" + (f" ({r.reason})" if r.reason else f" -> {r.path}"))
    return results, outs


def main(argv=None, *, check: Callable[[], int] | None = None, digests: Callable[[], Mapping[str, Any]] | None = None
         ) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    rd = runner_module()
    rd.refuse_unless_sealed(check, digests, expect_addenda=ns.expect_addenda)
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
        results, outs = make_all(out_root, only=only, n_resamples=ns.n_resamples)
        index = FG.figures_index(results, {"git_head": git_head(), "code_sha256": code["combined"],
                                           "deployed": deployed_arm(out_root), "runtime_s": round(time.perf_counter() - t0, 1),
                                           "confirmation_half_read": False, "v6_read_outside_confirmation_files": False})
        outs.append(write_json(Path(out_root) / "evaluation" / "figures" / "figures_index.json", H3.json_safe(index)))
        if run is not None:
            run.outputs(*[p for p in outs if Path(p).exists()])
            run.extra.update({"figures": [r.record() for r in results], "confirmation_half_read": False,
                              "v6_target_rows_scored": 0})
    log(f"figures: {sum(r.status == 'written' for r in results)} written, {sum(r.status == 'skipped' for r in results)} skipped")
    return 0


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
