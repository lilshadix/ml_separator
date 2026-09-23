"""``scripts/g19_run_discovery.py`` -- the resumable discovery runner (pre-registration section 0 order of work item 3;
sections 2, 3, 5, 6, 7, 12, 15, 16; sealed 2026-09-15).

This runner implements **POST-HOC addendum 1** of the sealed text (the compute-driven reduction; see
``discovery.READINGS`` ``addendum1_*``): every V5 design and variant of every learned arm and of B6 tunes on three
SIMULTANEOUS inner folds (``gen19ct.models.inner_design.SimultaneousInnerCells``, one inner fit per configuration per
fold with up to 30 cells hidden at once); discovery runs seed **104729 only**; a learned arm runs only the V5 strict
and HNO3-only refits; V5-PAIR carries M2 (with its M1 prerequisite).

Refuses to start (and to begin each stage) unless ``scripts/g19_seal_prereg.py --check`` exits 0 AND the sealed digest
of ``preregistration.md`` and ``manifests/prereg_sha256.txt`` equals the registered
``discovery.REGISTERED_PREREG_SHA256`` (an edited-and-re-sealed text is refused) AND the number of POST-HOC addenda
below the footer is the one this code implements (``--expect-addenda`` overrides); the sealed digest, the addendum
count and the SHA-256 of the addendum text are written into every fold record and the manifest.  Enumerates the jobs of
``gen19ct.evaluation.discovery.enumerate_plan`` (section 7 order, each pass in the registered arm order: the
nested-certificate safeguard, B6 / B6r0 with the batched-vs-exact and ten-fold checks and their strict / HNO3-only
refits, the seed-104729 pass B5 = M0, FLAT_CAT, B8 -> M1 -> M2, the stop rule, the S1(c) V5-PAIR run, the strict /
HNO3-only refits, conditional freezing-candidate runs; then prints ``M3+ not implemented`` and stops), with a marker
job naming everything the addendum does not run.  Every job is (arm, design, variant, seed = 104729,
half = selection).  Per outer fold:

1. the fold's selection-half scored rows (confirmation-half rows are never predicted; every row's registered half is
   re-derived and asserted), minus the acidic co-extractant rows; ``registered.assert_not_scored`` on them;
2. training rows = MODEL rows (minus Sr(III) for the Sr(III)-dropped sensitivity) minus the fold's hidden rows; the
   outer ``fold_isolation_check`` at the design level (``cell_holdout.guard_v5`` / ``guard_v5p``, ``source_holdout.
   guard_v1``, ``metal_holdout.guard_v2``) before any fit;
3. the arm: nested section 7 tuning on the outer-training rows (addendum 1 items 1-2: 3 inner folds, one inner fit per
   configuration per fold, V5 folds hidden simultaneously by ``simultaneous_inner_design`` and verified by
   ``verify_inner_splits``; the V1 / V2 inner designs unchanged) with the section 15 model seed of the (seed, fold)
   pair, the outer refit, predictions of the scored rows ("point" step), then cross-fitted split-conformal intervals
   ("intervals" step: inner fold j's residuals from refits of the configuration selected without fold j; B6 / B6r0 get
   theirs from ``factorized.B6TunedConformal(calibration="cross_fit")`` in the point step);
4. one parquet + one JSON record per arm (``discovery.fold_paths``) with the selected configuration, inner scores,
   best iterations / epochs, fit seconds, peak RSS and the resume digest; a fold whose record exists with the same
   digest and steps is skipped.

The nested-certificate safeguard (section 2 resolution) runs first on the registered sample of ``folds/INDEX.json``
and records ``evaluation/discovery/safeguard/<stem>.json``; a difference makes every_split mandatory for that design.
After the B6 jobs the batched-vs-exact and ten-fold checks are evaluated and written to
``evaluation/discovery/decisions/plan_state.json``; heavy-arm V5 / V1 jobs wait for them; the checks read only verified
records (digest, fold hash and fold set as the current code would write them).  Compute seconds per job and step and the
cumulative budget against 60 h (section 7 item 5) go to ``manifests/run_info/g19_run_discovery.json``; exhausting the
budget demotes only M3-M7 (not implemented), every job here keeps running.

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_discovery.py --workers 2

``--dry-run`` enumerates the jobs with fold counts and the cost estimate and writes the plan to
``evaluation/discovery/benchmark/plan_addendum1.txt``.  ``--benchmark-addendum1`` times, per arm (B6, B5, FLAT_CAT, B8,
M1, M2), one inner-tuning configuration fit, one calibration refit and one outer refit on the TRAINING rows of one
V5-primary batched fold under the addendum's simultaneous inner design (seed 104729; nothing is predicted on a test
row), reuses the sealed-plan V1 / V2 measurements, and writes
``evaluation/discovery/benchmark/cost_estimate_addendum1.json`` / ``.md`` with per-stage cumulative wall-clock
checkpoints on 2 workers and the fits-in-60-h verdict; ``--benchmark`` is the superseded sealed-plan measurement, kept
so that the 1,535.7 CPU-hour estimate the addendum rests on stays reproducible.  ``--only`` filters (``B6``, ``M2:V5``,
``V1``, ``safeguard``); ``--max-hours`` is an operator pause (no new fold is dispatched once the wall-clock ledger
reaches it; a later invocation resumes; never a registered demotion); ``--steps point`` defers the intervals step.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.data import leakage as L  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.folds import cell_holdout as CH  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.folds import metal_holdout as MH  # noqa: E402
from gen19ct.folds import registered as FR  # noqa: E402
from gen19ct.folds import source_holdout as SH  # noqa: E402
from gen19ct.manifest import Run, write_json, write_text  # noqa: E402
from gen19ct.models import interface as I  # noqa: E402

NAME = "g19_run_discovery"
#: the registry stages this runner writes records of (addendum 2 item 5; registry.READINGS['candidates']): 'discovery'
#: for every job up to the M3+ marker, 'discovery_candidates' for the freezing-candidate jobs of D.STAGES['candidates'],
#: which exist only after the scorer wrote plan_state.freezing_candidates; REG.job_stage(job) says which one a job is
STAGE = REG.DISCOVERY
SEAL_SCRIPT = Path(__file__).resolve().parent / "g19_seal_prereg.py"
PREDICTION_STEP, INTERVAL_STEP = "point", "intervals"
#: files whose content changes predictions (section 16 "feature version" / "model version"); plus the runner objects of
#: gen19ct.evaluation.discovery listed in RUNNER_OBJECTS
CODE_FILES: tuple[Path, ...] = tuple(Path(paths.G19_ROOT / p) for p in (
    "gen19ct/models/interface.py", "gen19ct/models/features.py", "gen19ct/models/boosted.py",
    "gen19ct/models/factorized.py", "gen19ct/models/neural.py", "gen19ct/models/previous_gen.py",
    "gen19ct/models/baselines.py", "gen19ct/models/inner_design.py", "gen19ct/folds/io.py",
    "gen19ct/folds/cell_holdout.py",
    "gen19ct/folds/source_holdout.py", "gen19ct/folds/metal_holdout.py", "gen19ct/folds/registered.py",
    "gen19ct/data/leakage.py", "gen19ct/data/normalize.py", "gen19ct/data/load.py",
    "gen19ct/chemistry/support_graph.py", "gen19ct/chemistry/metals.py", "gen19ct/chemistry/ligands.py",
    "gen19ct/evaluation/support.py",
    "descriptors/metals.csv", "descriptors/extractant_systems.csv", "descriptors/extractant_components.csv"))
RUNNER_OBJECTS: tuple[Any, ...] = (D.JobSpec, D.selection_scored_ids, D.job_folds, D.fittable_folds, D.fold_ordinals,
                                   D.assert_selection_rows, D.registered_row_halves, D.prediction_frame,
                                   D.attach_intervals, D.TuningSplitCalibration, D.InnerFoldSubset, D.FixedInnerSplits,
                                   D.CrossFitResidualConformal, D.calibration_folds, D.assert_no_provenance_features,
                                   D.fold_digest, D.guard_mode_source, D.job_from_record)
PREREG_SEALED = paths.G19_ROOT / "preregistration.md"
PREREG_SHA_FILE = paths.G19_ROOT / "manifests" / "prereg_sha256.txt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rel(path: Path) -> str:
    """``paths.rel`` where it applies (a repository file), the path itself otherwise (``--out-root`` elsewhere)."""
    try:
        return str(paths.rel(Path(path)))
    except ValueError:
        return str(path)


# ============================================================================================= #
# the seal gate (section 0)
# ============================================================================================= #

def seal_check() -> int:
    """Exit code of ``g19_seal_prereg.py --check`` (0 = the sealed file is intact)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = str(paths.G19_ROOT)
    r = subprocess.run([sys.executable, str(SEAL_SCRIPT), "--check"], cwd=str(paths.REPO_ROOT), env=env,
                       capture_output=True, text=True)
    return int(r.returncode)


def _seal_module():
    name = "g19_seal_prereg"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SEAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def prereg_digests(sealed: Path = PREREG_SEALED, sha_file: Path = PREREG_SHA_FILE) -> dict[str, Any]:
    """The sealed text's footer digest, the digest recomputed from the text above the footer, the digest file and the
    SHA-256 of the text below the footer (the POST-HOC addenda), read with the seal script's own functions."""
    sm = _seal_module()
    text = sm.read_normalised(Path(sealed))
    above, footer, below = sm.split_footer(text)
    return {"footer": sm.footer_digest(footer), "recomputed": sm.digest_of_above(above),
            "digest_file": sm.read_digest_file(Path(sha_file)),
            "addenda_sha256": D.sha256_text(below), "n_addenda": sum(1 for ln in below.split("\n")
                                                                     if sm._ADDENDUM.match(ln))}


def refuse_unless_sealed(check: Callable[[], int] | None = None,
                         digests: Callable[[], Mapping[str, Any]] | None = None, *,
                         expect_addenda: int | None = None, stage: str = STAGE) -> dict[str, Any]:
    """Section 0: fit scripts call ``--check`` and refuse to run when it fails; discovery also refuses unless the sealed
    digest (footer, recomputed and ``prereg_sha256.txt``) is the registered ``discovery.REGISTERED_PREREG_SHA256``
    (task X finding V-LP-02: ``--check`` alone accepts any self-consistent re-sealed text).

    The sealed text carries POST-HOC addenda BELOW the footer; ``--check`` accepts them (dated, consecutively numbered)
    and the footer digest is unaffected by them, so the gate also pins how many there are AND their text: the count and
    the SHA-256 of the LF-normalised below-footer text must be the ones REGISTERED FOR ``stage`` in
    ``manifests/digest_registry.json`` (POST-HOC addendum 2 item 5, ``registry.gate_expectations``; the constants
    ``discovery.N_ADDENDA_EXPECTED`` / ``REGISTERED_ADDENDA_SHA256`` only while no registry exists).  Task X finding
    V-F01: an edited seed set, refit list or V5-PAIR scope in an addendum would otherwise pass both ``--check`` and this
    gate.  ``expect_addenda`` / ``--expect-addenda N`` overrides the count, never the digest.  The stage's digest goes
    into every fold record's resume digest and every manifest.  Returns that digest record."""
    exp = REG.gate_expectations(stage)
    rc = (check or seal_check)()
    if rc != 0:
        raise SystemExit(f"refused: scripts/g19_seal_prereg.py --check exited {rc}; discovery never runs on an unsealed "
                         "or modified pre-registration (section 0)")
    try:
        dg = dict((digests or prereg_digests)())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"refused: the sealed pre-registration digest could not be read ({exc})") from None
    want = D.REGISTERED_PREREG_SHA256
    bad = {k: dg.get(k) for k in ("footer", "recomputed", "digest_file") if dg.get(k) != want}
    if bad:
        raise SystemExit(f"refused: the sealed pre-registration is not the registered text {want[:12]}... "
                         f"({ {k: (str(v)[:12] + '...') if v else v for k, v in bad.items()} }); a change to the "
                         "registered analysis is a POST-HOC addendum below the footer, never a re-seal (section 0)")
    want_n = int(exp["n_addenda"]) if expect_addenda is None else int(expect_addenda)
    got_n = int(dg.get("n_addenda") or 0)
    if got_n != want_n:
        raise SystemExit(f"refused: the sealed pre-registration carries {got_n} POST-HOC addendum/addenda below the "
                         f"footer, {want_n} expected for stage {stage!r} ({exp['source']}). Read the addenda; a stage is "
                         "registered under the text its records are written with (gen19ct.evaluation.registry)")
    want_add, got_add = exp["below_footer_sha256"], dg.get("addenda_sha256")
    if got_add != want_add:
        raise SystemExit(f"refused: the POST-HOC addenda below the sealed footer digest to {str(got_add)[:12]}..., not "
                         f"the registered addendum text {str(want_add)[:12]}... of stage {stage!r} ({exp['source']}): "
                         "the addendum text was edited or extended. Nothing runs against an addendum text its stage is "
                         "not registered under (--expect-addenda alone never passes this check; task X finding V-F01; "
                         "addendum 2 item 5)")
    return {"prereg_sha256": want, "addenda_sha256": got_add, "addenda_sha256_registered": want_add,
            "n_addenda": got_n, "n_addenda_expected": want_n, "addendum_implemented": int(exp["n_addenda"]),
            "seal_check_exit": rc, "stage": stage, "gate_source": exp["source"]}


# ============================================================================================= #
# the corpus every fold reads
# ============================================================================================= #

@dataclass
class Corpus:
    """Everything a fold job reads: the arm frame (``interface.prepare_frame`` layout + the registered publication
    group), its slim guard frame, the ``RowTable``, ``V6_TARGET_ROWS``, the acidic co-extractant rows, the registered row
    halves per design, the condition vectors and the parent-structure component map."""

    frame: pd.DataFrame
    slim: pd.DataFrame
    table: I.RowTable
    v6: pd.Series
    coext: pd.Series
    systems: pd.DataFrame | None
    comps: pd.DataFrame | None
    cv: pd.DataFrame | None
    pmap: dict[str, str] | None
    row_half: dict[str, pd.Series]
    folds_dir: Path
    index_json: dict = field(default_factory=dict)
    _folds: dict = field(default_factory=dict, repr=False)
    guard_cache: dict = field(default_factory=dict, repr=False)
    _wildcard: dict = field(default_factory=dict, repr=False)

    def wildcard_pairs(self) -> pd.DataFrame | None:
        """``folds/wildcard_copy_pairs.csv`` (section 2's wildcard copies: the same conditions and value under another
        structure key or state token), read once; ``None`` when the folds directory has none (a synthetic corpus)."""
        if "pairs" not in self._wildcard:
            f = Path(self.folds_dir) / "wildcard_copy_pairs.csv"
            self._wildcard["pairs"] = pd.read_csv(f) if f.exists() else None
        return self._wildcard["pairs"]

    def __post_init__(self) -> None:
        ids = self.frame[FI.ROW_ID].astype(str)
        if not ids.is_unique:
            raise ValueError("canonical_measurement_id must be unique")
        self.idmap = pd.Series(self.frame.index, index=ids.to_numpy())
        self.sr_index = self.frame.index[(self.frame[SG.METAL_COL] == I.SR_III).to_numpy()]
        self.coext_by_id = pd.Series(self.coext.reindex(self.frame.index).fillna(False).to_numpy(dtype=bool),
                                     index=ids.to_numpy())
        self.half_by_id = {d: pd.Series(s.reindex(self.frame.index).to_numpy(), index=ids.to_numpy())
                           for d, s in self.row_half.items()}

    def labels_of(self, ids: Sequence[str]) -> pd.Index:
        ids = [str(r) for r in ids]
        missing = [r for r in ids if r not in self.idmap.index]
        if missing:
            raise AssertionError(f"{len(missing)} row id(s) absent from the corpus (first {missing[0]!r})")
        return pd.Index(self.idmap.loc[ids].to_numpy())

    def folds(self, stem: str) -> list[FI.Fold]:
        if stem not in self._folds:
            self._folds[stem] = FI.read_design(stem, self.folds_dir)
        return self._folds[stem]

    def support_tau(self, stem: str) -> dict:
        key = ("support_tau", stem)
        if key not in self._folds:
            self._folds[key] = FI.read_fold_fields(stem, "support_tau", self.folds_dir)
        return self._folds[key]

    def design_hash(self, stem: str) -> str:
        key = ("design_hash", stem)
        if key not in self._folds:
            self._folds[key] = FI.design_hash(self.folds(stem))
        return self._folds[key]

    @property
    def remainder_groups(self) -> list[str]:
        """Publication groups below 20 MODEL rows (the pooled V1 remainder fold, section 3.2)."""
        vc = self.frame[FI.GROUP_COL].astype(str).value_counts()
        return sorted(vc.index[vc < SH.REGISTERED_MIN_ROWS])

    @property
    def coext_ids(self) -> list[str]:
        return sorted(self.coext_by_id.index[self.coext_by_id.to_numpy(dtype=bool)].astype(str))

    def fittable(self, job: D.JobSpec) -> list[tuple[FI.Fold, int]]:
        """``discovery.fittable_folds``: ``job_folds`` minus the folds whose selection-half scored rows are all acidic
        co-extractant rows."""
        return D.fittable_folds(job, self.folds(job.stem), self.coext_ids)


def coextractant_ids() -> list[str]:
    """The 14 acidic co-extractant rows (section 2), from ``g19_feasibility.load_frame`` as the pre-seal run took them."""
    spec = importlib.util.spec_from_file_location("g19_feasibility", Path(__file__).resolve().parent / "g19_feasibility.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fr, _ = mod.load_frame(6)
    ids = sorted(fr.loc[fr["acidic_coextractant_modifier"].astype(bool), FI.ROW_ID].astype(str))
    if len(ids) != 14:
        raise AssertionError(f"acidic co-extractant rows: {len(ids)} != 14")
    return ids


def load_corpus(coext_ids: Sequence[str], *, with_cv: bool = True) -> Corpus:
    from gen19ct.data import load
    from gen19ct.data import normalize as N

    model = load.load_model_rows()
    fr = I.prepare_frame(model)
    groups = FI.publication_groups(model)
    if not (groups.reindex(fr.index) == fr[I.PUB_GROUP_COL]).all():
        raise AssertionError("prepare_frame pub_group differs from folds.io.publication_groups")
    fr[FI.GROUP_COL] = fr[I.PUB_GROUP_COL]
    systems, comps = I.load_descriptor_tables()
    table = I.RowTable(fr, systems=systems, components=comps)
    v6 = FR.v6_target_mask(model, FR.v6_system_set(model)).reindex(fr.index).fillna(False).astype(bool)
    if int(v6.sum()) != 580 or set(fr.loc[v6.to_numpy(), FI.ROW_ID].astype(str)) != FI.registered_v6_ids():
        raise AssertionError("V6_TARGET_ROWS differ from the registered set")
    coext = fr[FI.ROW_ID].astype(str).isin(set(coext_ids))
    halves = {d: D.registered_row_halves(fr, d) for d in ("V5", "V5P", "V5PAIR", "V1", "V2")}
    cv = N.condition_vector(fr) if with_cv else None
    index_json = json.loads((paths.FOLDS_DIR / "INDEX.json").read_text(encoding="utf-8"))
    return Corpus(frame=fr, slim=FI.slim_frame(fr), table=table, v6=v6, coext=coext, systems=systems, comps=comps, cv=cv,
                  pmap=FR.parent_component_map(), row_half=halves, folds_dir=paths.FOLDS_DIR, index_json=index_json)


# ============================================================================================= #
# guards (section 2)
# ============================================================================================= #

def outer_guard(job: D.JobSpec, fold: FI.Fold, corpus: Corpus, universe: pd.Index) -> list[dict]:
    """The fold builders' guard at the registered level (raises on a violation)."""
    if job.design in ("V5", "V5PAIR"):
        cmap = corpus.pmap if fold.meta.get("parent_structure") else None
        reps = [CH.guard_v5(fold, corpus.slim, component_map=cmap, universe_index=universe)]
    elif job.design == "V5P":
        reps = CH.guard_v5p(fold, corpus.slim)
    elif job.design == "V1":
        reps = [SH.guard_v1(fold, corpus.slim, universe_index=universe)]
    elif job.design == "V2":
        reps = [MH.guard_v2(fold, corpus.slim, universe_index=universe)]
    else:
        raise ValueError(f"no outer guard for {job.design}")
    if not all(r["ok"] for r in reps):
        raise AssertionError(f"{job.key}/{fold.fold_id}: outer guard failed")
    return [{"level": r.get("level"), "ok": bool(r["ok"]), "publication_basis": r.get("publication_basis"),
             "n_train": r.get("n_train"), "n_test": r.get("n_test")} for r in reps]


def inner_isolation_check(job: D.JobSpec, corpus: Corpus) -> Callable[[pd.Index, pd.Index], dict]:
    """``fold_isolation_check`` of an inner split at the job's design level and variant component rule (cell-only: not
    component-aware; parent-structure: the parent component map); V1 on the registered publication group."""
    if job.design in ("V5", "V5P", "V5PAIR"):
        variant = job.variant if job.design == "V5" else "primary"
        kw = dict(level="V5", component_aware=variant != "cell_only",
                  component_map=corpus.pmap if variant == "parent_structure" else None)
        df = corpus.slim
    elif job.design == "V1":
        kw = dict(level="V1")
        df = FI.publication_group_frame(corpus.slim)
    elif job.design == "V2":
        kw = dict(level="V2", element_level=job.variant != "state")
        df = corpus.slim
    else:
        raise ValueError(f"no inner isolation level for {job.design}")

    def check(tr: pd.Index, te: pd.Index) -> dict:
        return L.fold_isolation_check(tr, te, df, near_dup_sig=FI.NEAR_DUP_SIG, near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL,
                                      raise_on_violation=False, **kw)
    return check


# ============================================================================================= #
# arm runners
# ============================================================================================= #

@dataclass
class FoldContext:
    job: D.JobSpec
    fold: FI.Fold
    ordinal: int
    corpus: Corpus
    mask: np.ndarray
    hidden: pd.Index
    sc_ids: list[str]
    sc_labels: pd.Index
    positions: np.ndarray
    ctx: I.FitContext
    out_root: Path
    guard_mode: str
    batching_label: str = ""
    code: str = ""
    state: D.PlanState | None = None
    design_hash: str = ""
    #: the fold's V5 inner design and its one draw of the splits, shared by tuning and calibration
    #: (:func:`inner_splits_v5`)
    inner_cache: dict = field(default_factory=dict, repr=False)
    #: how the CALLER locates and validates the sibling record of the same fold that this fold's arm needs -- today only
    #: M2's M1 record (section 6's "M2 keeps each outer fold's retained M1 hyperparameters").  ``None`` is discovery's
    #: own rule (:meth:`NeuralRunner.m1_record`: ``discovery.fold_paths`` under ``evaluation/discovery/<arm>/<design
    #: file>/s<seed>``, verified against :func:`fold_digest`).  The confirmation runner supplies its own because its
    #: records live under ``evaluation/confirmation/records`` keyed by an OPAQUE seed index and carry no fold digest at
    #: all -- discovery's locator there resolves a path that names the withheld seed and can never exist, so every M2
    #: fold of the single run would raise (task X finding, confirmed end to end).  Discovery's behaviour is unchanged:
    #: the field defaults to ``None`` and no discovery call site sets it.
    sibling_record: Callable[[D.JobSpec, str], dict] | None = field(default=None, repr=False)

    @property
    def model_seed(self) -> int:
        """Section 15 model seed of this (discovery seed, outer fold) pair (``discovery.fold_ordinals``)."""
        return model_seed_of(self.ordinal)


@dataclass
class ArmOutput:
    pred: pd.DataFrame
    selected_config: str
    model_seed: int | None
    record: dict
    intervals: dict | None = None          # set when the point step also produced the intervals (B6, comparators)
    feature_columns: tuple[str, ...] = ()
    seconds: float = 0.0
    shared_with: str | None = None


def model_seed_of(fold_number: int) -> int:
    """Section 15: ``42 + fold * 1009 + 9,999,991`` (``boosted.model_seed_for_fold``; the same rule in every arm)."""
    from gen19ct.models import boosted as BO
    return BO.model_seed_for_fold(int(fold_number))


def inner_variant(job: D.JobSpec) -> str:
    """The V5 variant a job's inner design is built from: the job's own for V5, the PRIMARY one for V5-P, V5-PAIR and V6
    (``inner_design.learned_arm_inner_design`` / ``boosted.inner_design_for`` make the same choice), and ``"state"``
    only for the V2 state-level sensitivity."""
    return job.variant if job.design == "V5" else ("state" if job.variant == "state" else "primary")


def max_cells_per_batch(job: D.JobSpec) -> int | None:
    return 4 if job.scheme.endswith("max4") else None


def boosted_design(job: D.JobSpec):
    """The registered V1 / V2 inner tuning design (``boosted.V1InnerTuning`` / ``V2InnerTuning``; addendum 1 item 1
    leaves them unchanged).  V5 designs come from :func:`simultaneous_inner_design` instead."""
    from gen19ct.models import boosted as BO
    if job.design in ("V1", "V2"):
        return BO.inner_design_for(job.design, job.variant)
    return BO.inner_design_for(job.design, inner_variant(job), max_cells_per_batch=max_cells_per_batch(job))


def exact_scheme(job: D.JobSpec) -> bool:
    return job.scheme in ("exact", "cell_x_group", "cell_pair")


def v5_family(job: D.JobSpec) -> bool:
    """Whether the job's outer design tunes on the V5 inner design (V5, V5-P, V5-PAIR and **V6**).

    POST-HOC addendum 7 item 2: section 7 registers *"V6: V5-style inner cells"* and addendum 1 item 1 makes the V5
    inner design the simultaneous-hiding one, so a V6 job takes the V5 inner design of section 7 as amended by
    addendum 1 item 1 -- the simultaneous-hiding inner cells, their calibration, their guard and their recorded
    signature.  Before that addendum this predicate read ``("V5", "V5P", "V5PAIR")`` and a V6 job would have been
    tuned on the registered V1 / V2 path (:func:`validation_splits`, :func:`_calibration_splits`), mislabelled by
    :func:`inner_design_record` and left without an :func:`inner_design_signature` in its resume digest.  V6 is scored
    on Pr/Nd ROWS rather than V5 cells; the name is kept because the inner design is what every caller here asks
    about.  The edit is logged in ``manifests/digest_registry.json`` and validates or invalidates no record written
    earlier (addendum 2 item 5): no discovery, candidates, ladder, H3 or power job is of design V6."""
    return job.design in ("V5", "V5P", "V5PAIR", "V6")


def simultaneous_inner_design(job: D.JobSpec, corpus: Corpus):
    """POST-HOC addendum 1 items 1-2: the ``interface``-level V5 inner design of every learned arm and of B6, on every
    seed -- ``gen19ct.models.inner_design.SimultaneousInnerCells`` with the job's V5 variant (thresholds, medium,
    component rule), through ``inner_design.learned_arm_inner_design`` so that one implementation serves the fold
    builder, the tuners and the conformal calibration.  Three inner folds, each ONE split with all of its inner cells
    (<= 30) hidden simultaneously under the registered hiding, eligibility re-checked with all of them hidden, a
    failing cell kept hidden but dropped from the inner score.  The inner batching of sections 3.1 / 7 is not used for
    tuning, so ``max_cells_per_batch`` applies to the OUTER folds only."""
    from gen19ct.models import inner_design as ID

    design = ID.learned_arm_inner_design(job.design, inner_variant(job), frame=corpus.frame, n_folds=D.INNER_N_FOLDS,
                                         max_cells_per_fold=D.INNER_MAX_CELLS_PER_FOLD)
    ID.assert_learned_arm_design(design, f"{job.key}: the inner design of a learned arm")
    sig, desc = inner_design_signature(job), design.describe()
    drift = {k: (sig[k], desc.get(k)) for k in sig if k != "module" and desc.get(k) != sig[k]}
    if drift:
        raise AssertionError(f"{job.key}: the inner design differs from the digested signature {drift} (VL-A1-01)")
    return design


def inner_design_object(job: D.JobSpec, corpus: Corpus):
    """The inner design a TUNER is handed (``boosted.inner_design_for``): ``boosted.V5SimultaneousTuning`` -- the same
    simultaneous design as ``TuningSplit`` s -- for every V5 design and variant, and the registered V1 / V2 inner
    designs, which addendum 1 leaves unchanged."""
    return boosted_design(job)


def verify_inner_splits(fc: FoldContext, splits: Sequence[I.InnerSplit]) -> None:
    """Every inner split of a V5 fold before it is fitted: inside the outer training rows, a partition (no calibration
    row in its own training rows), a scored calibration population (known state, not Sr(III), not ``V6_TARGET_ROWS``,
    not an excluded row), one averaging unit per calibration row, no row scored in two inner splits, and the registered
    isolation check under the fold's guard mode -- ``interface.ConformalWrapper._verify``, the check
    ``factorized.run_inner_tuning`` and ``boosted.verify_split`` apply to the other designs."""
    from gen19ct.models import baselines as B

    if not splits:
        raise ValueError(f"{fc.job.key}/{fc.fold.fold_id}: the inner design produced no validation split")
    t, mask = fc.corpus.table, fc.mask
    verifier = I.ConformalWrapper(B.BaselineArm("B0"), splitter=D.FixedInnerSplits(splits), guard=fc.guard_mode)
    ok = I.scorable_mask(t, fc.ctx, known_state_only=True)
    seen = np.zeros(t.n, dtype=bool)
    for sp in splits:
        what = f"{fc.job.key}/{fc.fold.fold_id}/inner {sp.unit}"
        if (sp.train_mask & ~mask).any() or not mask[sp.cal_positions].all() or sp.train_mask[sp.cal_positions].any():
            raise AssertionError(f"{what}: not a partition of the outer training rows")
        if not ok[sp.cal_positions].all():
            raise AssertionError(f"{what}: an X(?), Sr(III), V6_TARGET_ROWS or excluded row would be scored")
        if seen[sp.cal_positions].any():
            raise AssertionError(f"{what}: a calibration row is scored in two inner splits")
        seen[sp.cal_positions] = True
        if sp.row_units is None or len(sp.row_units) != len(sp.cal_positions):
            raise AssertionError(f"{what}: no averaging unit per calibration row (InnerSplit.row_units)")
        FR.assert_not_scored(t.index[sp.cal_positions], fc.ctx.v6_mask, what)
        verifier._verify(t, sp, fc.ctx)


def inner_splits_v5(fc: FoldContext) -> tuple[Any, list[I.InnerSplit]]:
    """The addendum's V5 inner design of this fold and its splits, drawn once with the run seed, verified, and cached on
    the fold context so tuning and calibration share one draw."""
    if fc.inner_cache.get("splits") is None:
        design = simultaneous_inner_design(fc.job, fc.corpus)
        splits = list(design.splits(fc.corpus.table, fc.mask, fc.ctx))
        verify_inner_splits(fc, splits)
        fc.inner_cache.update(design=design, splits=splits)
    return fc.inner_cache["design"], fc.inner_cache["splits"]


def inner_design_record(fc: FoldContext) -> dict[str, Any]:
    """What the fold's V5 inner design did, for the record (``inner_design.splits_summary``): the splits, their cells,
    the calibration rows, the cells that failed the eligibility re-check with the whole inner fold hidden (kept
    hidden, dropped from the inner score) and, per inner fold, the calibration rows whose section 2 wildcard copy sits
    in that split's training rows (a diagnostic; the selection score is unchanged -- ``READINGS['inner_wildcard_copies']``,
    task X finding VL-A1-02)."""
    from gen19ct.models import inner_design as ID

    if not v5_family(fc.job):
        design = boosted_design(fc.job)
        return {"design": getattr(design, "design", fc.job.design),
                "note": "the registered V1 / V2 inner design, unchanged by addendum 1 (already one fit per inner fold)",
                **({"describe": design.describe()} if hasattr(design, "describe") else {})}
    design, splits = inner_splits_v5(fc)
    dropped = dict(getattr(design, "last_dropped", None) or {})
    checks = dict(getattr(design, "last_checks", None) or {})
    pairs = fc.corpus.wildcard_pairs()
    wildcard = (ID.wildcard_copy_partners_in_inner_training(splits, fc.corpus.table,
                                                            fc.corpus.frame[FI.ROW_ID].astype(str), pairs)
                if pairs is not None else {"status": "folds/wildcard_copy_pairs.csv absent: not counted"})
    return {"design": getattr(design, "name", D.INNER_DESIGN_NAME), "n_inner_folds": len(splits),
            "wildcard_copy_partners_in_inner_training": wildcard,
            "wildcard_reading": D.READINGS["inner_wildcard_copies"],
            "splits": ID.splits_summary(splits).to_dict(orient="records"),
            "n_cells_per_fold": [int(len(set(map(str, sp.row_units)))) for sp in splits],
            "n_calibration_rows_per_fold": [int(len(sp.cal_positions)) for sp in splits],
            "n_hidden_rows_per_fold": [int(len(sp.hidden_positions)) for sp in splits],
            "dropped_cells_per_fold": {str(k): [str(c) for c in v] for k, v in dropped.items()},
            "n_dropped_cells": int(sum(len(v) for v in dropped.values())),
            "eligibility_recheck": {str(k): v for k, v in checks.items()},
            "registration_choices": ID.REGISTRATION_CHOICES, "reading": D.READINGS["addendum1_inner_design"]}


def validation_splits(fc: FoldContext) -> list[Any]:
    """The fold's inner splits as ``neural.ValidationSplit`` s (label indices): the addendum's simultaneous design on
    V5, converted by ``neural.splits_from_inner_splits`` from the one draw this fold shares between tuning and
    calibration; the registered boosted inner design on V1 / V2 (``boosted.verify_split`` guards those, as before).
    :func:`v5_family` routes V6 down the first branch (POST-HOC addendum 7 item 2), so a V6 job is tuned on the
    simultaneous-hiding V5 inner cells, never on the V1 / V2 path."""
    from gen19ct.models import boosted as BO
    from gen19ct.models import neural as NN

    c = fc.corpus
    if v5_family(fc.job):
        _, splits = inner_splits_v5(fc)
        return NN.splits_from_inner_splits(c.frame.loc[c.table.index[fc.mask]], c.table, splits,
                                           guard=f"g19_run_discovery.verify_inner_splits:{fc.guard_mode}")
    rows = c.frame.loc[c.table.index[fc.mask]]
    tsplits = D.TuningSplitCalibration(boosted_design(fc.job), c.frame, "full").tuning_splits(c.table, fc.mask, fc.ctx)
    if not tsplits:
        raise ValueError(f"{fc.job.key}/{fc.fold.fold_id}: the inner design produced no validation split")
    for sp in tsplits:
        BO.verify_split(sp, rows, fc.ctx)
    return [NN.ValidationSplit(name=sp.split_id, inner_fold=int(sp.inner_fold), train_index=sp.train_index,
                               valid_index=sp.val_index, valid_units=sp.val_units, hidden_index=sp.hidden_index,
                               guard="boosted.verify_split (fold_isolation_check on every inner split)")
            for sp in tsplits]


def _calibration_splits(fc: FoldContext) -> tuple[Any, list[int]]:
    """The arm's inner splitter of this fold (one draw with the run seed) and the inner folds holding a split: the
    addendum's simultaneous V5 design (V6 included, POST-HOC addendum 7 item 2: its calibration is the V5 one), or the
    tuning splits of the registered V1 / V2 inner design."""
    if v5_family(fc.job):
        _, splits = inner_splits_v5(fc)
        return D.FixedInnerSplits(splits, D.INNER_DESIGN_NAME), sorted({int(s.fold) for s in splits})
    full = D.TuningSplitCalibration(boosted_design(fc.job), fc.corpus.frame, "full")
    avail = sorted({int(s.inner_fold) for s in full.all_splits(fc.corpus.table, fc.mask, fc.ctx)})
    return full, avail


def _restricted(full: Any, folds: Sequence[int]) -> D.InnerFoldSubset:
    """``full`` restricted to the calibration inner folds (addendum 1 item 2: fold ``j``'s residuals from fold ``j``'s
    splits, under the configuration selected without it)."""
    return D.InnerFoldSubset(full, folds)


def _check_tuned_folds(fc: FoldContext, plan: Mapping[str, Any], used: Sequence[int]) -> None:
    if sorted(int(f) for f in used) != list(plan["tuning_folds"]):
        raise AssertionError(f"{fc.job.key}/{fc.fold.fold_id}: tuning used inner folds {sorted(used)}, the calibration "
                             f"plan expects {plan['tuning_folds']}")


class B6Runner:
    """B6 and B6r0 from one set of inner fits (``factorized.fit_b6_and_b6r0``): inner selection on the design's averaging
    units, cross-fitted split-conformal intervals, the section 15 model seed for the factor initialisation."""

    arms = D.B6_ARMS
    has_interval_step = False

    def __init__(self) -> None:
        self._registered: set[int] = set()

    def splitter(self, fc: FoldContext):
        """The inner design of the fold (addendum 1 item 1: the simultaneous V5 design for every V5 design, variant and
        outer scheme -- exact and batched alike; the registered V1 / V2 designs unchanged).  Every split carries its
        calibration rows' averaging units (``InnerSplit.row_units``), which B6 selects on."""
        job = fc.job
        if v5_family(job):
            return D.FixedInnerSplits(inner_splits_v5(fc)[1], D.INNER_DESIGN_NAME)
        return I.GroupKFoldCalibration(3, unit="v1_unit") if job.design == "V1" else I.InnerMetalCalibration(3, 100, 5)

    def point(self, fc: FoldContext) -> dict[str, ArmOutput]:
        from gen19ct.models import factorized as FZ

        t = fc.corpus.table
        if id(t) not in self._registered:
            FZ.register_table(t, fc.corpus.frame)
            self._registered.add(id(t))
        t0 = time.perf_counter()
        ms = fc.model_seed
        fitted = FZ.fit_b6_and_b6r0(t, fc.mask, fc.ctx, self.splitter(fc), guard=fc.guard_mode,
                                    inner_mode=fc.job.inner_mode, init_seed=ms, require_row_units=True)
        secs = time.perf_counter() - t0
        out = {}
        for v in D.B6_ARMS:
            w = fitted[v]
            pred = w.predict_positions(fc.positions)
            d = w.fitted_arm.design
            cols = tuple(d.x_names) + tuple(d.zm_names) + tuple(d.zs_names)
            D.assert_no_provenance_features(cols, f"{v} inputs")
            rec = {"selection": w.selection_record(), "tuning_summary": w.tuning.summary().to_dict(orient="records"),
                   "inner_design": inner_design_record(fc), "registration_choices": FZ.REGISTRATION_CHOICES}
            cal = w.calibration_record
            out[v] = ArmOutput(pred=pred, selected_config=f"k{w.selected[0]}_lam{w.selected[1]:g}", model_seed=ms,
                               record=rec, feature_columns=cols, seconds=secs if v == "B6" else 0.0,
                               shared_with=None if v == "B6" else "B6",
                               intervals={"n_calibration": int(len(w.residuals)), "n_inner_splits": len(w.calibration_units),
                                          "quantiles": {str(k): q for k, q in w.quantiles.items()},
                                          "guard": fc.guard_mode, "seed": w.fit_seed, "status": cal.get("status"),
                                          "calibration": cal,
                                          "method": "factorized.B6TunedConformal(calibration='cross_fit')"})
        return out


class BoostedRunner:
    """B5 (= M0) and FLAT_CAT (``models.boosted``)."""

    has_interval_step = True

    def __init__(self, name: str):
        self.name = name
        self.arms = (name,)

    def point(self, fc: FoldContext) -> dict[str, ArmOutput]:
        from gen19ct.models import boosted as BO

        c = fc.corpus
        rows = c.frame.loc[c.table.index[fc.mask]]
        design = inner_design_object(fc.job, c)
        if v5_family(fc.job):
            inner_splits_v5(fc)                         # one draw, verified, shared with the calibration step
        arm = BO.BoostedArm(self.name, fold_index=fc.ordinal, design=design, inner_mode=fc.job.inner_mode,
                            frame=c.frame, cv=c.cv, max_cells_per_batch=max_cells_per_batch(fc.job))
        t0 = time.perf_counter()
        arm.fit(rows, fc.ctx)
        pred = arm.predict(c.frame.loc[fc.sc_labels])
        secs = time.perf_counter() - t0
        D.assert_no_provenance_features(arm.fitted.columns, f"{self.name} features")
        rec = arm.record()
        return {self.name: ArmOutput(pred=pred, selected_config=arm.fitted.config.label, model_seed=arm.model_seed,
                                     record={"arm": rec, "frozen": {"config": arm.fitted.config.record(),
                                                                    "iterations": arm.fitted.iterations},
                                             "inner_design": inner_design_record(fc)},
                                     feature_columns=tuple(arm.fitted.columns), seconds=secs)}

    def calibration(self, fc: FoldContext, rec: Mapping[str, Any]) -> tuple[D.CrossFitResidualConformal | None, dict]:
        """Cross-fitted calibration (``discovery.READINGS['heavy_arm_intervals']``)."""
        from gen19ct.models import boosted as BO

        c = fc.corpus
        full, avail = _calibration_splits(fc)
        plan = D.calibration_folds(avail, fc.job.inner_mode, tuned=True)
        tuning = rec["arm"]["tuning"]
        _check_tuned_folds(fc, plan, tuning["inner_folds_used"])
        if not plan["calibration_folds"]:
            return None, plan
        arms, sels = {}, {}
        design = inner_design_object(fc.job, c)
        for j in plan["calibration_folds"]:
            if plan["cross_fit"]:
                cfg, iters, sel = BO.select_excluding_folds(tuning, (j,))
            else:
                cfg = BO.CatBoostConfig(**{k: v for k, v in rec["frozen"]["config"].items() if k != "label"})
                iters = int(rec["frozen"]["iterations"])
                sel = {"selected": cfg.label, "iterations": iters, "tuned_on_inner_folds": plan["tuning_folds"]}
            arms[j] = BO.BoostedArm(self.name, fold_index=fc.ordinal, design=design, config=cfg, iterations=iters,
                                    frame=c.frame, cv=c.cv, max_cells_per_batch=max_cells_per_batch(fc.job))
            sels[j] = sel
        return D.CrossFitResidualConformal(arms, _restricted(full, plan["calibration_folds"]), selections=sels), plan


class B8Runner:
    """B8 level arm (``models.previous_gen.B8Level``; fixed settings, no tuning)."""

    arms = ("B8",)
    has_interval_step = True

    def point(self, fc: FoldContext) -> dict[str, ArmOutput]:
        from gen19ct.models import previous_gen as PG

        c = fc.corpus
        rows = c.frame.loc[c.table.index[fc.mask]]
        t0 = time.perf_counter()
        arm = PG.B8Level(c.frame, cv=c.cv, fold=fc.ordinal).fit(rows, fc.ctx)
        pred = arm.predict(c.frame.loc[fc.sc_labels])
        secs = time.perf_counter() - t0
        cols = tuple(arm.model.design_columns(arm.wide_columns))
        D.assert_no_provenance_features(cols, "B8 design columns")
        return {"B8": ArmOutput(pred=pred, selected_config="MONO_ET_registered", model_seed=arm.random_state,
                                record={"state": arm.state(), "inner_design": inner_design_record(fc),
                                        "registration_choices": PG.REGISTRATION_CHOICES},
                                feature_columns=(), seconds=secs)}

    def calibration(self, fc: FoldContext, rec: Mapping[str, Any]) -> tuple[D.CrossFitResidualConformal | None, dict]:
        """B8 has no tuning: the inner splits of its inner mode, the registered forest on each."""
        from gen19ct.models import previous_gen as PG

        full, avail = _calibration_splits(fc)
        plan = D.calibration_folds(avail, fc.job.inner_mode, tuned=False)
        if not plan["calibration_folds"]:
            return None, plan
        arms = {j: PG.B8Level(fc.corpus.frame, cv=fc.corpus.cv, fold=fc.ordinal) for j in plan["calibration_folds"]}
        return D.CrossFitResidualConformal(arms, _restricted(full, plan["calibration_folds"])), plan


class NeuralRunner:
    """M1 / M2 (``models.neural``) on the boosted inner design's splits (identical inner splits for every heavy arm)."""

    has_interval_step = True

    def __init__(self, step: str):
        self.step = step
        self.arms = (step,)

    @staticmethod
    def m1_job(job: D.JobSpec) -> D.JobSpec:
        from dataclasses import replace
        return replace(job, arm="M1", writes=("M1",))

    def m1_record(self, fc: FoldContext) -> dict:
        """M1's record of the same fold, refused unless its digest is the one the current code, fold file and plan state
        give M1 (``discovery.READINGS['record_verification']``).

        ``fc.sibling_record`` overrides the location AND the staleness rule when the caller owns both: the confirmation
        runner's records are under ``evaluation/confirmation/records/<arm>/<design file>/i<k>`` and carry ``code_digest``
        / ``fold_hash`` / ``seed_index`` instead of a fold digest, so it supplies its own locator with the equivalent
        check.  Unset -- every discovery call site -- this is discovery's own path, byte for byte as before.
        """
        mjob = self.m1_job(fc.job)
        locate = getattr(fc, "sibling_record", None)     # getattr, not fc.sibling_record: a test stands a namespace in
        if locate is not None:
            return locate(mjob, "M1")
        pq, js = D.fold_paths(fc.out_root, mjob, "M1", fc.fold.fold_id)
        rec = D.read_record(js)
        if rec is None or not pq.exists() or "point" not in (rec.get("steps") or {}):
            raise RuntimeError(f"{fc.job.key}/{fc.fold.fold_id}: M2 needs M1's per-fold values (run the M1 job of this "
                               "design, variant and seed first; section 6 resolution)")
        want = fold_digest(mjob, fc.fold, fc.code, fc.state or D.PlanState(), NeuralRunner("M1"), fc.out_root,
                           ordinal=fc.ordinal, design_hash=fc.design_hash)
        if rec.get("digest") != want or rec.get("fold_hash") != fc.fold.fold_hash:
            raise D.StaleRecordError(f"{fc.job.key}/{fc.fold.fold_id}: M1's record is stale (digest or fold hash differs "
                                     "from the current code, fold file and plan state); rerun the M1 job first")
        return rec

    def point(self, fc: FoldContext) -> dict[str, ArmOutput]:
        from gen19ct.models import neural as NN

        c = fc.corpus
        rows = c.frame.loc[c.table.index[fc.mask]]
        vs = validation_splits(fc)
        t0 = time.perf_counter()
        if self.step == "M1":
            res = NN.tune_m1(rows, vs, fold_index=fc.ordinal, condition_vectors=c.cv)
            m1_used = None
        else:
            m1 = self.m1_record(fc)
            sel = m1["arm_record"]["selected"]
            m1_used = NN.NeuralConfig(int(sel["emb_dim"]), float(sel["weight_decay"]), 0)
            res = NN.tune_m2(rows, vs, m1_used, fold_index=fc.ordinal, condition_vectors=c.cv)
        arm = res.arm(rows=c.frame, condition_vectors=c.cv).fit(rows)
        pred = arm.predict(c.frame.loc[fc.sc_labels])
        secs = time.perf_counter() - t0
        cols = tuple(col for cs in arm.encoder.block_columns.values() for col in cs)
        D.assert_no_provenance_features(cols, f"{self.step} inputs")
        fits = res.fits[[k for k in ("config", "split", "inner_fold", "best_epoch", "epochs_run", "valid_macro_mae",
                                     "n_parameters", "seconds") if k in res.fits.columns]]
        rec = {"selected": asdict(res.selected), "n_epochs": int(res.n_epochs), "model_seed": int(res.model_seed),
               "scores": res.scores.to_dict(orient="records"), "fits": fits.to_dict(orient="records"),
               "split_unit_errors": res.split_unit_errors.to_dict(orient="records"),
               "inner_folds_used": sorted({int(sp.inner_fold) for sp in vs}),
               "fit_record": arm.fit_record(), "m1_config_used": None if m1_used is None else asdict(m1_used),
               "m1_record_digest": None if m1_used is None else m1.get("digest"),
               "inner_design": inner_design_record(fc), "registration_choices": NN.REGISTRATION_CHOICES}
        return {self.step: ArmOutput(pred=pred, selected_config=res.selected.label(), model_seed=int(res.model_seed),
                                     record=rec, feature_columns=cols, seconds=secs)}

    def calibration(self, fc: FoldContext, rec: Mapping[str, Any]) -> tuple[D.CrossFitResidualConformal | None, dict]:
        """Cross-fitted calibration (``discovery.READINGS['heavy_arm_intervals']``); M2's M1 values stay the outer
        fold's (disclosed)."""
        from gen19ct.models import neural as NN

        full, avail = _calibration_splits(fc)
        plan = D.calibration_folds(avail, fc.job.inner_mode, tuned=True)
        _check_tuned_folds(fc, plan, rec["inner_folds_used"])
        if not plan["calibration_folds"]:
            return None, plan
        arms, sels = {}, {}
        for j in plan["calibration_folds"]:
            if plan["cross_fit"]:
                cfg, epochs, sel = NN.select_excluding_folds(rec["scores"], rec["fits"], rec["split_unit_errors"], (j,))
            else:
                s = rec["selected"]
                cfg, epochs = NN.NeuralConfig(int(s["emb_dim"]), float(s["weight_decay"]), int(s["rank"])), int(rec["n_epochs"])
                sel = {"config": cfg.label(), "n_epochs": epochs, "tuned_on_inner_folds": plan["tuning_folds"]}
            arms[j] = NN.FactorisedArm(cfg, n_epochs=epochs, model_seed=int(rec["model_seed"]), rows=fc.corpus.frame,
                                       condition_vectors=fc.corpus.cv)
            sels[j] = sel
        return D.CrossFitResidualConformal(arms, _restricted(full, plan["calibration_folds"]), selections=sels), plan

    def digest_extra(self, job: D.JobSpec, fold: FI.Fold, code: str, state: D.PlanState, out_root: Path, *,
                     ordinal: int, design_hash: str) -> dict:
        if self.step != "M2":
            return {}
        return {"m1_expected_digest": fold_digest(self.m1_job(job), fold, code, state, NeuralRunner("M1"), out_root,
                                                  ordinal=ordinal, design_hash=design_hash)}


class ComparatorIntervalsRunner:
    """B0 / B3 / B3x / B3i with split-conformal intervals drawn with the job's discovery seed (section 15 resolution).

    The splitter is the job's design's registered inner calibration design.  POST-HOC addendum 7 item 2: a **V6** job
    takes the V5 one, not the V2 metal-holdout one the old ``else`` branch gave it -- section 3.4 defines V6 as the V5
    state-level rule of section 3.1, so the B3x / B3i yardsticks of the single V6 run are calibrated on V5 inner cells.
    The branch is ``("V5", "V6")`` and deliberately NOT :func:`v5_family`: V5-P and V5-PAIR comparator-interval jobs
    exist (``B3x@V5PAIR__primary__batched`` is an S1(c) yardstick of the confirmation plan) and reach the ``else``
    branch today, and addendum 7 item 2 authorises the V6 correction only -- widening this branch would change the
    S1(c) yardstick intervals, which no addendum registers.  That V5-PAIR reading is reported, not silently altered."""

    has_interval_step = False

    def __init__(self, name: str):
        self.name = name
        self.arms = (name,)

    def point(self, fc: FoldContext) -> dict[str, ArmOutput]:
        from gen19ct.models import baselines as B

        job = fc.job
        if job.design in ("V5", "V6"):
            spl = I.InnerCellCalibration(10, 1, 3, 3, 30, component_aware=True)
        elif job.design == "V1":
            spl = I.GroupKFoldCalibration(3)
        else:
            spl = I.InnerMetalCalibration(3, 100, 5)
        t0 = time.perf_counter()
        w = I.ConformalWrapper(B.BaselineArm(self.name), splitter=spl, guard=fc.guard_mode).fit_table(
            fc.corpus.table, fc.mask, fc.ctx)
        pred = w.predict_positions(fc.positions)
        secs = time.perf_counter() - t0
        return {self.name: ArmOutput(pred=pred, selected_config="closed_form", model_seed=None,
                                     record={"note": "deterministic comparator; point predictions equal the pre-seal run's"},
                                     seconds=secs, intervals={"n_calibration": int(len(w.residuals)),
                                                              "n_inner_splits": len(w.calibration_units),
                                                              "quantiles": {str(k): q for k, q in w.quantiles.items()},
                                                              "guard": fc.guard_mode, "seed": w.fit_seed,
                                                              "status": "calibrated",
                                                              "method": "interface.ConformalWrapper"})}


def default_runners() -> dict[str, Any]:
    return {"B6": B6Runner(), "B5": BoostedRunner("B5"), "FLAT_CAT": BoostedRunner("FLAT_CAT"), "B8": B8Runner(),
            "M1": NeuralRunner("M1"), "M2": NeuralRunner("M2"),
            **{f"comparator:{a}": ComparatorIntervalsRunner(a) for a in D.COMPARATOR_INTERVAL_ARMS}}


def runner_for(job: D.JobSpec, runners: Mapping[str, Any]):
    key = f"comparator:{job.arm}" if job.kind == "comparator_intervals" else job.arm
    if key not in runners:
        raise KeyError(f"no runner for {key}")
    return runners[key]


# ============================================================================================= #
# section 13 support features and domain status (target-free; one file per fold file, seed and fold)
# ============================================================================================= #

SUPPORT_FEATURES: tuple[str, ...] = (
    "exact_pair_rows", "exact_pair_publications", "alias_unknown_state_rows", "n_neighbour_metals",
    "n_neighbour_metals_same_category", "n_neighbour_metals_same_charge", "system_family", "n_same_family_rows_for_metal",
    "n_same_family_other_system_rows_for_metal", "nearest_radius_metal", "nearest_radius_distance_A",
    "nearest_radius_same_charge", "nearest_radius_same_species_charge", "nearest_radius_basis",
    "n_radius_neighbours_within_tol", "radius_bracket_lower_metal", "radius_bracket_upper_metal",
    "radius_bracketed_same_charge", "nearest_ligand_system", "nearest_ligand_tanimoto", "n_systems_for_metal",
    "n_publications_pair", "n_publications_system", "n_publications_metal", "condition_distance_system",
    "condition_distance_system_same_acid", "condition_distance_pair", "condition_dims_used", "n_series_neighbours_pm1",
    "n_series_neighbours_pm2", "series_bracketed")


def support_paths(out_root: Path, job: D.JobSpec, fold_id: str) -> tuple[Path, Path]:
    d = D.discovery_root(out_root) / "_support" / job.design_dir / f"s{job.seed}"
    name = D.safe_fold_name(fold_id)
    return d / f"{name}.parquet", d / f"{name}.json"


def fold_support_frame(fc: FoldContext) -> tuple[pd.DataFrame, dict]:
    """Section 13 for every scored row of the fold: ``SupportIndex`` features fitted on the fold's training rows, the
    per-fold thresholds (checked against the fold builder's ``support_tau``), the registered ``support_score``
    (``support.s4_registered``) and the domain-status labels under both readings of the anion clause (``ambiguous`` when
    they differ) -- the pre-seal ``fold_support`` rule.  Never reads ``log_D``."""
    from collections import Counter

    from gen19ct.evaluation import support as ES
    from gen19ct.models import baselines as B

    c, t, mask = fc.corpus, fc.corpus.table, fc.mask
    sup = c.frame[list(SG.REQUIRED_COLUMNS)]
    train = sup.loc[t.index[mask]]
    si = SG.SupportIndex(train, systems=c.systems)
    tau_in, tau_ext, tau_max, n_tau = ES.tau_thresholds(si, train)
    builder = c.support_tau(fc.job.stem).get(fc.fold.fold_id)
    mine = {"tau_in": tau_in, "tau_ext": tau_ext, "tau_max": tau_max, "n_tau_rows": n_tau}
    if builder is not None and not fc.job.drop_sr:
        if not all((np.isnan(mine[k]) and builder[k] is None) or mine[k] == builder[k] for k in mine):
            raise AssertionError(f"{fc.fold.fold_id}: tau {mine} differ from the fold builder's {builder}")
    eng = B.make_engine(t, mask, fc.ctx)
    mu, sd = eng._d_desc_scale(B._BASE)
    row_family = np.array([t.sys_family[s] if s >= 0 else None for s in t.sys], dtype=object)
    row_expert = np.array([t.sys_expert[s] if s >= 0 else None for s in t.sys], dtype=object)
    temp = pd.to_numeric(c.frame[SG.TEMP_COL], errors="coerce").to_numpy(dtype=float)
    fam_counts, exp_counts = Counter(row_family[mask]), Counter(row_expert[mask])
    mp = np.flatnonzero(mask & (t.sys >= 0))
    sys_anion = set(zip(t.sys[mp].tolist(), t.anion[mp].tolist()))
    fam_anion = set(zip(row_family[mp].tolist(), t.anion[mp].tolist()))
    fam_systems: dict = {}
    for s_code in np.unique(t.sys[mp]):
        fam_systems.setdefault(t.sys_family[s_code], []).append(t.sys_labels[s_code])
    feat_cache, s4_cache, recs = {}, {}, []
    for p in fc.positions:
        m, s = t.state_labels[t.state[p]], t.sys_labels[t.sys[p]]
        anion = t.anion_labels[t.anion[p]]
        acid, ext, tp = float(t.acid[p]), float(t.ext[p]), float(temp[p])
        key = (m, s, anion, repr(acid), repr(ext), repr(tp))
        if key not in feat_cache:
            cond = {SG.LOG_ACID_COL: acid, SG.LOG_EXT_COL: ext, SG.TEMP_COL: tp,
                    SG.ACID_ANION_COL: None if anion == I.NA_ANION else anion}
            feat_cache[key] = (si.features(m, s, cond), cond)
        f, cond = feat_cache[key]
        st = t.static(s)
        fam, expert = st["family"], st["expert"]
        if (m, s) not in s4_cache:
            mc = t.state_code.get(m)
            cands = []
            for code, pp in (t.systems_by_state.get(mc, []) if mc is not None else []):
                n_rows = int(mask[pp].sum())
                if n_rows:
                    cands.append((t.sys_labels[code], t.sys_fp[code], (t.sys_desc[code] - mu) / sd, n_rows))
            s4_cache[(m, s)] = ES.s4_registered(s, st["fp"], (st["desc"] - mu) / sd, cands)
        comps = ES.support_components(f, s4=s4_cache[(m, s)], tau_ext=tau_ext,
                                      metal_series=SG.metal_properties(m)["series"])
        system_present, metal_present = f["n_publications_system"] > 0, f["n_systems_for_metal"] > 0
        family_rows = int(fam_counts.get(fam, 0)) if fam is not None else 0
        mechanism_rows = int(exp_counts.get(expert, 0)) if expert not in (None, I.NO_EXPERT) else 0
        ac = t.anion_code.get(anion)
        a_sys, a_fam = (t.sys_code.get(s), ac) in sys_anion, (fam, ac) in fam_anion
        fam_cd = float("nan")
        if not system_present and fam in fam_systems:
            ds = [si.condition_distance(k, cond)[0] for k in fam_systems[fam] if k != s]
            ds = [d for d in ds if np.isfinite(d)]
            fam_cd = min(ds) if ds else float("nan")
        base = dict(system_present=system_present, metal_present=metal_present, family_rows=family_rows,
                    mechanism_rows=mechanism_rows, expert=expert, f=f, fam_cd=fam_cd, tau=(tau_in, tau_ext, tau_max))
        sa = ES.domain_status(**base, anion_unseen=(not a_sys) if system_present else (not a_fam))
        sb = ES.domain_status(**base, anion_unseen=not (a_sys or a_fam))
        rec = {"row_id": t.ids[p], "fold_id": fc.fold.fold_id, "seed": int(fc.job.seed)}
        rec.update({k: f[k] for k in SUPPORT_FEATURES})
        rec.update({f"support_{k}": v for k, v in comps.items()})
        rec.update({"support_score": ES.support_score(comps), "family_rows": family_rows, "mechanism_rows": mechanism_rows,
                    "domain_status": sa[0], "domain_status_reading_neither": sb[0],
                    "domain_status_ambiguous": sa[0] != sb[0], "condition_extrapolated": sa[1],
                    "both_nodes_new": sa[2], "conditions_unknown": sa[3], "tau_in": tau_in, "tau_ext": tau_ext,
                    "tau_max": tau_max, "n_tau_rows": n_tau})
        recs.append(rec)
    info = {"tau": mine, "builder_tau": builder, "builder_tau_checked": builder is not None and not fc.job.drop_sr,
            "n_rows": len(recs), "n_domain_status_ambiguous": int(sum(r["domain_status_ambiguous"] for r in recs)),
            "s4_reading": ES.S4_READING}
    return pd.DataFrame(recs), info


def support_digest(job: D.JobSpec, fold: FI.Fold, code: str) -> str:
    """The resume / verification digest of one fold's section 13 support file."""
    sjob = D.JobSpec(kind="fit", arm="support", design=job.design, variant=job.variant, scheme=job.scheme,
                     seed=job.seed, fold_seed=job.fold_seed, writes=("support",))
    return D.fold_digest(sjob, fold, code)


def ensure_support(fc: FoldContext, code: str) -> dict:
    """Write the fold's support file once per (fold file, variant, seed, fold) unless a record with the digest exists.
    Sr(III)-dropped training sets are skipped (their thresholds are not the fold builder's, as pre-seal)."""
    if fc.job.drop_sr:
        return {"status": "skipped_sr_iii_dropped_training"}
    digest = support_digest(fc.job, fc.fold, code)
    pq, js = support_paths(fc.out_root, fc.job, fc.fold.fold_id)
    rec = D.read_record(js)
    if rec is not None and rec.get("digest") == digest and pq.exists():
        return {"status": "skipped_done"}
    t0 = time.perf_counter()
    frame, info = fold_support_frame(fc)
    D.assert_v6_clean(fc.corpus.labels_of(frame["row_id"]), fc.corpus.v6, f"{fc.job.key}/{fc.fold.fold_id} support")
    _atomic_parquet(frame, pq)
    _atomic_json({"schema": D.SCHEMA, "digest": digest, "fold_id": fc.fold.fold_id, "fold_hash": fc.fold.fold_hash,
                  "stem": fc.job.stem, "seconds": round(time.perf_counter() - t0, 3), **info}, js)
    return {"status": "written", "seconds": round(time.perf_counter() - t0, 3)}


# ============================================================================================= #
# one fold
# ============================================================================================= #

def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    paths.ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.reset_index(drop=True).to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def _atomic_json(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_json(tmp, obj)
    os.replace(tmp, path)


def guard_mode_for(job: D.JobSpec, state: D.PlanState) -> str:
    """``nested_certificate`` unless the safeguard made every_split mandatory, or its record for the design is
    missing (then every_split, the conservative mode)."""
    src = D.guard_mode_source(job.stem)
    mode = state.guard_mode.get(src)
    return "nested_certificate" if mode == "nested_certificate" else "every_split"


def inner_design_signature(job: D.JobSpec) -> dict[str, Any] | None:
    """What a fit job's V5 inner design is resolved from, for the resume digest (task X finding VL-A1-01: the two
    module constants ``discovery.INNER_N_FOLDS`` / ``INNER_MAX_CELLS_PER_FOLD`` are not digested code, and the inner
    cells, calibration rows and selection would change silently with them): the design's module and name, the
    variant's thresholds, medium and component rule, the inner folds and the cells-per-fold cap -- the fields of
    ``SimultaneousInnerCells.describe()`` that :func:`simultaneous_inner_design` checks it against.  ``None`` for a
    job without a V5 inner design (V1 / V2 tune on the registered designs, which live in the digested ``boosted.py``;
    safeguards and markers).

    The table this returns has one entry per design of :func:`v5_family`: V5 (each variant), V5-P, V5-PAIR and -- POST-HOC
    addendum 7 item 2 -- **V6**, whose entry is the V5 PRIMARY variant's (``inner_variant`` maps every non-V5 design to
    ``"primary"``), so a V6 fold's resume digest carries the inner-design signature and ``simultaneous_inner_design``
    runs its addendum-1 drift check on the V6 path too.  No V5 entry changes, so no existing fold digest moves."""
    if job.kind != "fit" or not v5_family(job):
        return None
    v = CH.VARIANTS[inner_variant(job)]
    return {"module": D.INNER_DESIGN_MODULE, "inner_design": D.INNER_DESIGN_NAME, "thresholds": v.thresholds.tag,
            "medium": v.medium, "component_aware": bool(v.component_aware), "parent_structure": bool(v.parent_structure),
            "n_inner": int(D.INNER_N_FOLDS), "max_cells": int(D.INNER_MAX_CELLS_PER_FOLD)}


def fold_digest(job: D.JobSpec, fold: FI.Fold, code: str, state: D.PlanState, runner: Any, out_root: Path, *,
                ordinal: int, design_hash: str) -> str:
    """The resume and verification digest of one fold (``discovery.fold_digest``) with the runner's extras: the inner
    guard mode, the batching label, the section 15 model fold number and model seed, the fold file's design hash, the
    registered pre-registration digest AND the addenda digest of the JOB'S registry stage
    (``registry.below_footer_sha256(REG.job_stage(job))``: the addendum-1 digest for a discovery job, the two-addenda
    digest for a freezing-candidate job; task X finding V-F01, addendum 2 item 5), the resolved V5 inner
    design (:func:`inner_design_signature`; finding VL-A1-01) and, for M2, M1's expected digest of the same fold."""
    extra = {"guard_mode": guard_mode_for(job, state), "batching_label": batching_label(job, state),
             "model_fold_number": int(ordinal), "model_seed": model_seed_of(ordinal) if job.kind == "fit" else None,
             "design_hash": str(design_hash), "prereg_sha256": D.REGISTERED_PREREG_SHA256,
             "prereg_addenda_sha256": REG.below_footer_sha256(REG.job_stage(job)),
             "inner_design": inner_design_signature(job)}
    if hasattr(runner, "digest_extra"):
        extra.update(runner.digest_extra(job, fold, code, state, out_root, ordinal=ordinal, design_hash=design_hash))
    return D.fold_digest(job, fold, code, extra)


def batching_label(job: D.JobSpec, state: D.PlanState) -> str:
    if job.kind == "fit" and job.design == "V5" and job.arm in D.HEAVY_ARMS and job.scheme.startswith("batched"):
        return state.heavy_v5_label
    return ""


def prepare_fold(job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Corpus, out_root: Path, state: D.PlanState,
                 guard_fn: Callable = outer_guard, inner_check: Callable = inner_isolation_check, *, code: str = ""
                 ) -> tuple[FoldContext, dict]:
    """Section 2 guards and the outer training mask of one fold (no fit)."""
    what = f"{job.key}/{fold.fold_id}"
    if fold.half == D.CONFIRMATION:
        raise AssertionError(f"{what}: a confirmation-half fold is never fitted in discovery")
    sel = list(D.selection_scored_ids(fold))
    sc_ids = [r for r in sel if not bool(corpus.coext_by_id.get(r, False))]
    if not sc_ids:
        raise ValueError(f"{what}: no selection-half scored row")
    D.assert_selection_rows(sc_ids, corpus.half_by_id[job.design], what)
    sc_labels = corpus.labels_of(sc_ids)
    D.assert_v6_clean(sc_labels, corpus.v6, what)
    st = corpus.frame.loc[sc_labels, SG.METAL_COL]
    if st.isna().any() or (st == I.SR_III).any():
        raise AssertionError(f"{what}: an X(?) or Sr(III) row would be scored")
    hidden = corpus.labels_of(fold.hidden_row_ids)
    if not sc_labels.isin(hidden).all():
        raise AssertionError(f"{what}: a scored row is not hidden")
    sr_extra = corpus.sr_index.difference(hidden) if job.drop_sr else pd.Index([], dtype=hidden.dtype)
    universe = corpus.frame.index.difference(sr_extra)
    guards = guard_fn(job, fold, corpus, universe)
    t = corpus.table
    mask = np.ones(t.n, dtype=bool)
    mask[t.positions(hidden)] = False
    if len(sr_extra):
        mask[t.positions(sr_extra)] = False
    if mask[t.positions(sc_labels)].any():
        raise AssertionError(f"{what}: a scored row is a training row")
    gmode = guard_mode_for(job, state)
    cache = corpus.guard_cache.setdefault((job.design, job.variant, job.drop_sr), {})
    ctx = I.FitContext(systems=corpus.systems, components=corpus.comps, table=t, hidden_index=hidden.union(sr_extra),
                       v6_mask=corpus.v6, exclude_from_scoring=corpus.coext,
                       isolation_check=inner_check(job, corpus), guard_cache=cache, seed=int(job.seed))
    fc = FoldContext(job=job, fold=fold, ordinal=ordinal, corpus=corpus, mask=mask, hidden=hidden, sc_ids=sc_ids,
                     sc_labels=sc_labels, positions=t.positions(sc_labels), ctx=ctx, out_root=out_root, guard_mode=gmode,
                     batching_label=batching_label(job, state), code=code, state=state,
                     design_hash=corpus.design_hash(job.stem))
    info = {"n_hidden": len(hidden), "n_train": int(mask.sum()), "n_scored_selection": len(sc_ids),
            "n_excluded_from_scoring": len(sel) - len(sc_ids), "n_sr_iii_dropped": int(len(sr_extra)),
            "outer_guard": guards, "inner_guard_mode": gmode}
    return fc, info


def run_fold(job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Corpus, out_root: Path, *, steps: Sequence[str],
             runners: Mapping[str, Any], code: str, state: D.PlanState, guard_fn: Callable = outer_guard,
             inner_check: Callable = inner_isolation_check, with_support: bool = True,
             prereg: Mapping[str, Any] | None = None) -> dict:
    """Run (or skip) one fold of one job (module docstring)."""
    runner = runner_for(job, runners)
    design_hash = corpus.design_hash(job.stem)
    # addendum 2 item 5 + the freezing-candidate pass: the digest a record of THIS job's stage is verified / written
    # with is the stage's registry entry (the live digest until the registry exists); ``live`` is the live digest
    jstage, live = REG.job_stage(job), code
    code = REG.code_digest_for(jstage, live=live)
    digest = fold_digest(job, fold, code, state, runner, out_root, ordinal=ordinal, design_hash=design_hash)
    status = D.resume_status(job, fold, out_root, digest, steps)
    if status["complete"]:
        return {"job": job.key, "fold_id": fold.fold_id, "status": "skipped_done"}
    REG.refuse_unless_writable(jstage, live)     # a record of a registered stage is written only under its code: a
    #                                             discovery job found incomplete after registration is an error, not a refit
    t_all = time.perf_counter()
    fc, info = prepare_fold(job, fold, ordinal, corpus, out_root, state, guard_fn, inner_check, code=code)
    if with_support and job.kind == "fit":
        info["support"] = ensure_support(fc, code)
    base = {"schema": D.SCHEMA, "job": job.record(), "fold_id": fold.fold_id, "fold_hash": fold.fold_hash, "stem": job.stem,
            "design_hash": design_hash, "digest": digest, "code_digest": code, "fold_ordinal": ordinal,
            "model_fold_number": ordinal, "model_seed": model_seed_of(ordinal) if job.kind == "fit" else None,
            "batching_label": fc.batching_label, "prereg_sha256": D.REGISTERED_PREREG_SHA256,
            "prereg_addenda_sha256": (prereg or {}).get("addenda_sha256"),
            "prereg_n_addenda": (prereg or {}).get("n_addenda"), "addendum_implemented": REG.addenda_count(jstage),
            "registry_stage": jstage,
            "prereg_gate": dict(prereg) if prereg else None, **info}
    need_point = any(PREDICTION_STEP not in (status["steps_done"].get(a) or []) for a in job.writes) or status["stale"]
    records: dict[str, dict] = {}
    if need_point:
        outputs = runner.point(fc)
        if set(outputs) != set(job.writes):
            raise AssertionError(f"{job.key}: the runner wrote {sorted(outputs)}, the job declares {list(job.writes)}")
        rss = D.peak_rss_bytes()
        for arm, o in outputs.items():
            if o.intervals is None:
                istat = "pending"
            else:
                istat = "split_conformal_inner" if o.intervals.get("status", "calibrated") == "calibrated" \
                    else str(o.intervals["status"])
            frame = D.prediction_frame(o.pred, job=job, arm=arm, fold=fold, ordinal=ordinal, row_ids=fc.sc_ids,
                                       selected_config=o.selected_config, model_seed=o.model_seed,
                                       intervals_status=istat, batching_label=fc.batching_label, fit_seconds=o.seconds)
            D.assert_v6_clean(corpus.labels_of(frame["row_id"]), corpus.v6, f"{job.key}/{fold.fold_id}/{arm} written")
            D.assert_selection_rows(frame["row_id"], corpus.half_by_id[job.design], f"{job.key}/{fold.fold_id}/{arm} written")
            steps_rec = {PREDICTION_STEP: {"seconds": round(o.seconds, 3), "peak_rss_bytes": rss,
                                           "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                                           **({"shared_with": o.shared_with} if o.shared_with else {})}}
            if o.intervals is not None:
                steps_rec[INTERVAL_STEP] = {"seconds": 0.0, "in_point_step": True, **o.intervals}
            rec = {**base, "arm": arm, "selected_config": o.selected_config, "model_seed": o.model_seed,
                   "arm_record": o.record, "n_feature_columns": len(o.feature_columns), "steps": steps_rec}
            pq, js = D.fold_paths(out_root, job, arm, fold.fold_id)
            _atomic_parquet(frame, pq)
            _atomic_json(rec, js)
            records[arm] = rec
    else:
        for arm in job.writes:
            records[arm] = D.read_record(D.fold_paths(out_root, job, arm, fold.fold_id)[1])
    if INTERVAL_STEP in steps and getattr(runner, "has_interval_step", False):
        for arm in job.writes:
            rec = records[arm]
            if INTERVAL_STEP in rec.get("steps", {}):
                continue
            t0 = time.perf_counter()
            cal, plan = runner.calibration(fc, rec["arm_record"])
            pq, js = D.fold_paths(out_root, job, arm, fold.fold_id)
            frame = pd.read_parquet(pq)
            if cal is None:
                frame = frame.assign(intervals_status=plan["status"])
                rec["steps"][INTERVAL_STEP] = {"seconds": round(time.perf_counter() - t0, 3), "status": plan["status"],
                                               "plan": plan, "n_calibration": 0,
                                               "method": "not calibrated (discovery.calibration_folds)"}
            else:
                cal.fit_table(corpus.table, fc.mask, fc.ctx)
                frame = D.attach_intervals(frame, cal.quantiles, len(cal.residuals))
                rec["steps"][INTERVAL_STEP] = {"seconds": round(time.perf_counter() - t0, 3),
                                               "peak_rss_bytes": D.peak_rss_bytes(), "status": "calibrated", "plan": plan,
                                               "method": "discovery.CrossFitResidualConformal (cross-fitted inner splits)",
                                               **cal.record()}
            D.assert_v6_clean(corpus.labels_of(frame["row_id"]), corpus.v6, f"{job.key}/{fold.fold_id}/{arm} intervals")
            _atomic_parquet(frame, pq)
            _atomic_json(rec, js)
    return {"job": job.key, "fold_id": fold.fold_id, "status": "fitted", "seconds": round(time.perf_counter() - t_all, 3)}


def expected_fold_records(job: D.JobSpec, folds: Sequence[FI.Fold], *, code: str, state: D.PlanState, out_root: Path,
                          excluded_ids: Iterable[str], runners: Mapping[str, Any] | None = None) -> dict[str, dict]:
    """``{fold_id: {digest, fold_hash}}`` of every fittable fold of ``job`` as the current code, fold file and plan state
    would record them (``discovery.read_discovery_record_set``'s ``expected``)."""
    runner = runner_for(job, runners or default_runners())
    dh = FI.design_hash(list(folds))
    return {f.fold_id: {"digest": fold_digest(job, f, code, state, runner, out_root, ordinal=k, design_hash=dh),
                        "fold_hash": f.fold_hash}
            for f, k in D.fittable_folds(job, folds, excluded_ids)}


def verified_predictions(out_root: Path, arm: str, design_dir: str, seed: int, *, code: str | None = None,
                         state: D.PlanState,
                         excluded_ids: Iterable[str], folds_dir: Path | None = None,
                         runners: Mapping[str, Any] | None = None, steps: Sequence[str] = (PREDICTION_STEP,),
                         fold_cache: dict | None = None) -> tuple[pd.DataFrame | None, dict]:
    """The predictions of one (arm, design directory, seed) read only when every record is exactly what the code the
    record set's REGISTRY STAGE was written under (``registry.code_digest_for(registry.job_stage(job))``: the
    ``discovery`` entry, or ``discovery_candidates`` for a freezing-candidate job; addendum 2 item 5: never the live code
    once the registry exists), fold file and plan state would write (``discovery.READINGS['record_verification']``): the
    job is taken from the records (one job per directory, its arm, design directory and seed matching), its expected
    digests are recomputed, and ``discovery.read_discovery_record_set`` refuses a stale or foreign record.
    ``(None, status)`` when nothing or not every fittable fold is recorded.  ``code=None`` reads the registry entry of
    the records' stage."""
    d = D.discovery_root(out_root) / arm / design_dir / f"s{seed}"
    jsons = sorted(d.glob("*.json")) if d.exists() else []
    if not jsons:
        return None, {"arm": arm, "design_dir": design_dir, "seed": int(seed), "status": "missing"}
    bodies = [D.read_record(j) for j in jsons]
    if any(b is None or "job" not in b for b in bodies):
        raise D.StaleRecordError(f"{d}: an unreadable record")
    keys = {b["job"].get("key") for b in bodies}
    if len(keys) != 1:
        raise D.StaleRecordError(f"{d}: records of {len(keys)} different jobs")
    job = D.job_from_record(bodies[0])
    code = REG.code_digest_for(REG.job_stage(job)) if code is None else code    # the entry of THIS record set's stage
    if arm not in job.writes or job.design_dir != design_dir or int(job.seed) != int(seed):
        raise D.StaleRecordError(f"{d}: the records' job {job.key} does not write {arm}/{design_dir}/s{seed}")
    cache = fold_cache if fold_cache is not None else {}
    if job.stem not in cache:
        cache[job.stem] = FI.read_design(job.stem, folds_dir or paths.FOLDS_DIR)
    expected = expected_fold_records(job, cache[job.stem], code=code, state=state, out_root=out_root,
                                     excluded_ids=excluded_ids, runners=runners)
    return D.read_discovery_record_set(out_root, arm, design_dir, seed, expected=expected, steps=steps)


def current_code_digest() -> str:
    """The LIVE combined code digest this runner writes into a record it fits now (``CODE_FILES`` + this script +
    ``RUNNER_OBJECTS``).  Readers verify records with :func:`verification_code_digest` instead (addendum 2 item 5)."""
    return D.code_digest(CODE_FILES + (Path(__file__).resolve(),), RUNNER_OBJECTS)["combined"]


def verification_code_digest(stage: str = STAGE) -> str:
    """The code digest records of ``stage`` are verified against: its registry entry when ``manifests/digest_registry.json``
    holds it (the discovery entry read from the records at registration; the candidates entry registered from the tree
    before the pass), else the live digest."""
    return REG.code_digest_for(stage)


# ============================================================================================= #
# the nested-certificate safeguard (section 2 resolution)
# ============================================================================================= #

def safeguard_splitter(stem: str, corpus: Corpus):
    """The inner design the safeguard probes: for the V5 family the addendum's simultaneous design -- the one every
    learned arm and B6 now tune and calibrate on, whose certificate covers a whole inner fold at once -- and the
    registered V1 / V2 designs otherwise.

    ``V6`` is in the V5 family here for the same reason as everywhere else (POST-HOC addendum 7 item 2: section 3.4
    defines V6 as the V5 state-level rule of section 3.1), so a V6 stem no longer falls through to the V2 metal-holdout
    design.  No safeguard job is enumerated at confirmation, so this branch is unreachable in the single run and no
    safeguard record exists for any V6 stem -- the fix removes a latent trap rather than changing a number, and it is
    recorded as such."""
    design, variant, scheme = stem.split("__")
    if design in ("V5", "V5P", "V5PAIR", "V6"):
        return simultaneous_inner_design(
            D.JobSpec(kind="fit", arm=D.SAFEGUARD_PROBE_ARM, design=design, variant=variant, scheme=scheme,
                      seed=D.PRIMARY_SEED), corpus)
    if design == "V1":
        return I.GroupKFoldCalibration(3)
    return I.InnerMetalCalibration(3, 100, 5)


def run_safeguard(job: D.JobSpec, corpus: Corpus, out_root: Path) -> dict:
    """``every_split`` vs ``nested_certificate`` on the registered sample of one fold file (closed-form probe arm)."""
    from gen19ct.models import baselines as B

    stem = job.stem
    out = D.discovery_root(out_root) / "safeguard" / f"{stem}.json"
    sample = corpus.index_json["nested_certificate_safeguard_sample"]["designs"][stem]
    folds = {f.fold_id: f for f in corpus.folds(stem)}
    if FI.design_hash(list(folds.values())) != sample["design_hash"]:
        raise AssertionError(f"{stem}: the fold file differs from the sampled design")
    prev = D.read_record(out)
    if prev is not None and prev.get("design_hash") == sample["design_hash"] and prev.get("complete"):
        return {"job": job.key, "status": "skipped_done"}
    design, variant, _ = stem.split("__")
    pseudo = D.JobSpec(kind="fit", arm="B3x", design=design, variant=variant, scheme=stem.split("__")[2],
                       seed=D.PRIMARY_SEED)
    recs = []
    for fid in sample["fold_ids"]:
        f = folds[fid]
        if f.fold_hash != sample["fold_hashes"][fid]:
            raise AssertionError(f"{stem}/{fid}: fold hash differs from the drawn sample")
        hidden = corpus.labels_of(f.hidden_row_ids)
        mask = np.ones(corpus.table.n, dtype=bool)
        mask[corpus.table.positions(hidden)] = False
        res = {}
        for mode in ("nested_certificate", "every_split"):
            calls = {"n": 0}
            base = inner_isolation_check(pseudo, corpus)

            def check(tr, te, _b=base, _c=calls):
                _c["n"] += 1
                return _b(tr, te)
            ctx = I.FitContext(systems=corpus.systems, components=corpus.comps, table=corpus.table, hidden_index=hidden,
                               v6_mask=corpus.v6, exclude_from_scoring=corpus.coext, isolation_check=check, guard_cache={},
                               seed=D.PRIMARY_SEED)
            t0 = time.perf_counter()
            try:
                w = I.ConformalWrapper(B.BaselineArm(D.SAFEGUARD_PROBE_ARM), splitter=safeguard_splitter(stem, corpus),
                                       guard=mode).fit_table(corpus.table, mask, ctx)
                res[mode] = {"ok": True, "residuals": w.residuals, "quantiles": w.quantiles, "checks": calls["n"],
                             "seconds": time.perf_counter() - t0}
            except AssertionError as exc:
                res[mode] = {"ok": False, "error": str(exc)[:500], "checks": calls["n"],
                             "seconds": time.perf_counter() - t0}
        a, b = res["nested_certificate"], res["every_split"]
        same = (a["ok"] == b["ok"]) and (not a["ok"] or (np.array_equal(a["residuals"], b["residuals"])
                                                        and a["quantiles"] == b["quantiles"]))
        recs.append({"fold_id": fid, "fold_hash": f.fold_hash, "identical": bool(same),
                     **{f"{m}_{k}": v for m, r in res.items() for k, v in r.items() if k in ("ok", "checks", "seconds", "error")}})
    spl = safeguard_splitter(stem, corpus)
    body = {"schema": D.SCHEMA, "stem": stem, "design_hash": sample["design_hash"], "probe_arm": D.SAFEGUARD_PROBE_ARM,
            "inner_design": getattr(spl, "name", type(spl).__name__),
            "reading": D.READINGS["safeguard_probe"], "folds": recs, "complete": True,
            "every_split_mandatory": not all(r["identical"] for r in recs),
            "vacuous_under_current_code": bool(sample.get("vacuous_under_current_code"))}
    _atomic_json(body, paths.ensure_dir(out.parent) / out.name)
    return {"job": job.key, "status": "done", "every_split_mandatory": body["every_split_mandatory"]}


def apply_safeguard_records(state: D.PlanState, out_root: Path) -> D.PlanState:
    for p in sorted((D.discovery_root(out_root) / "safeguard").glob("*.json")):
        body = D.read_record(p)
        if body and body.get("complete"):
            state.guard_mode[body["stem"]] = "every_split" if body["every_split_mandatory"] else "nested_certificate"
    return state


# ============================================================================================= #
# the B6 checks (sections 3.1, 3.2; section 7 item 6)
# ============================================================================================= #

def unit_mae_table(pred: pd.DataFrame, corpus: Corpus, design: str, v1_scheme: str) -> pd.Series:
    """Section 4 per-unit MAE of stored predictions (targets and unit columns from the corpus)."""
    ids = pred["row_id"].astype(str)
    lab = corpus.labels_of(ids)
    fr = pd.DataFrame({"row_id": ids.to_numpy(), "fold_id": pred["fold_id"].to_numpy(),
                       EM.PRED_COL: pred["mean_logD"].to_numpy(dtype=float),
                       EM.Y_COL: corpus.frame.loc[lab, "log_D"].to_numpy(dtype=float),
                       EM.METAL_STATE_COL: corpus.frame.loc[lab, SG.METAL_COL].to_numpy(dtype=object),
                       EM.SYSTEM_COL: corpus.frame.loc[lab, SG.SYSTEM_COL].to_numpy(dtype=object),
                       EM.PUB_GROUP_COL: corpus.frame.loc[lab, FI.GROUP_COL].astype(str).to_numpy()}).set_index("row_id",
                                                                                                           drop=False)
    fr.index.name = None
    v6 = pd.Series(corpus.v6.reindex(lab).to_numpy(dtype=bool), index=fr.index)
    return D.unit_mae(fr, design, v6_mask=v6, v1_scheme=v1_scheme, remainder_groups=corpus.remainder_groups)


def evaluate_b6_checks(out_root: Path, corpus: Corpus, state: D.PlanState, code: str,
                       runners: Mapping[str, Any] | None = None) -> tuple[D.PlanState, dict]:
    """The B6 batched-vs-exact (V5, per seed) and ten-fold (V1, per seed) checks from the stored B6 predictions; a seed
    counts only when every fittable fold of both designs is recorded, and only verified records are read
    (``verified_predictions``: a stale or foreign record raises).  ``code`` is the invocation's live digest, kept for the
    caller's signature; each record set is verified against the registry entry of the stage its records carry
    (``code=None``; addendum 2 item 5), never against the live code."""
    statuses: list[dict] = []

    def per_seed(design: str, variant: str, scheme: str, v1_scheme: str) -> dict[int, pd.Series]:
        out = {}
        stem = FI.design_stem(design, variant, scheme)
        if not (corpus.folds_dir / f"{stem}.json").exists():
            return out
        multi = any(f.seed is not None for f in corpus.folds(stem))
        for s in D.PLAN_SEEDS:                     # addendum 1 item 3: discovery runs seed 104729 only
            job = D.JobSpec(kind="fit", arm="B6", design=design, variant=variant, scheme=scheme, seed=s,
                            fold_seed=s if multi else None, writes=D.B6_ARMS)
            # the B6 record sets are verified against the entry of their own stage, not the live ``code`` of this
            # invocation (run_plan re-evaluates the checks after stage 01 on every invocation, the candidates pass included)
            p, st = verified_predictions(out_root, "B6", job.design_dir, s, code=None, state=state,
                                         excluded_ids=corpus.coext_ids, folds_dir=corpus.folds_dir, runners=runners,
                                         fold_cache={stem: corpus.folds(stem)})
            statuses.append(st)
            if p is None:
                continue
            out[s] = unit_mae_table(p, corpus, design, v1_scheme)
        return out
    exact5 = per_seed("V5", "primary", "exact", "exact")
    v5 = D.b6_check(exact5, per_seed("V5", "primary", "batched", "exact"))
    v5b = None
    if state.v5_batched_check in ("recolour", "passed_after_recolour", "failed"):
        v5b = D.b6_check(exact5, per_seed("V5", "primary", "batched_max4", "exact"))
    v1 = D.b6_check(per_seed("V1", "copy", "exact", "exact"), per_seed("V1", "copy", "grouped10", "grouped"))
    state.v5_batched_check = D.next_v5_check_state(state.v5_batched_check, v5, v5b)
    state.v1_tenfold_check = "pending" if v1["passed"] is None else ("passed" if v1["passed"] else "failed")
    body = {"v5_batched_vs_exact": v5, "v5_recoloured": v5b, "v1_tenfold_vs_exact": v1,
            "v5_state": state.v5_batched_check, "v1_state": state.v1_tenfold_check, "record_sets": statuses}
    return state, body


# ============================================================================================= #
# jobs
# ============================================================================================= #

def plan_state_path(out_root: Path) -> Path:
    return D.discovery_root(out_root) / "decisions" / "plan_state.json"


def runner_stage(out_root: Path) -> str:
    """The registry stage this invocation is gated on (``registry.READINGS['candidates']``): ``REG.CANDIDATES`` once the
    plan state holds freezing candidates (written by the scorer, hence after addendum 2 is appended; the jobs of
    ``D.STAGES['candidates']`` are then the only fit jobs without a record set), else ``REG.DISCOVERY``.  The seal gate
    thus pins the two-addenda text for the candidates pass and the addendum-1 text for everything before it; a job's
    OWN stage (``REG.job_stage``) decides the digests its record carries."""
    st = D.PlanState.read(plan_state_path(out_root))
    return REG.CANDIDATES if st.freezing_candidates else REG.DISCOVERY


def fold_tasks(job: D.JobSpec, corpus: Corpus) -> list[tuple[FI.Fold, int]]:
    if job.stem.endswith("max4") and not (corpus.folds_dir / f"{job.stem}.json").exists():
        raise FileNotFoundError(f"{job.stem}: the re-coloured fold file is not built; run scripts/g19_build_folds.py with "
                                "the section 7 item 6 cap (max_cells_per_batch=4) first")
    return corpus.fittable(job)


W = SimpleNamespace(corpus=None, runners=None)


def _init_worker(coext_ids: list[str]) -> None:
    import torch
    torch.set_num_threads(2)
    W.corpus = load_corpus(coext_ids)
    W.runners = default_runners()


def _worker_fold(job: D.JobSpec, fold_id: str, ordinal: int, out_root: str, steps: list[str], code: str,
                 state_rec: dict, prereg: dict | None = None) -> dict:
    try:
        corpus = W.corpus
        fold = next(f for f in corpus.folds(job.stem) if f.fold_id == fold_id)
        state = D.PlanState(**{k: state_rec[k] for k in ("v5_batched_check", "v1_tenfold_check", "guard_mode",
                                                            "freezing_candidates", "notes")})
        return run_fold(job, fold, ordinal, corpus, Path(out_root), steps=steps, runners=W.runners, code=code, state=state,
                        prereg=prereg)
    except Exception as exc:                                        # noqa: BLE001 -- reported with the traceback
        return {"job": job.key, "fold_id": fold_id, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]}


def run_jobs(jobs: Sequence[D.JobSpec], corpus: Corpus, out_root: Path, *, steps: Sequence[str], code: str,
             state: D.PlanState, runners: Mapping[str, Any] | None = None, workers: int = 1,
             max_hours: float | None = None, coext_ids: Sequence[str] = (), guard_fn: Callable = outer_guard,
             inner_check: Callable = inner_isolation_check, progress: Callable[[dict], None] | None = None,
             stop_on_error: bool = True, with_support: bool = True,
             budget_used_s: Callable[[], float] | None = None, budget_hours: float = D.BUDGET_HOURS,
             prereg: Mapping[str, Any] | None = None) -> dict:
    """Run jobs in plan order; fold-level parallelism inside a job (``workers`` 1 in-process, 2 processes, at most
    ``workers`` folds in flight so every dispatch sees the ledger).  Returns the per-job ledger.

    Section 7 item 5: once ``budget_used_s()`` (default: the wall clock of earlier invocations recorded in
    ``decisions/wall_clock.json`` plus this call) reaches ``budget_hours`` (60), the exhaustion is recorded and only
    :func:`discovery.demotable` jobs (M3-M7) are demoted; every other job keeps running.  ``max_hours`` is an operator
    pause: no new fold is dispatched once the ledger reaches it (a later invocation resumes)."""
    runners = runners or default_runners()
    ledger = {"jobs": {}, "stopped": None, "markers": [], "budget": None, "demoted": []}
    t_start = time.perf_counter()
    prev = wall_clock_total(out_root)
    if budget_used_s is None:
        def budget_used_s() -> float:
            return prev + time.perf_counter() - t_start
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(list(coext_ids),)) \
        if workers > 1 else None
    prereg_rec = None if prereg is None else dict(prereg)

    def paused() -> bool:
        return max_hours is not None and budget_used_s() / 3600.0 >= max_hours

    try:
        for job in jobs:
            if job.kind == "marker":
                log(f"[marker] {job.message or job.arm}")
                ledger["markers"].append(job.message or job.arm)
                if job.arm == "M3+":
                    print(D.NOT_IMPLEMENTED, flush=True)
                    ledger["stopped"] = D.NOT_IMPLEMENTED
                    break
                continue
            if job.kind == "h3_spec":
                continue
            used_s = budget_used_s()
            if used_s / 3600.0 >= budget_hours and ledger["budget"] is None:
                ledger["budget"] = {**D.budget_status(used_s, budget_hours=budget_hours), "exhausted_before": job.key}
                log(f"section 7 item 5: the {budget_hours:g} h budget is exhausted ({used_s / 3600:.2f} h) before "
                    f"{job.key}; only M3-M7 are demoted, every other job continues")
            if ledger["budget"] is not None and D.demotable(job):
                ledger["demoted"].append(job.key)
                continue
            if paused():
                ledger["stopped"] = (f"--max-hours {max_hours} reached ({used_s / 3600:.2f} h): operator pause, resumable; "
                                     "not a registered demotion")
                log(ledger["stopped"])
                break
            t0 = time.perf_counter()
            if job.kind == "safeguard":
                r = run_safeguard(job, corpus, out_root)
                ledger["jobs"][job.key] = {"seconds": round(time.perf_counter() - t0, 1), **r}
                apply_safeguard_records(state, out_root)
                continue
            tasks = fold_tasks(job, corpus)
            results = []
            if pool is None:
                for f, k in tasks:
                    if paused():
                        ledger["stopped"] = f"--max-hours {max_hours} reached inside {job.key} (operator pause, resumable)"
                        break
                    try:
                        results.append(run_fold(job, f, k, corpus, out_root, steps=steps, runners=runners, code=code,
                                                state=state, guard_fn=guard_fn, inner_check=inner_check,
                                                with_support=with_support, prereg=prereg_rec))
                    except Exception as exc:                        # noqa: BLE001
                        results.append({"job": job.key, "fold_id": f.fold_id, "status": "error",
                                        "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
                        if stop_on_error:
                            break
            else:
                pending, inflight, halt = iter(tasks), set(), False
                while True:
                    while not halt and len(inflight) < workers:
                        if paused():
                            ledger["stopped"] = (f"--max-hours {max_hours} reached inside {job.key} (operator pause, "
                                                 "resumable)")
                            halt = True
                            break
                        nxt = next(pending, None)
                        if nxt is None:
                            break
                        f, k = nxt
                        inflight.add(pool.submit(_worker_fold, job, f.fold_id, k, str(out_root), list(steps), code,
                                                 state.record(), prereg_rec))
                    if not inflight:
                        break
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for fu in done:
                        r = fu.result()
                        results.append(r)
                        if r["status"] == "error" and stop_on_error:
                            halt = True
            errs = [r for r in results if r["status"] == "error"]
            ledger["jobs"][job.key] = {"n_folds": len(tasks), "fitted": sum(r["status"] == "fitted" for r in results),
                                       "skipped": sum(r["status"] == "skipped_done" for r in results),
                                       "errors": errs[:3], "seconds": round(time.perf_counter() - t0, 1),
                                       "stage": job.stage}
            log(f"{job.key}: {ledger['jobs'][job.key]['fitted']} fitted, {ledger['jobs'][job.key]['skipped']} skipped, "
                f"{len(errs)} errors, {ledger['jobs'][job.key]['seconds']} s")
            if progress:
                progress(ledger)
            if errs and stop_on_error:
                ledger["stopped"] = f"error in {job.key}: {errs[0]['error']}"
                break
            if ledger["stopped"]:
                break
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return ledger


# ============================================================================================= #
# fold counts, cost estimate, benchmark
# ============================================================================================= #

def fold_counts(jobs: Sequence[D.JobSpec], folds_dir: Path) -> dict[str, int]:
    cache: dict[str, list[FI.Fold]] = {}
    out = {}
    for j in jobs:
        if j.kind not in ("fit", "comparator_intervals", "safeguard"):
            continue
        if not (folds_dir / f"{j.stem}.json").exists():
            out[j.key] = 0
            continue
        if j.stem not in cache:
            cache[j.stem] = FI.read_design(j.stem, folds_dir, verify=False)
        if j.kind == "safeguard":
            idx = json.loads((folds_dir / "INDEX.json").read_text(encoding="utf-8"))
            out[j.key] = int(idx["nested_certificate_safeguard_sample"]["designs"][j.stem]["n_drawn"])
        else:
            out[j.key] = len(D.job_folds(j, cache[j.stem]))
    return out


#: pre-seal per-(fold, arm) seconds with conformal inner calibration (manifests/run_info/g19_run_preseal_jobs.json:
#: V5__primary 2468.1 s / (210 folds x 11 arms); V1__copy 724.6 / (83 x 11); V2__element 63.0 / (23 x 11))
PRESEAL_CLOSED_FORM_S = {"V5": 2468.1 / (210 * 11), "V1": 724.6 / (83 * 11), "V2": 63.0 / (23 * 11)}
#: safeguard seconds per drawn fold: two conformal calibrations of the probe arm over the inner splits of the design it
#: probes.  Under addendum 1 item 1 a V5 fold has 3 simultaneous inner splits, not ~90 single-cell ones, so the probe
#: costs 3 fold_isolation_check calls (about 0.35 s each) and 2 x 3 closed-form calibration fits
SAFEGUARD_S_PER_FOLD = {"V5": D.INNER_N_FOLDS * 0.35 + 2 * PRESEAL_CLOSED_FORM_S["V5"] * D.INNER_N_FOLDS,
                        "V1": 3 * 1.0 + 2, "V2": 3 * 1.0 + 2}


#: per fitted fold: the outer fold_isolation_check (about 1 s on the MODEL rows), the section 13 support file (<= about 1.2
#: s measured on V5 exact, V5 batched and V1 folds) and parquet / JSON I/O
FOLD_OVERHEAD_S = 3.0


def other_costs(jobs: Sequence[D.JobSpec], counts: Mapping[str, int]) -> dict[str, float]:
    out = {}
    for j in jobs:
        if j.kind == "comparator_intervals":
            out[j.key] = counts.get(j.key, 0) * PRESEAL_CLOSED_FORM_S[j.design]
        elif j.kind == "safeguard":
            d = j.stem.split("__")[0]
            out[j.key] = counts.get(j.key, 0) * SAFEGUARD_S_PER_FOLD["V5" if d.startswith("V5") else d]
    return out


BENCH_FOLDS = {"V5_batched": ("V5__primary__batched", "s104729_S_b000"),
               "V5_exact": ("V5__primary__exact", None),
               "V1": ("V1__copy__grouped10", "s104729_f0"),
               "V2": ("V2__element__exact", None)}
BENCH_DESIGNS = {"B6": ("V5_exact", "V5_batched", "V1", "V2"), "B5": ("V5_batched", "V1", "V2"),
                 "FLAT_CAT": ("V5_batched", "V1", "V2"), "B8": ("V5_batched", "V1", "V2"),
                 "M1": ("V5_batched", "V1", "V2"), "M2": ("V5_batched", "V1", "V2")}
#: the arms both benchmarks measure, in plan order
BENCH_ARMS: tuple[str, ...] = ("B6", "B5", "FLAT_CAT", "B8", "M1", "M2")
#: addendum 1: the design class re-measured per arm (one V5-primary batched fold, seed 104729).  The V1 and V2 inner
#: designs are unchanged by the addendum, so their unit costs are reused from the sealed-plan measurements, and B6's
#: exact V5 folds take the batched measurement (the same inner design, a similar training-row count)
BENCH_DESIGN_ADDENDUM1 = "V5_batched"


class _OneSplit:
    def __init__(self, splits):
        self._s = splits
        self.name = "benchmark_one_split"

    def splits(self, table, mask, context):
        return list(self._s)


def sealed_plan_splitter(fc: FoldContext):
    """The V5 inner design of the SEALED section 7 plan (exact leave-one-inner-cell-out on exact folds, the batched
    inner design on batched folds), superseded by addendum 1 item 1 and kept only so that ``--benchmark`` can still
    reproduce the measurement the sealed plan was priced with (``cost_estimate.md``)."""
    job, c = fc.job, fc.corpus
    if exact_scheme(job):
        v = CH.VARIANTS[inner_variant(job)] if job.design == "V5" else CH.VARIANTS["primary"]
        return D.V5ExactInnerCells(k=v.thresholds.k, p=v.thresholds.p, m=v.thresholds.m, n_folds=3,
                                   max_cells_per_fold=30, component_aware=v.component_aware,
                                   component_map=c.pmap if v.parent_structure else None, medium=v.medium, frame=c.frame)
    return D.TuningSplitCalibration(boosted_design(job), c.frame, "full")


def benchmark_arm(arm: str, corpus: Corpus) -> list[dict]:
    """Fit-only timings of one arm under the SEALED section 7 inner design (module docstring): never a prediction of a
    test row.  Superseded by :func:`benchmark_addendum1_arm`; kept so that the 1,535.7 CPU-hour estimate the addendum
    rests on stays reproducible."""
    from gen19ct.models import boosted as BO
    from gen19ct.models import factorized as FZ
    from gen19ct.models import neural as NN
    from gen19ct.models import previous_gen as PG

    out = []
    for dclass in BENCH_DESIGNS[arm]:
        stem, fid = BENCH_FOLDS[dclass]
        folds = corpus.folds(stem)
        design, variant, scheme = stem.split("__")
        seed = D.PRIMARY_SEED
        multi = any(x.seed is not None for x in folds)
        job = D.JobSpec(kind="fit", arm=arm, design=design, variant=variant, scheme=scheme, seed=seed,
                        fold_seed=seed if multi else None, writes=D.B6_ARMS if arm == "B6" else (arm,))
        cand = corpus.fittable(job)
        f, k = next((x for x in cand if x[0].fold_id == fid), cand[0]) if fid else cand[0]
        state = D.PlanState(guard_mode={})
        fc, info = prepare_fold(job, f, k, corpus, Path(tempfile.gettempdir()), state)
        rows = corpus.frame.loc[corpus.table.index[fc.mask]]
        if rows.index.isin(fc.sc_labels).any() or rows.index.isin(fc.hidden).any():
            raise AssertionError("benchmark: a hidden row among the training rows")
        rec = {"arm": arm, "design_class": dclass, "stem": stem, "fold_id": f.fold_id, "n_train": int(fc.mask.sum())}
        t0 = time.perf_counter()
        if arm == "B6":
            FZ.register_table(corpus.table, corpus.frame)
            spl = sealed_plan_splitter(fc) if v5_family(job) else B6Runner().splitter(fc)
            full = spl.splits(corpus.table, fc.mask, fc.ctx)
            first = D.FirstInnerFold(_OneSplit(full)).splits(corpus.table, fc.mask, fc.ctx)
            rec.update(splits_full=len(full), splits_first=len(first), design_s=time.perf_counter() - t0)
            t0 = time.perf_counter()
            FZ.run_inner_tuning(corpus.table, fc.mask, fc.ctx, _OneSplit(full[:1]), guard="every_split")
            rec["inner_fit_s"] = time.perf_counter() - t0                  # all 12 configurations of one inner split
            rec["cal_fit_s"] = 0.0
            t0 = time.perf_counter()
            FZ.B6Factorized(2, 1.0).fit_table(corpus.table, fc.mask, fc.ctx)
            rec["outer_refit_s"] = time.perf_counter() - t0
        else:
            dsg = boosted_design(job)
            tsp = D.TuningSplitCalibration(dsg, corpus.frame, "full").tuning_splits(corpus.table, fc.mask, fc.ctx)
            first = [s for s in tsp if s.inner_fold == min(x.inner_fold for x in tsp)]
            rec.update(splits_full=len(tsp), splits_first=len(first), design_s=time.perf_counter() - t0)
            sp = tsp[0]
            BO.verify_split(sp, rows, fc.ctx)
            tr = rows.loc[sp.train_index]
            if arm in ("B5", "FLAT_CAT"):
                cfg = BO.CatBoostConfig(6, 3.0)
                tuner = BO.Tuner(arm, _FixedSplits([sp]), grid=(cfg,), mode="full", cv=corpus.cv)
                t0 = time.perf_counter()
                res = tuner.run(rows, fc.ctx, BO.model_seed_for_fold(k))
                rec["inner_fit_s"] = time.perf_counter() - t0
                iters = int(res.selected_iterations)
                rec["inner_trees"] = iters
                t0 = time.perf_counter()
                BO.BoostedArm(arm, fold_index=k, design=dsg, config=cfg, iterations=iters, cv=corpus.cv).fit(tr, fc.ctx)
                rec["cal_fit_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                BO.BoostedArm(arm, fold_index=k, design=dsg, config=cfg, iterations=iters, cv=corpus.cv).fit(rows, fc.ctx)
                rec["outer_refit_s"] = time.perf_counter() - t0
            elif arm == "B8":
                t0 = time.perf_counter()
                PG.B8Level(corpus.frame, cv=corpus.cv, fold=k).fit(tr, fc.ctx)
                rec["inner_fit_s"] = rec["cal_fit_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                PG.B8Level(corpus.frame, cv=corpus.cv, fold=k).fit(rows, fc.ctx)
                rec["outer_refit_s"] = time.perf_counter() - t0
            else:
                cfg = NN.NeuralConfig(8, 1e-3, 0 if arm == "M1" else 4)
                vs = [NN.ValidationSplit(name=sp.split_id, inner_fold=sp.inner_fold, train_index=sp.train_index,
                                         valid_index=sp.val_index, valid_units=sp.val_units, hidden_index=sp.hidden_index,
                                         guard="benchmark")]
                t0 = time.perf_counter()
                res = NN.tune(rows, vs, [cfg], model_seed=NN.registered_model_seed(k), condition_vectors=corpus.cv)
                rec["inner_fit_s"] = time.perf_counter() - t0
                ep = int(res.n_epochs)
                rec["inner_best_epoch"] = ep
                t0 = time.perf_counter()
                NN.FactorisedArm(cfg, n_epochs=ep, model_seed=NN.registered_model_seed(k), condition_vectors=corpus.cv).fit(tr)
                rec["cal_fit_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                NN.FactorisedArm(cfg, n_epochs=ep, model_seed=NN.registered_model_seed(k), condition_vectors=corpus.cv).fit(rows)
                rec["outer_refit_s"] = time.perf_counter() - t0
        rec["peak_rss_bytes"] = D.peak_rss_bytes()
        rec["n_test_rows_predicted"] = 0
        out.append(rec)
        log(f"benchmark {arm} {dclass}: {json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()})}")
    return out


class _FixedSplits:
    """A boosted inner design object returning pre-built tuning splits (benchmark only)."""

    def __init__(self, splits):
        self._s = list(splits)

    def splits(self, rows, context):
        return list(self._s)

    def describe(self):
        return {"design": "benchmark fixed split"}


def unit_costs_from_measurements(meas: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], D.UnitCost]:
    """UnitCost per (arm, design class); B6r0 shares B6's; V5-P / V5-PAIR take the V5 classes; the exact V5 class of a
    heavy arm is not used (heavy arms never run exact V5 folds)."""
    out = {}
    for m in meas:
        uc = D.UnitCost(arm=m["arm"], design_class=m["design_class"], inner_fit_s=float(m["inner_fit_s"]),
                        cal_fit_s=float(m["cal_fit_s"]), outer_refit_s=float(m["outer_refit_s"]),
                        splits_full=float(m["splits_full"]), splits_first=float(m["splits_first"]), source="benchmark")
        out[(m["arm"], m["design_class"])] = uc
    return out


def cost_markdown(summary: Mapping[str, Any], table: pd.DataFrame, meas: Sequence[Mapping[str, Any]],
                  conditional: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> str:
    extra = dict(extra or {})
    lines = ["# Discovery cost estimate", "",
             "Generated by `scripts/g19_run_discovery.py --benchmark` from fit-only timings on the TRAINING rows of one "
             "V5-primary batched fold, one V1 ten-fold fold, one V2 fold and one V5-primary exact fold (B6), seed 104729. "
             "No test row was predicted; no score was computed. The estimate multiplies measured per-fit seconds by the "
             "section 7 compute plan: folds x (inner splits x configurations x inner fit + inner splits x calibration "
             "refit + outer refit + inner guard), with 3 inner folds on seed 104729 and the first inner fold elsewhere "
             "(calibration: the cross-fitted refits on seed 104729, the next inner fold elsewhere -- the same number of "
             "refits; B6 re-runs its selected configurations on the next inner fold on those seeds). "
             "Wall clock = compute / workers (ideal; the machine was shared with other agents during the benchmark). "
             "The inner-fit seconds include the per-split feature / encoder fit, so preprocessing is counted once per "
             "configuration instead of once per split (a small overcount: 1.2 s of 63.7 s for B5 on V5).", "",
             "## Measurements (seconds)", "",
             "| arm | design | fold | train rows | inner splits (full / first) | inner config fit | calibration refit | outer refit | peak RSS (MB) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in meas:
        rss = m.get("peak_rss_bytes")
        lines.append(f"| {m['arm']} | {m['design_class']} | {m['fold_id']} | {m['n_train']} | {m['splits_full']} / "
                     f"{m['splits_first']} | {m['inner_fit_s']:.1f} | {m['cal_fit_s']:.1f} | {m['outer_refit_s']:.1f} | "
                     f"{'' if rss is None else f'{rss / 1e6:.0f}'} |")
    lines += ["", "B6 'inner config fit' is all 12 (rank, lambda) configurations of one inner split (solved jointly).", "",
              "## Estimated compute by arm (hours, 1 worker)", "", "| arm | compute h |", "|---|---|"]
    for a, h in sorted(summary["compute_h_by_arm"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {a} | {h:.1f} |")
    lines += ["", "## Estimated compute by stage (hours, 1 worker)", "", "| stage | compute h |", "|---|---|"]
    for s, h in sorted(summary["compute_h_by_stage"].items()):
        lines.append(f"| {s} | {h:.1f} |")
    lines += ["", f"**Total: {summary['total_compute_h']:.1f} h compute; {summary['total_wall_h']:.1f} h wall clock on "
                  f"{summary['workers']} workers against the {summary['budget_hours']:.0f} h budget -> "
                  f"{'fits' if summary['fits_budget'] else 'does NOT fit'}.**",
              f"First job beyond the budget (plan order): `{summary['first_job_beyond_budget']}`. Section 7 item 5 "
              "demotes only M3-M7 on exhaustion, so every job in this plan keeps running past it.", ""]
    for k, v in extra.items():
        lines.append(f"- {k}: {json.dumps(v, sort_keys=True)}")
    lines += ["", "Conditional jobs not in the total: " + json.dumps(conditional, sort_keys=True), ""]
    return "\n".join(lines) + "\n"


def _milestones(table: pd.DataFrame) -> dict[str, Any]:
    """Cumulative wall clock at the end of the H1 checkpoints of the plan order."""
    out = {}
    for label, key in (("B6 stage done", None), ("M2 seed 104729 V5-primary done", "fit:M2:V5__primary_batched:s104729"),
                       ("M2 seed 104729 V5-PAIR done", "fit:M2:V5PAIR__primary_batched:s104729")):
        if key is None:
            sub = table[table["stage"] == D.STAGES["b6"]]
            out[label] = None if sub.empty else round(float(sub["cumulative_wall_h"].iloc[-1]), 1)
        else:
            sub = table[table["job"] == key]
            out[label] = None if sub.empty else round(float(sub["cumulative_wall_h"].iloc[0]), 1)
    sub = table[table["stage"].isin([D.STAGES["b6"], D.STAGES["p_b5"], D.STAGES["p_m1"], D.STAGES["p_m2"]])
                & (table["arm"] != "B8")]
    out["seed-104729 main designs of B5, FLAT_CAT, M1, M2 plus every B6 job, wall h"] = round(
        float(sub[sub["design"].isin(["V5", "V1", "V2", "V5P"])]["compute_s"].sum()) / 3600 / 2, 1)
    out["the same plus the M1 / M2 V5-PAIR jobs, wall h"] = round(float(sub["compute_s"].sum()) / 3600 / 2, 1)
    return out


def write_cost_estimate(meas: Sequence[Mapping[str, Any]], out_root: Path, workers: int = 2) -> dict:
    ucs = unit_costs_from_measurements(meas)
    state = D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed")
    jobs = D.enumerate_plan(state)
    counts = fold_counts(jobs, paths.FOLDS_DIR)
    kw = dict(other_costs_s=other_costs(jobs, counts), workers=workers, fold_overhead_s=FOLD_OVERHEAD_S)
    table = D.estimate_cost(jobs, counts, _with_aliases(ucs), **kw)
    summary = D.cost_summary(table, workers=workers)
    alt = D.estimate_cost(jobs, counts, _with_aliases(ucs), first_fold_reading=D.FIRST_FOLD_READINGS[1], **kw)
    alt_summary = D.cost_summary(alt, workers=workers)
    extra = {"H1 checkpoints (cumulative wall h, plan order)": _milestones(table),
             "alternative reading of 'first inner fold (one fit per configuration)': one inner split per configuration "
             "on the first-inner-fold seeds (task X finding V-04; needs a POST-HOC addendum), total wall h":
                 {"total_compute_h": alt_summary["total_compute_h"], "total_wall_h": alt_summary["total_wall_h"],
                  "H1 checkpoints": _milestones(alt)}}
    # conditional work not in the plan total: freezing-candidate V5-P runs of a heavy arm, the B6 re-colouring
    v5p = D.JobSpec(kind="fit", arm="M2", design="V5P", variant="base", scheme="batched", seed=D.PRIMARY_SEED,
                    fold_seed=D.PRIMARY_SEED)
    n_v5p = fold_counts([v5p], paths.FOLDS_DIR)[v5p.key]
    cond = {f"V5-P batched per heavy candidate arm ({n_v5p} folds), hours": {
        a: round(n_v5p * _with_aliases(ucs)[(a, "V5_batched")].fold_seconds("full")["total"] / 3600, 2)
        for a in D.HEAVY_ARMS},
        "B6 re-coloured batches (5 seeds), hours": round(sum(
            counts.get(D.JobSpec(kind="fit", arm="B6", design="V5", variant="primary", scheme="batched", seed=s,
                                 fold_seed=s, writes=D.B6_ARMS).key, 0)
            * ucs[("B6", "V5_batched")].fold_seconds("full" if s == D.PRIMARY_SEED else "first")["total"]
            for s in D.DISCOVERY_SEEDS) / 3600, 2)}
    bdir = paths.ensure_dir(D.discovery_root(out_root) / "benchmark")
    body = {"schema": D.SCHEMA, "measurements": list(meas), "summary": summary, "conditional_not_in_total": cond,
            "milestones": extra, "alternative_first_inner_fold_reading": {"summary": alt_summary,
                                                                         "reading": D.FIRST_FOLD_READINGS[1]},
            "plan_state_assumed": state.record(), "per_job": table.to_dict(orient="records"),
            "model": {"configs_per_split": D.CONFIGS_PER_SPLIT, "calibration_fits_per_split": D.CALIBRATION_FITS_PER_SPLIT,
                      "closed_form_s_per_fold_arm": PRESEAL_CLOSED_FORM_S, "safeguard_s_per_fold": SAFEGUARD_S_PER_FOLD,
                      "fold_overhead_s": FOLD_OVERHEAD_S, "b6_first_mode_calibration": "one joint fit per next-fold split",
                      "wall_clock": "compute / workers"},
            "superseded_by": ("POST-HOC addendum 1: this estimate prices the SEALED section 7 inner design (the "
                              "1,535.7 CPU-hour figure the addendum rests on); the plan that is run is priced in "
                              "cost_estimate_addendum1.json"),
            "readings": {k: D.READINGS[k] for k in ("heavy_arm_intervals", "addendum1_inner_design", "m2_prerequisite",
                                                    "plan_order", "budget", "fold_ordinal")}}
    write_json(bdir / "cost_estimate.json", body)
    write_text(bdir / "cost_estimate.md", cost_markdown(summary, table, meas, cond, extra))
    return body


def _with_aliases(ucs: Mapping[tuple[str, str], D.UnitCost]) -> dict[tuple[str, str], D.UnitCost]:
    out = dict(ucs)
    for (arm, dc), uc in list(ucs.items()):
        if arm == "B6":
            out[("B6r0", dc)] = uc
    return out



# ============================================================================================= #
# POST-HOC addendum 1: the re-priced cost estimate
# ============================================================================================= #

def benchmark_addendum1_arm(arm: str, corpus: Corpus) -> list[dict]:
    """Fit-only timings of one arm under POST-HOC addendum 1, on the TRAINING rows of one V5-primary batched fold
    (seed 104729, :data:`BENCH_FOLDS`): the fold's three simultaneous inner splits (each hiding a whole inner fold of
    up to 30 cells), then ONE inner fit per configuration on one split, one calibration refit on its inner training
    rows, and one outer refit on all outer-training rows.  No test row is predicted and no score is computed."""
    from gen19ct.models import boosted as BO
    from gen19ct.models import factorized as FZ
    from gen19ct.models import neural as NN
    from gen19ct.models import previous_gen as PG

    stem, fid = BENCH_FOLDS[BENCH_DESIGN_ADDENDUM1]
    folds = corpus.folds(stem)
    design, variant, scheme = stem.split("__")
    seed = D.PRIMARY_SEED
    job = D.JobSpec(kind="fit", arm=arm, design=design, variant=variant, scheme=scheme, seed=seed, fold_seed=seed,
                    writes=D.B6_ARMS if arm == "B6" else (arm,))
    cand = corpus.fittable(job)
    f, k = next((x for x in cand if x[0].fold_id == fid), cand[0])
    fc, _ = prepare_fold(job, f, k, corpus, Path(tempfile.gettempdir()), D.PlanState(guard_mode={}))
    rows = corpus.frame.loc[corpus.table.index[fc.mask]]
    if rows.index.isin(fc.sc_labels).any() or rows.index.isin(fc.hidden).any():
        raise AssertionError("benchmark: a hidden row among the training rows")
    t0 = time.perf_counter()
    _, splits = inner_splits_v5(fc)                          # drawn, verified and guarded exactly as in a real fold
    design_s = time.perf_counter() - t0
    rec = {"arm": arm, "design_class": BENCH_DESIGN_ADDENDUM1, "stem": stem, "fold_id": f.fold_id,
           "n_train": int(fc.mask.sum()), "splits_full": len(splits), "splits_first": len(splits),
           "inner_mode": "full", "inner_design": D.INNER_DESIGN_NAME, "design_s": design_s,
           "n_cells_per_split": [int(len(set(map(str, sp.row_units)))) for sp in splits],
           "n_hidden_rows_per_split": [int(len(sp.hidden_positions)) for sp in splits],
           "n_calibration_rows_per_split": [int(len(sp.cal_positions)) for sp in splits]}
    sp = splits[0]
    tr = corpus.frame.loc[corpus.table.index[sp.train_mask]]
    if arm == "B6":
        FZ.register_table(corpus.table, corpus.frame)
        t0 = time.perf_counter()
        FZ.run_inner_tuning(corpus.table, fc.mask, fc.ctx, D.FixedInnerSplits([sp]), guard=fc.guard_mode,
                            require_row_units=True)
        rec["inner_fit_s"] = time.perf_counter() - t0          # all 12 (rank, lambda) configurations of one split
        rec["cal_fit_s"] = 0.0                                 # cross-fitted from the tuning predictions, no refit
        t0 = time.perf_counter()
        FZ.B6Factorized(2, 1.0).fit_table(corpus.table, fc.mask, fc.ctx)
        rec["outer_refit_s"] = time.perf_counter() - t0
    elif arm in ("B5", "FLAT_CAT"):
        cfg = BO.CatBoostConfig(6, 3.0)
        dsg = inner_design_object(job, corpus)                  # boosted.V5SimultaneousTuning: the same design
        tsp = dsg.splits(rows, fc.ctx)
        if len(tsp) != len(splits):
            raise AssertionError(f"benchmark: the tuner drew {len(tsp)} inner splits, the calibration design "
                                 f"{len(splits)}")
        tuner = BO.Tuner(arm, _FixedSplits(tsp[:1]), grid=(cfg,), mode="full", cv=corpus.cv)
        t0 = time.perf_counter()
        res = tuner.run(rows, fc.ctx, BO.model_seed_for_fold(k))
        rec["inner_fit_s"] = time.perf_counter() - t0
        iters = int(res.selected_iterations)
        rec["inner_trees"] = iters
        t0 = time.perf_counter()
        BO.BoostedArm(arm, fold_index=k, design=dsg, config=cfg, iterations=iters, cv=corpus.cv).fit(tr, fc.ctx)
        rec["cal_fit_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        BO.BoostedArm(arm, fold_index=k, design=dsg, config=cfg, iterations=iters, cv=corpus.cv).fit(rows, fc.ctx)
        rec["outer_refit_s"] = time.perf_counter() - t0
    elif arm == "B8":
        t0 = time.perf_counter()
        PG.B8Level(corpus.frame, cv=corpus.cv, fold=k).fit(tr, fc.ctx)
        rec["inner_fit_s"] = rec["cal_fit_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        PG.B8Level(corpus.frame, cv=corpus.cv, fold=k).fit(rows, fc.ctx)
        rec["outer_refit_s"] = time.perf_counter() - t0
    else:
        cfg = NN.NeuralConfig(8, 1e-3, 0 if arm == "M1" else 4)
        vs = validation_splits(fc)[:1]
        t0 = time.perf_counter()
        res = NN.tune(rows, vs, [cfg], model_seed=NN.registered_model_seed(k), condition_vectors=corpus.cv)
        rec["inner_fit_s"] = time.perf_counter() - t0
        ep = int(res.n_epochs)
        rec["inner_best_epoch"] = ep
        t0 = time.perf_counter()
        NN.FactorisedArm(cfg, n_epochs=ep, model_seed=NN.registered_model_seed(k), condition_vectors=corpus.cv).fit(tr)
        rec["cal_fit_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        NN.FactorisedArm(cfg, n_epochs=ep, model_seed=NN.registered_model_seed(k), condition_vectors=corpus.cv).fit(rows)
        rec["outer_refit_s"] = time.perf_counter() - t0
    rec["peak_rss_bytes"] = D.peak_rss_bytes()
    rec["n_test_rows_predicted"] = 0
    log(f"benchmark (addendum 1) {arm}: "
        f"{json.dumps({k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in rec.items()})}")
    return [rec]


def unit_costs_addendum1(meas: Sequence[Mapping[str, Any]], sealed: Sequence[Mapping[str, Any]]
                         ) -> dict[tuple[str, str], D.UnitCost]:
    """Unit costs of the addendum-1 plan: the re-measured V5 class per arm, the sealed-plan measurements for V1 and V2
    (whose inner design the addendum leaves unchanged), and B6's exact V5 folds priced with its batched measurement
    (the same inner design; the exact folds hide one cell of the outer training rows more)."""
    from dataclasses import replace as _replace

    ucs = dict(unit_costs_from_measurements(meas))
    for m in sealed:
        key = (m["arm"], m["design_class"])
        if m["design_class"] in ("V1", "V2") and key not in ucs:
            ucs[key] = D.UnitCost(arm=m["arm"], design_class=m["design_class"], inner_fit_s=float(m["inner_fit_s"]),
                                  cal_fit_s=float(m["cal_fit_s"]), outer_refit_s=float(m["outer_refit_s"]),
                                  splits_full=float(m["splits_full"]), splits_first=float(m["splits_first"]),
                                  source="benchmark (sealed-plan measurement; the V1 / V2 inner design is unchanged "
                                         "by addendum 1)")
    for (arm, dc), uc in list(ucs.items()):
        if dc == "V5_batched" and (arm, "V5_exact") not in ucs:
            ucs[(arm, "V5_exact")] = _replace(uc, design_class="V5_exact",
                                              source=uc.source + " (V5_batched measurement applied to the exact V5 "
                                                                 "folds: the same simultaneous inner design)")
    return _with_aliases(ucs)


def addendum1_markdown(summary: Mapping[str, Any], table: pd.DataFrame, meas: Sequence[Mapping[str, Any]],
                       checkpoints: Sequence[Mapping[str, Any]], conditional: Mapping[str, Any],
                       sealed: Mapping[str, Any] | None = None) -> str:
    lines = ["# Discovery cost estimate under POST-HOC addendum 1", "",
             "Generated by `scripts/g19_run_discovery.py --benchmark-addendum1` from fit-only timings on the TRAINING "
             "rows of one V5-primary batched fold (seed 104729). No test row was predicted; no score was computed. "
             "Each arm was timed on the fold's simultaneous inner splits (addendum 1 items 1-2: three inner folds, each "
             "ONE inner fit per configuration with all of that fold's inner cells hidden at once): one inner fit per "
             "configuration on one split, one calibration refit, one outer refit. V1 and V2 unit costs are the "
             "sealed-plan measurements, whose inner design the addendum leaves unchanged (`cost_estimate.json`). The "
             "estimate multiplies them by the addendum-1 plan (`discovery.enumerate_plan`): folds x (3 inner splits x "
             "configurations x inner fit + 3 calibration refits + outer refit + inner guard), every job on seed 104729 "
             "in `full` mode. Wall clock = compute / workers.", "",
             "## Measurements (seconds)", "",
             "| arm | design | fold | train rows | inner splits | cells hidden per split | inner config fit | "
             "calibration refit | outer refit | peak RSS (MB) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in meas:
        rss = m.get("peak_rss_bytes")
        cells = m.get("n_cells_per_split") or []
        lines.append(f"| {m['arm']} | {m['design_class']} | {m['fold_id']} | {m['n_train']} | {m['splits_full']} | "
                     f"{'/'.join(str(c) for c in cells)} | {m['inner_fit_s']:.1f} | {m['cal_fit_s']:.1f} | "
                     f"{m['outer_refit_s']:.1f} | {'' if rss is None else f'{rss / 1e6:.0f}'} |")
    lines += ["", "B6 'inner config fit' is all 12 (rank, lambda) configurations of one inner split (solved jointly); "
                  "its calibration is cross-fitted from those predictions, so it needs no refit.", "",
              "## Estimated compute by arm (hours, 1 worker)", "", "| arm | compute h |", "|---|---|"]
    for a, h in sorted(summary["compute_h_by_arm"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {a} | {h:.1f} |")
    lines += ["", "## Cumulative wall clock by stage (2 workers)", "",
              "| stage | jobs | folds | stage compute h | cumulative compute h | cumulative wall h | within 60 h |",
              "|---|---|---|---|---|---|---|"]
    for c in checkpoints:
        lines.append(f"| {c['stage']} | {c['n_jobs']} | {c['n_folds']} | {c['stage_compute_h']:.1f} | "
                     f"{c['cumulative_compute_h']:.1f} | {c['cumulative_wall_h']:.1f} | "
                     f"{'yes' if c['within_budget_wall'] else 'NO'} |")
    verdict = "FITS the 60 h budget" if summary["fits_budget"] else "does NOT fit the 60 h budget"
    lines += ["", f"**Total: {summary['total_compute_h']:.1f} h compute; {summary['total_wall_h']:.1f} h wall clock on "
                  f"{summary['workers']} workers against the {summary['budget_hours']:.0f} h budget -> {verdict}.**"]
    if not summary["fits_budget"]:
        lines.append(f"First job beyond the budget (plan order): `{summary['first_job_beyond_budget']}`. Section 7 item "
                     "5 demotes only M3-M7 on exhaustion, so every job of this plan keeps running past it.")
    if sealed is not None:
        lines += ["", f"Sealed section 7 plan, for comparison: {sealed.get('total_compute_h', float('nan')):.1f} h "
                      f"compute, {sealed.get('total_wall_h', float('nan')):.1f} h wall clock on 2 workers "
                      "(`cost_estimate.md`)."]
    lines += ["", "Conditional jobs not in the total: " + json.dumps(conditional, sort_keys=True), "",
              "Not run at all under addendum 1 (each named in the plan as a marker job): the pass over the four other "
              "discovery seeds; the loose, cell-only, parent-structure and Sr(III)-dropped V5 refits; the heavy-arm "
              "V5-P runs; the V1 / V2 refit sensitivities; every comparator-interval job.", ""]
    return "\n".join(lines) + "\n"


def write_cost_estimate_addendum1(meas: Sequence[Mapping[str, Any]], out_root: Path, workers: int = 2,
                                  sealed_meas: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """Price the addendum-1 plan with the re-measured unit costs and write
    ``evaluation/discovery/benchmark/cost_estimate_addendum1.json`` / ``.md`` (per-stage cumulative wall-clock
    checkpoints on 2 workers and the fits-in-60-h verdict)."""
    bdir = paths.ensure_dir(D.discovery_root(out_root) / "benchmark")
    # the V1 / V2 unit costs come from the sealed-plan measurement (unchanged by the addendum); it lives beside this
    # estimate, or in the repository's own benchmark directory when --out-root points elsewhere
    sealed_body = (D.read_record(bdir / "cost_estimate.json")
                   or D.read_record(D.discovery_root(paths.G19_ROOT) / "benchmark" / "cost_estimate.json") or {})
    sealed = list(sealed_meas if sealed_meas is not None else sealed_body.get("measurements") or [])
    ucs = unit_costs_addendum1(meas, sealed)
    state = D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed")
    jobs = D.enumerate_plan(state)
    counts = fold_counts(jobs, paths.FOLDS_DIR)
    table = D.estimate_cost(jobs, counts, ucs, other_costs_s=other_costs(jobs, counts), workers=workers,
                            fold_overhead_s=FOLD_OVERHEAD_S)
    summary = D.cost_summary(table, workers=workers)
    checkpoints = D.stage_checkpoints(table, workers=workers)
    v5p = D.JobSpec(kind="fit", arm="M2", design="V5P", variant="base", scheme="batched", seed=D.PRIMARY_SEED,
                    fold_seed=D.PRIMARY_SEED)
    pair = D.JobSpec(kind="fit", arm="M2", design="V5PAIR", variant="primary", scheme="batched", seed=D.PRIMARY_SEED,
                     fold_seed=D.PRIMARY_SEED)
    n_v5p, n_pair = fold_counts([v5p, pair], paths.FOLDS_DIR)[v5p.key], fold_counts([pair], paths.FOLDS_DIR)[pair.key]
    cond = {f"V5-P batched per heavy arm ({n_v5p} folds), hours -- NOT run under addendum 1 item 4": {
        a: round(n_v5p * ucs[(a, "V5_batched")].fold_seconds("full")["total"] / 3600, 2) for a in D.HEAVY_ARMS
        if (a, "V5_batched") in ucs},
        f"V5-PAIR batched per further freezing candidate ({n_pair} folds), hours": {
            a: round(n_pair * ucs[(a, "V5_batched")].fold_seconds("full")["total"] / 3600, 2)
            for a in D.HEAVY_ARMS if (a, "V5_batched") in ucs},
        "B6 re-coloured batches (seed 104729), hours": round(
            counts.get(D.JobSpec(kind="fit", arm="B6", design="V5", variant="primary", scheme="batched",
                                 seed=D.PRIMARY_SEED, fold_seed=D.PRIMARY_SEED, writes=D.B6_ARMS).key, 0)
            * ucs[("B6", "V5_batched")].fold_seconds("full")["total"] / 3600, 2) if ("B6", "V5_batched") in ucs else None}
    body = {"schema": D.SCHEMA, "addendum": REG.addenda_count(STAGE), "measurements": list(meas),
            "measurements_reused_from_sealed_plan": [m for m in sealed if m["design_class"] in ("V1", "V2")],
            "summary": summary, "stage_checkpoints": checkpoints, "conditional_not_in_total": cond,
            "fits_in_60h": bool(summary["fits_budget"]), "plan_state_assumed": state.record(),
            "per_job": table.to_dict(orient="records"),
            "sealed_plan_summary": sealed_body.get("summary"),
            "model": {"configs_per_split": D.CONFIGS_PER_SPLIT,
                      "calibration_fits_per_split": D.CALIBRATION_FITS_PER_SPLIT,
                      "inner_folds": D.INNER_N_FOLDS, "max_cells_per_inner_fold": D.INNER_MAX_CELLS_PER_FOLD,
                      "closed_form_s_per_fold_arm": PRESEAL_CLOSED_FORM_S, "safeguard_s_per_fold": SAFEGUARD_S_PER_FOLD,
                      "fold_overhead_s": FOLD_OVERHEAD_S, "wall_clock": "compute / workers"},
            "readings": {k: D.READINGS[k] for k in ("addendum1_inner_design", "addendum1_seeds",
                                                    "addendum1_sensitivities", "addendum1_v5pair", "plan_order",
                                                    "budget", "heavy_arm_intervals", "fold_ordinal")}}
    write_json(bdir / "cost_estimate_addendum1.json", body)
    write_text(bdir / "cost_estimate_addendum1.md",
               addendum1_markdown(summary, table, meas, checkpoints, cond, sealed_body.get("summary")))
    return body


def write_plan_listing(jobs: Sequence[D.JobSpec], counts: Mapping[str, int], out_root: Path) -> Path:
    """``--dry-run``: the addendum-1 job plan as text (one line per job, in plan order) for the record."""
    lines = [f"# gen19 discovery plan under POST-HOC addendum {REG.addenda_count(STAGE)} "
             f"(scripts/g19_run_discovery.py --dry-run); seeds {list(D.PLAN_SEEDS)}",
             f"# {sum(j.kind == 'fit' for j in jobs)} fit jobs, "
             f"{sum(counts.get(j.key, 0) for j in jobs if j.kind == 'fit')} outer folds, "
             f"{sum(j.kind == 'marker' for j in jobs)} markers (what is NOT run, named)"]
    for j in jobs:
        extra = "" if j.kind in ("marker", "h3_spec") else f"folds={counts.get(j.key, 0)}"
        lines.append(f"{j.stage:>24} | {j.kind:<20} | {j.key:<60} | {extra} | {j.message or j.purpose}")
    p = paths.ensure_dir(D.discovery_root(out_root) / "benchmark") / "plan_addendum1.txt"
    write_text(p, "\n".join(lines) + "\n")
    return p


# ============================================================================================= #
# main
# ============================================================================================= #

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, choices=(1, 2), default=2)
    ap.add_argument("--only", default=None, help="comma list: arm, design token, arm:design, or job kind")
    ap.add_argument("--dry-run", action="store_true", help="enumerate jobs with fold counts and the cost estimate")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="operator pause: dispatch no new fold once the wall-clock ledger reaches this many hours "
                         "(resumable; not the section 7 item 5 budget, which demotes only M3-M7)")
    ap.add_argument("--steps", default="point,intervals", help="point,intervals (default) or point")
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--benchmark", action="store_true",
                    help="fit-only timings of the SEALED section 7 inner design and its cost estimate (no prediction)")
    ap.add_argument("--benchmark-addendum1", action="store_true",
                    help="fit-only timings of the POST-HOC addendum 1 inner design on one V5-primary batched fold "
                         "(seed 104729) and the re-priced estimate in cost_estimate_addendum1.json / .md (no prediction)")
    ap.add_argument("--benchmark-force", action="store_true", help="re-measure arms already measured")
    ap.add_argument("--benchmark-arms", default=None, help="comma list of arms to re-measure with --benchmark-force")
    ap.add_argument("--benchmark-arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--benchmark-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--benchmark-mode", default="sealed", choices=("sealed", "addendum1"), help=argparse.SUPPRESS)
    ap.add_argument("--expect-addenda", type=int, default=None,
                    help="POST-HOC addenda expected below the sealed footer (default: the registry entry of stage "
                         f"'{STAGE}', else {D.N_ADDENDA_EXPECTED}); the run is refused on any other count")
    ap.add_argument("--no-manifest", action="store_true")
    ns = ap.parse_args(argv)
    ns.steps = [s for s in ns.steps.split(",") if s]
    if not set(ns.steps) <= set(D.STEPS) or D.STEPS[0] not in ns.steps:
        raise SystemExit(f"--steps must include 'point' and be among {D.STEPS}")
    return ns


def main(argv=None, *, check: Callable[[], int] | None = None,
         digests: Callable[[], Mapping[str, Any]] | None = None) -> int:
    ns = parse_args(argv)

    out_root = Path(ns.out_root)

    def gate() -> dict:
        # the runner's registry stage: 'discovery_candidates' once plan_state.freezing_candidates is non-empty (the
        # scorer wrote it; the freezing-candidate jobs are then the only jobs left to fit), else 'discovery'
        return refuse_unless_sealed(check, digests, expect_addenda=ns.expect_addenda, stage=runner_stage(out_root))
    prereg = gate()
    if ns.benchmark_arm:
        coext = json.loads(Path(ns.benchmark_out).with_suffix(".coext.json").read_text(encoding="utf-8"))
        corpus = load_corpus(coext)
        import torch
        torch.set_num_threads(2)
        measure = benchmark_addendum1_arm if ns.benchmark_mode == "addendum1" else benchmark_arm
        write_json(Path(ns.benchmark_out), measure(ns.benchmark_arm, corpus))
        return 0
    state = D.PlanState.read(plan_state_path(out_root))
    apply_safeguard_records(state, out_root)
    jobs = D.filter_jobs(D.enumerate_plan(state, include_h3_specs=True), ns.only)
    if ns.dry_run:
        counts = fold_counts(jobs, paths.FOLDS_DIR)
        for j in jobs:
            extra = "" if j.kind in ("marker", "h3_spec") else f"folds={counts.get(j.key, 0)}"
            print(f"{j.stage:>24} | {j.kind:<20} | {j.key:<60} | {extra} | {j.message or j.purpose}")
        listing = write_plan_listing(jobs, counts, out_root)
        print(f"plan written to {_rel(listing)}")
        bdir = D.discovery_root(out_root) / "benchmark"
        add1 = bdir / "cost_estimate_addendum1.json"
        bench = add1 if add1.exists() else bdir / "cost_estimate.json"
        if bench.exists():
            body = json.loads(bench.read_text(encoding="utf-8"))
            ucs = (unit_costs_addendum1(body["measurements"], body.get("measurements_reused_from_sealed_plan") or [])
                   if bench == add1 else _with_aliases(unit_costs_from_measurements(body["measurements"])))
            try:
                table = D.estimate_cost(jobs, counts, ucs, other_costs_s=other_costs(jobs, counts), workers=ns.workers,
                                        fold_overhead_s=FOLD_OVERHEAD_S)
                print(json.dumps({"priced_with": _rel(bench), **D.cost_summary(table, workers=ns.workers)}, indent=2))
                print(json.dumps(D.stage_checkpoints(table, workers=ns.workers), indent=2))
            except KeyError as exc:
                print(f"cost estimate unavailable: {exc} (run --benchmark-addendum1)")
        else:
            print("no benchmark yet: run --benchmark-addendum1 for the cost estimate")
        print(D.NOT_IMPLEMENTED)
        return 0
    if ns.benchmark or ns.benchmark_addendum1:
        mode = "addendum1" if ns.benchmark_addendum1 else "sealed"
        prefix = "measurements_addendum1_" if mode == "addendum1" else "measurements_"
        log(f"coextractant ids ({mode} benchmark)")
        coext = coextractant_ids()
        meas = []
        bdir = paths.ensure_dir(D.discovery_root(out_root) / "benchmark")
        force = set(a.strip() for a in (ns.benchmark_arms or "").split(",") if a.strip()) if ns.benchmark_force else set()
        with tempfile.TemporaryDirectory(prefix="g19_bench_") as td:
            for arm in BENCH_ARMS:
                kept = bdir / f"{prefix}{arm}.json"                  # timing outputs only (resumable per arm)
                redo = ns.benchmark_force and (not force or arm in force)
                if kept.exists() and not redo:
                    meas += json.loads(kept.read_text(encoding="utf-8"))
                    continue
                outp = Path(td) / f"{arm}.json"
                outp.with_suffix(".coext.json").write_text(json.dumps(coext), encoding="utf-8")
                env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(paths.G19_ROOT))
                r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--benchmark-arm", arm,
                                    "--benchmark-mode", mode, "--benchmark-out", str(outp), "--out-root",
                                    str(out_root)], cwd=str(paths.REPO_ROOT), env=env)
                if r.returncode != 0:
                    raise SystemExit(f"benchmark of {arm} failed ({r.returncode})")
                m = json.loads(outp.read_text(encoding="utf-8"))
                write_json(kept, m)
                meas += m
        with (Run(f"{NAME}_benchmark" + ("_addendum1" if mode == "addendum1" else ""),
                  args={k: v for k, v in vars(ns).items()}, seed=D.PRIMARY_SEED,
                  extra={"rule": "fit-only timings on outer-TRAINING rows of one fold per design class (seed 104729); "
                                 "no test row predicted, no score computed",
                         "inner_design": D.INNER_DESIGN_NAME if mode == "addendum1" else "sealed section 7 design",
                         "prereg_gate": prereg,
                         "n_test_rows_predicted": int(sum(int(m.get("n_test_rows_predicted", 0)) for m in meas))})
              if not ns.no_manifest else _Null()) as run:
            body = (write_cost_estimate_addendum1(meas, out_root, workers=2) if mode == "addendum1"
                    else write_cost_estimate(meas, out_root, workers=2))
            names = (("cost_estimate_addendum1.json", "cost_estimate_addendum1.md") if mode == "addendum1"
                     else ("cost_estimate.json", "cost_estimate.md"))
            if run is not None:
                run.inputs(paths.FOLDS_DIR / "INDEX.json",
                           *sorted({paths.FOLDS_DIR / f"{m['stem']}.{e}" for m in meas for e in ("json", "parquet")}))
                run.outputs(bdir / names[0], bdir / names[1], *sorted(bdir.glob(f"{prefix}*.json")))
                run.extra["summary"] = body["summary"]
        log(json.dumps(body["summary"], indent=2))
        if mode == "addendum1":
            log(f"fits in {D.BUDGET_HOURS:g} h on 2 workers: {body['fits_in_60h']}")
        return 0
    code = D.code_digest(CODE_FILES + (Path(__file__).resolve(),), RUNNER_OBJECTS)
    log("coextractant ids and corpus")
    coext = coextractant_ids()
    corpus = load_corpus(coext, with_cv=ns.workers == 1)
    ctx_run = Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=None,
                  extra={"discovery_seeds": list(D.DISCOVERY_SEEDS), "code_sha256": code, "readings": D.READINGS,
                         "prereg_gate": prereg}) \
        if not ns.no_manifest else None
    with (ctx_run or _Null()) as run:
        ledger = run_plan(corpus, out_root, steps=ns.steps, code=code["combined"], workers=ns.workers,
                          max_hours=ns.max_hours, only=ns.only, coext_ids=coext, gate=gate)
        if run is not None:
            outs = [plan_state_path(out_root), D.discovery_root(out_root) / "decisions" / "b6_checks.json",
                    *sorted((D.discovery_root(out_root) / "safeguard").glob("*.json")),
                    write_records_index(out_root)]
            run.outputs(*[p for p in outs if Path(p).exists()])
            run.extra.update({"ledger": _json_safe(ledger), "budget_wall_clock": D.budget_status(
                wall_clock_total(out_root)), "plan_state": D.PlanState.read(
                plan_state_path(out_root)).record(), "learned_arms_scored_on_confirmation_half": [],
                "v6_target_rows_scored": 0})
    return 0


def run_plan(corpus: Corpus, out_root: Path, *, steps: Sequence[str], code: str, workers: int = 1,
             max_hours: float | None = None, only: str | None = None, coext_ids: Sequence[str] = (),
             runners: Mapping[str, Any] | None = None, guard_fn: Callable = outer_guard,
             inner_check: Callable = inner_isolation_check,
             gate: Callable[[], Mapping[str, Any]] | None = None) -> dict:
    """Stage by stage in plan order: the safeguard records and the B6 checks change the later stages' jobs, so the plan
    is re-enumerated after every stage.  ``gate`` (the pre-registration seal and digest check) runs before every
    stage.  Stops at the ``M3+`` marker, on an error, or at the operator pause ``max_hours``."""
    t0 = time.perf_counter()
    prev_wall = wall_clock_total(out_root)
    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    def used() -> float:
        return prev_wall + time.perf_counter() - t0
    ledger_all: dict[str, Any] = {"jobs": {}, "markers": [], "stopped": None, "budget": None, "demoted": [],
                                  "prereg_gates": 0}
    done_stages: set[str] = set()
    recoloured_once = False
    while True:
        prereg = gate() if gate is not None else None
        ledger_all["prereg_gates"] += 1 if gate is not None else 0
        state = D.PlanState.read(plan_state_path(out_root))
        apply_safeguard_records(state, out_root)
        jobs = D.filter_jobs(D.enumerate_plan(state, folds_dir=corpus.folds_dir), only)
        pending = next((s for s in sorted({j.stage for j in jobs}) if s not in done_stages), None)
        if pending is None:
            break
        led = run_jobs([j for j in jobs if j.stage == pending], corpus, out_root, steps=steps, code=code, state=state,
                       workers=workers, max_hours=max_hours, coext_ids=coext_ids, runners=runners, guard_fn=guard_fn,
                       inner_check=inner_check, budget_used_s=used, prereg=prereg)
        ledger_all["jobs"].update(led["jobs"])
        ledger_all["markers"] += led["markers"]
        ledger_all["demoted"] += led.get("demoted", [])
        if led.get("budget") and ledger_all["budget"] is None:
            ledger_all["budget"] = led["budget"]
        done_stages.add(pending)
        if pending == D.STAGES["b6"] and not led.get("stopped"):
            state, checks = evaluate_b6_checks(out_root, corpus, state, code, runners=runners)
            write_json(D.discovery_root(out_root) / "decisions" / "b6_checks.json", checks)
            write_json(plan_state_path(out_root), state.record())
            log(f"B6 checks: V5 {state.v5_batched_check}, V1 {state.v1_tenfold_check}")
            if state.v5_batched_check == "recolour" and not recoloured_once:
                recoloured_once = True                      # section 7 item 6: run the re-coloured B6 folds, check again
                done_stages.discard(pending)
        record_wall_clock(out_root, started, time.perf_counter() - t0, sorted(done_stages))
        _write_progress(out_root, ledger_all, time.perf_counter() - t0)
        if led.get("stopped"):
            ledger_all["stopped"] = led["stopped"]
            break
    return ledger_all


def wall_clock_path(out_root: Path) -> Path:
    return D.discovery_root(out_root) / "decisions" / "wall_clock.json"


def wall_clock_total(out_root: Path) -> float:
    """Section 7 item 5: wall-clock discovery seconds of every earlier invocation."""
    body = D.read_record(wall_clock_path(out_root)) or {}
    return float(sum(float(i.get("seconds", 0.0)) for i in body.get("invocations", [])))


def record_wall_clock(out_root: Path, started: str, seconds: float, stages: Sequence[str]) -> None:
    body = D.read_record(wall_clock_path(out_root)) or {"schema": D.SCHEMA, "invocations": []}
    inv = [i for i in body.get("invocations", []) if i.get("started_utc") != started]
    inv.append({"started_utc": started, "seconds": round(float(seconds), 1), "stages_done": list(stages)})
    body["invocations"] = inv
    body["total_hours"] = round(sum(float(i["seconds"]) for i in inv) / 3600.0, 4)
    body["budget"] = D.budget_status(sum(float(i["seconds"]) for i in inv))
    paths.ensure_dir(wall_clock_path(out_root).parent)
    _atomic_json(body, wall_clock_path(out_root))


def write_records_index(out_root: Path) -> Path:
    """``evaluation/discovery/records_index.csv``: every fold record with its digest (the section 16 output list)."""
    from gen19ct.manifest import write_csv

    recs = []
    for js in sorted(D.discovery_root(out_root).rglob("*.json")):
        body = D.read_record(js)
        if not body or "steps" not in body or "job" not in body:
            continue
        pq = js.with_suffix(".parquet")
        recs.append({"record": js.relative_to(D.discovery_root(out_root)).as_posix(), "arm": body.get("arm"),
                     "job": body["job"].get("key"), "fold_id": body.get("fold_id"), "fold_hash": body.get("fold_hash"),
                     "digest": body.get("digest"), "steps": ",".join(sorted(body.get("steps", {}))),
                     "parquet_sha256": paths.digests(pq)["sha256"] if pq.exists() else None})
    return write_csv(pd.DataFrame(recs, columns=["record", "arm", "job", "fold_id", "fold_hash", "digest", "steps",
                                                 "parquet_sha256"]), D.discovery_root(out_root) / "records_index.csv")


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _write_progress(out_root: Path, ledger: Mapping[str, Any], seconds: float) -> None:
    led = D.ledger_from_records(out_root)
    body = {"script": NAME, "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "invocation_seconds": round(seconds, 1), "budget_wall_clock": D.budget_status(wall_clock_total(out_root)),
            "compute_seconds_total_records": round(float(led["seconds"].sum()), 1) if len(led) else 0.0,
            "compute_seconds_by_stage": led.groupby("stage")["seconds"].sum().round(1).to_dict() if len(led) else {},
            "compute_seconds_by_arm_step": ({f"{a}|{s}": v for (a, s), v in
                                             led.groupby(["arm", "step"])["seconds"].sum().round(1).items()}
                                            if len(led) else {}),
            "jobs": _json_safe(ledger.get("jobs", {})), "markers": ledger.get("markers", [])}
    write_json(paths.MANIFESTS_DIR / "run_info" / f"{NAME}_progress.json"
               if Path(out_root).resolve() == paths.G19_ROOT.resolve()
               else D.discovery_root(out_root) / "run_info_progress.json", body)


if __name__ == "__main__":
    raise SystemExit(main())
