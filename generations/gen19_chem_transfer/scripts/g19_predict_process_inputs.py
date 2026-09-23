"""``scripts/g19_predict_process_inputs.py`` -- the deployment prediction step of the process case: fits the deployed
configuration **M0 (= B5, CatBoost; alias ``B5`` in every record)** and fills ``evaluation/process/inputs/
todga_prnd_predictions.csv`` -- the 450-row request of ``scripts/g19_run_process.py`` (Nd and Pr in the TODGA / HNO3 /
aliphatic system ``sys_5cb78e5000d40860`` on the 25 x 9 (log acid, log ligand) grid) -- with ``mean_logD``, ``std_logD``,
the split-conformal intervals, ``domain_status``, ``support_score`` and ``nearest_support`` (pre-registration sections
11, 12, 13 and 14; POST-HOC addendum 2 (exploratory mode), addendum 3 item 2 (deployed configuration = M0) and
addendum 5 item 3 (the deployed lanthanide configuration is the WITHOUT-actinide fit)).

**Everything this script writes is labelled ``transfer-unsupported`` and is an EXPLORATORY deployment fit.**  S1 is
UNDECIDED (the confirmation run has not run), so section 14 forbids a registered process evaluation; the tables here
feed ``g19_run_process.py --exploratory`` only.  No confirmation-half fold, no V6 fold and no withheld seed is read or
needed: the fit is one outer fit on the whole training set, with the section 12 inner-fold calibration.

Readings (implementation choices where the sealed text is silent; :data:`READINGS`, written into the manifest)
--------------------------------------------------------------------------------------------------------------
* **Training rows.**  WITHOUT (the deployed lanthanide configuration, addendum 5 item 3): every MODEL row minus every
  actinide-ELEMENT row, unknown-state X(?) actinide rows included in the removal (``discovery.actinide_rows``, the
  section 11 WITHOUT rule).  WITH (section 11's other arm; disclosed beside it because section 11 makes the deployment
  choice either way): every MODEL row.  V6_TARGET_ROWS stay in training (this is a deployment fit, not V6; the single
  V6 run of section 3.4 is untouched) and are never a calibration row (``interface.scorable_mask``).
* **Configuration.**  Discovery tuned M0 per outer fold and the selection varies by fold, so the deployment fit takes
  the MODE of ``selected_config`` over the V5-primary SELECTION-half M0 records at discovery seed 104729
  (``evaluation/discovery/B5/V5__primary_batched_max4/s104729``; the records are verified against the registry first)
  and the MEDIAN of their outer iteration counts (section 7 tie rule on a tied mode: the smaller configuration).  No
  tuning is run here.  Model seed: the section 15 rule at fold number 0 (``boosted.model_seed_for_fold(0)``).
* **Intervals.**  Section 12 split conformal on the V5 inner design of section 7 as amended by addendum 1 item 1
  (``inner_design.SimultaneousInnerCells``, three inner folds, all inner cells hidden at once, run seed 104729), drawn
  on the fit's own training rows; the frozen configuration is refitted on each inner fold's training rows and the
  pooled absolute residuals give the 50 / 80 / 95 % quantiles (``discovery.CrossFitResidualConformal`` with one frozen
  arm per fold, the discovery runner's non-cross-fit branch).  The interval centre is the outer fit.
* **The adapter's columns.**  ``lower_95`` / ``upper_95`` = mean -+ q95; ``std_logD`` = q95 / 1.96 and
  ``conformal_q95`` = 1.96, so the adapter's Gaussian residual scale ``std_logD * conformal_q95 / 1.96`` equals q95 / 1.96
  (the Gaussian whose central 95 % interval is the conformal one, ``monte_carlo.REGISTRATION_READINGS['draw_residual']``).
  **A CatBoost arm has no ensemble: the five ``member_logD_k`` columns are all the mean**, so the registered draw
  (one member + a conformal residual) degenerates to mean + residual -- the fallback reading of POST-HOC addendum 2
  without the truncation; the table says so (``members_note``) and ``g19_run_process.py`` flags it.
* **Query rows.**  A request row becomes a corpus-shaped row by copying a TEMPLATE training row of the system (TODGA,
  nitrate, aliphatic dodecane, no modifier, no complexant, O/A 1, 25 C, lanthanide; the smallest index) and setting the
  request's acid and formal ligand concentration, the metal (Nd(III) / Pr(III)), a TRACER metal concentration
  (:data:`TRACER_METAL_M`, the median of the system's nitrate lanthanide rows -- the adapter applies gen18's loading
  correction to a tracer-limit table) and the median contact time of the system's nitrate rows.  Every other condition
  column is the template's.  The gen18 system ``sys_5cb78e5000d40860`` maps to the corpus system key through the
  ligand SMILES of the gen18 entry (single-extractant TODGA).
* **Support (section 13).**  ``SupportIndex`` and the tau thresholds on the fit's training rows, the registered
  ``support_score`` (s4 registered reading) and the domain-status label under the anion clause's first reading, exactly
  as the discovery runner's ``fold_support_frame``; ``nearest_support`` is ``<state>|<system label>`` when the exact
  pair has training rows, else the nearest radius metal and nearest ligand system.
* **Pr/Nd residual correlation (section 14).**  From the WITHOUT fit's inner calibration residuals (signed, per row):
  comparable pairs (``evaluation.pairs.comparable_pairs`` on publication group, system and condition key within an inner
  fold) whose two states are Pr(III) and Nd(III); Pearson correlation of the two residuals.  Fewer than
  :data:`MIN_RHO_PAIRS` pairs -> the Ln(III)-Ln(III) pairs are used instead and flagged.  No V6_TARGET_ROWS row is a
  calibration row, so none enters.

Outputs: ``evaluation/process/inputs/todga_prnd_predictions.csv`` (WITHOUT), ``todga_prnd_predictions_with_actinides.csv``
(WITH, labelled), ``prnd_residual_correlation.json``, ``deployment_fit.json`` (configuration source, calibration,
support summary, timings), ``manifests/g19_predict_process_inputs.json``.

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_predict_process_inputs.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.data import normalize as N  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import pairs as EP  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.evaluation import support as ES  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json  # noqa: E402
from gen19ct.models import boosted as BO  # noqa: E402
from gen19ct.models import interface as I  # noqa: E402
from gen19ct.process import gen18_adapter as GA  # noqa: E402

NAME = "g19_predict_process_inputs"
SCHEMA = "gen19.process_inputs.v1"
REGISTRY_STAGE = "process"
SCRIPTS = Path(__file__).resolve().parent
ARM = "B5"                      # the alias every M0 record is written under
DEPLOYED = "M0"                 # POST-HOC addendum 3 item 2
TRANSFER_UNSUPPORTED = "transfer-unsupported"
LABEL = f"{TRANSFER_UNSUPPORTED}; exploratory deployment fit (S1 UNDECIDED: not a registered run)"
SELECTION_DESIGN_DIR = "V5__primary_batched_max4"
SELECTION_SEED = D.PRIMARY_SEED
VARIANTS: tuple[str, ...] = ("WITHOUT", "WITH")
DEPLOYED_VARIANT = "WITHOUT"    # addendum 5 item 3
Z_975 = 1.959963984540054
TRACER_METAL_M = 1e-4           # a ROUNDED tracer reading; the corpus median of the system's nitrate Ln rows is 8e-5 M
MIN_RHO_PAIRS = 20
GEN18_SYSTEM_ID = "sys_5cb78e5000d40860"
PROCESS_PUB = "pub_process_request"
MEMBERS_NOTE = ("single-model arm (CatBoost, no ensemble): the 5 member_logD_k columns equal mean_logD; the registered "
                "draw degenerates to mean + conformal-scaled Gaussian residual (addendum 2 fallback reading, untruncated)")

READINGS: dict[str, str] = {
    "label": LABEL,
    "training_rows": ("WITHOUT: every MODEL row minus every actinide-element row, X(?) actinide rows included in the "
                      "removal (discovery.actinide_rows; section 11 WITHOUT); WITH: every MODEL row. V6_TARGET_ROWS stay in "
                      "training (a deployment fit, not V6) and are never calibration rows"),
    "deployed_variant": "WITHOUT (POST-HOC addendum 5 item 3: actinide rows enter the deployed Ln configuration only on helps)",
    "configuration": ("MODE of selected_config over the V5-primary selection-half M0 (alias B5) records at seed 104729 and "
                      "the MEDIAN outer iteration count; a tied mode resolves by the section 7 tie rule (lower depth, then "
                      "larger l2_leaf_reg); no tuning here; model seed = section 15 rule at fold 0"),
    "intervals": ("section 12 split conformal on the addendum-1 V5 inner design (SimultaneousInnerCells, 3 inner folds, run "
                  "seed 104729) drawn on the fit's training rows; the frozen configuration refitted per inner fold; pooled "
                  "absolute residuals -> q50 / q80 / q95 (CrossFitResidualConformal, frozen branch); centre = the outer fit"),
    "adapter_columns": ("lower_95 / upper_95 = mean -+ q95; std_logD = q95 / 1.96 and conformal_q95 = 1.96 so the adapter's "
                        "residual scale std_logD * conformal_q95 / 1.96 = q95 / 1.96; member_logD_0..4 = mean_logD"),
    "members": MEMBERS_NOTE,
    "query_rows": ("a template training row of the system (nitrate, aliphatic dodecane, no modifier / complexant, O/A 1, "
                   "25 C, lanthanide; smallest index) with the request's acid and formal ligand concentration, the metal, a "
                   f"tracer metal concentration {TRACER_METAL_M} M (a rounded tracer value, NOT the corpus median: the "
                   "median metal_concentration_M of the system's 1,303 nitrate lanthanide rows is 8e-5 M and the two "
                   "template rows SAE:11532 / 11533 carry 3.5e-4 M; task X finding V-PROC-06) and the median contact "
                   "time of the system's nitrate rows"),
    "system_key": "gen18 sys_5cb78e5000d40860 -> the corpus extractant_system_key through the gen18 entry's ligand SMILES",
    "support": ("SupportIndex + tau thresholds on the fit's training rows; support_score v1 with the registered s4; "
                "domain_status under the first reading of the anion clause (fold_support_frame of the discovery runner)"),
    "nearest_support": "<state>|<system label> when the exact pair has training rows, else <nearest radius metal>|<nearest ligand system>",
    "rho": ("Pearson correlation of the signed inner-calibration residuals over comparable Pr(III)/Nd(III) pairs "
            "(publication group, system, condition key, same inner fold) of the WITHOUT fit; fewer than "
            f"{MIN_RHO_PAIRS} pairs -> the Ln(III)-Ln(III) pairs, flagged; V6_TARGET_ROWS are never calibration rows"),
}

#: the files whose code this fit depends on (digested into the manifest)
CODE_FILES: tuple[Path, ...] = (Path(__file__).resolve(),
                                paths.G19_ROOT / "gen19ct" / "models" / "boosted.py",
                                paths.G19_ROOT / "gen19ct" / "models" / "features.py",
                                paths.G19_ROOT / "gen19ct" / "models" / "interface.py",
                                paths.G19_ROOT / "gen19ct" / "models" / "inner_design.py",
                                paths.G19_ROOT / "gen19ct" / "evaluation" / "support.py",
                                paths.G19_ROOT / "gen19ct" / "chemistry" / "support_graph.py",
                                paths.G19_ROOT / "gen19ct" / "data" / "normalize.py")


def _load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def inputs_dir(out_root: Path) -> Path:
    return Path(out_root) / "evaluation" / "process" / "inputs"


def request_path(out_root: Path) -> Path:
    return inputs_dir(out_root) / "todga_prnd_prediction_request.csv"


def predictions_path(out_root: Path, variant: str = DEPLOYED_VARIANT) -> Path:
    if variant == DEPLOYED_VARIANT:
        return inputs_dir(out_root) / "todga_prnd_predictions.csv"
    return inputs_dir(out_root) / "todga_prnd_predictions_with_actinides.csv"


def rho_path(out_root: Path) -> Path:
    return inputs_dir(out_root) / "prnd_residual_correlation.json"


def fit_record_path(out_root: Path) -> Path:
    return inputs_dir(out_root) / "deployment_fit.json"


# ============================================================================================= #
# the configuration (mode over the selection-half records)
# ============================================================================================= #

def selection_records(out_root: Path, *, design_dir: str = SELECTION_DESIGN_DIR, seed: int = SELECTION_SEED,
                      arm: str = ARM) -> list[dict[str, Any]]:
    """The M0 (alias B5) records of the V5-primary selection half at the discovery seed, each verified against the
    registry entry it was written under (``registry.verify_record``; a record that verifies against no entry refuses)."""
    d = D.discovery_root(out_root) / arm / design_dir / f"s{seed}"
    files = sorted(d.glob("*.json"))
    if not files:
        raise SystemExit(f"refused: no {arm} record under {d} (the discovery run is the configuration source)")
    out = []
    for js in files:
        rec = D.read_record(js)
        if rec is None or "selected_config" not in rec or "arm_record" not in rec:
            raise SystemExit(f"refused: {js} is not a complete {arm} record")
        if rec.get("fold_id", "").split("_")[1:2] not in (["S"], []) and "_S_" not in rec.get("fold_id", ""):
            raise SystemExit(f"refused: {js} is not a selection-half record (fold_id {rec.get('fold_id')!r})")
        if not (js.with_suffix(".parquet")).exists():
            raise SystemExit(f"refused: {js} has no parquet beside it")
        v = REG.verify_record(rec, None)
        if v.get("ok") is not True:
            raise SystemExit(f"refused: {js} does not verify against any registry entry of its stage: {v.get('reason')}")
        rec["_path"] = str(js)
        rec["_matched_entry"] = v.get("matched_entry")
        out.append(rec)
    return out


def deployed_configuration(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The mode of ``selected_config`` (section 7 tie rule on a tied mode) and the median outer iteration count."""
    if not records:
        raise ValueError("no records")
    labels = [str(r["selected_config"]) for r in records]
    counts = Counter(labels)
    top = max(counts.values())
    tied = sorted(k for k, v in counts.items() if v == top)
    by_label: dict[str, BO.CatBoostConfig] = {}
    for r in records:
        fz = (r.get("arm_record") or {}).get("frozen") or {}
        cfg = {k: v for k, v in dict(fz.get("config") or {}).items() if k != "label"}
        if cfg and str(r["selected_config"]) not in by_label:
            by_label[str(r["selected_config"])] = BO.CatBoostConfig(**cfg)
    missing = [t for t in tied if t not in by_label]
    if missing:
        raise ValueError(f"no frozen configuration recorded for {missing}")
    mode = min(tied, key=lambda k: by_label[k].complexity_key()) if len(tied) > 1 else tied[0]
    iters = [int(((r.get("arm_record") or {}).get("frozen") or {}).get("iterations")) for r in records]
    median = int(round(statistics.median(iters)))
    return {"selected_config": mode, "config": by_label[mode], "config_record": by_label[mode].record(),
            "iterations": median, "counts": dict(sorted(counts.items())), "n_records": len(records),
            "tied_modes": tied if len(tied) > 1 else [], "iterations_all": sorted(iters),
            "rule": READINGS["configuration"],
            "records_verified": dict(Counter(str(r.get("_matched_entry")) for r in records)),
            "record_paths": [paths.rel(Path(r["_path"])) for r in records if r.get("_path")]}


# ============================================================================================= #
# query rows (corpus-shaped rows for the request)
# ============================================================================================= #

def gen19_system_key(entry: Any, systems: pd.DataFrame) -> str:
    """The corpus ``extractant_system_key`` of a gen18 system entry: the single-extractant system whose primary SMILES is
    the entry's ligand SMILES."""
    ligs = list(getattr(entry, "organic_ligands", []) or [])
    if len(ligs) != 1:
        raise ValueError(f"gen18 entry {getattr(entry, 'system_id', '?')} has {len(ligs)} organic ligands; one expected")
    smi = getattr(ligs[0], "smiles", None)
    if not smi:
        raise ValueError("gen18 entry ligand has no SMILES")
    tab = systems.reset_index(drop=True) if systems.index.name == "extractant_system_key" else systems
    hit = tab[(tab["primary_extractant_smiles"].astype(str) == str(smi)) & (tab["n_organic_extractants"].astype(int) == 1)]
    keys = sorted(set(hit["extractant_system_key"].astype(str)))
    if len(keys) != 1:
        raise ValueError(f"{len(keys)} corpus systems match the gen18 ligand SMILES {smi!r}: {keys}")
    return keys[0]


def template_rows(frame: pd.DataFrame, system_key: str, *, metals: Sequence[str]) -> dict[str, pd.Series]:
    """Per requested metal, the template training row (module docstring ``query_rows`` reading): the system's nitrate,
    aliphatic dodecane, additive-free, O/A 1, 25 C lanthanide row of that element with the smallest index; when the
    element has none, the lanthanide template with the element's identity columns taken from any row of the element."""
    sel = frame[(frame[SG.SYSTEM_COL] == system_key) & (frame[SG.ACID_ANION_COL] == "nitrate")
                & (frame.get("diluent_family", pd.Series(index=frame.index, dtype=object)) == "aliphatic")]
    if "solvent_primary" in frame.columns:
        sel = sel[sel["solvent_primary"] == "dodecane"]
    for c in ("modifier_name", "complexant_name"):
        if c in sel.columns:
            sel = sel[sel[c].isna()]
    if "phase_ratio_org_aq" in sel.columns:
        sel = sel[sel["phase_ratio_org_aq"] == 1.0]
    if SG.TEMP_COL in sel.columns:
        sel = sel[sel[SG.TEMP_COL] == 25.0]
    if "metal_category" in sel.columns:
        sel = sel[sel["metal_category"] == "lanthanide"]
    if not len(sel):
        raise ValueError(f"no template row for system {system_key!r} under the reading")
    sel = sel.sort_index()
    base = sel.iloc[0]
    out: dict[str, pd.Series] = {}
    for m in metals:
        own = sel[sel[SG.ELEMENT_COL] == m]
        if len(own):
            out[m] = own.iloc[0].copy()
            continue
        any_row = frame[(frame[SG.ELEMENT_COL] == m) & frame[SG.METAL_COL].notna()].sort_index()
        if not len(any_row):
            raise ValueError(f"the corpus has no row of element {m!r}")
        t = base.copy()
        for c in ("metal_symbol", "metal_oxidation_state", "metal_category", "is_lanthanide", "atomic_number",
                  "lanthanide_index", "ionic_radius_cn8_A", "ionic_radius_status", "series_id", "g19_ox",
                  SG.METAL_COL, SG.ELEMENT_COL):
            if c in any_row.columns:
                t[c] = any_row.iloc[0][c]
        out[m] = t
    return out


def median_contact_time(frame: pd.DataFrame, system_key: str) -> float:
    sel = frame[(frame[SG.SYSTEM_COL] == system_key) & (frame[SG.ACID_ANION_COL] == "nitrate")]
    v = pd.to_numeric(sel.get("contact_time_min", pd.Series(dtype=float)), errors="coerce").dropna()
    return float(v.median()) if len(v) else float("nan")


def query_rows(request: pd.DataFrame, frame: pd.DataFrame, *, system_key: str, templates: Mapping[str, pd.Series],
               tracer_metal_M: float = TRACER_METAL_M, contact_time_min: float | None = None) -> pd.DataFrame:
    """Corpus-shaped query rows for every request row (module docstring reading); index labels continue the frame's."""
    need = ("metal", "log_acid", "log_ligand")
    missing = [c for c in need if c not in request.columns]
    if missing:
        raise KeyError(f"request lacks {missing}")
    start = int(frame.index.max()) + 1 if len(frame) and pd.api.types.is_integer_dtype(frame.index) else 0
    recs, labels = [], []
    for k, r in enumerate(request.itertuples(index=False)):
        metal = str(r.metal)
        if metal not in templates:
            raise KeyError(f"no template for metal {metal!r}")
        acid_M, lig_M = float(10.0 ** float(r.log_acid)), float(10.0 ** float(r.log_ligand))
        row = dict(templates[metal].to_dict())          # a dict holds array-valued cells without broadcasting
        rid = f"PROC:{metal}:{k:04d}"
        row[FI.ROW_ID] = rid
        for c in ("source_record_id", "representative_raw_row_id"):
            if c in row:
                row[c] = rid
        row[SG.METAL_COL] = f"{metal}(III)"
        row[SG.ELEMENT_COL] = metal
        if "metal_symbol" in row:
            row["metal_symbol"] = metal
        row["acid_concentration_M"] = acid_M
        row[SG.LOG_ACID_COL] = float(r.log_acid)
        row["extractant_primary_concentration_M"] = lig_M
        row[SG.LOG_EXT_COL] = float(r.log_ligand)
        if "extractant_concentrations_M" in row:
            row["extractant_concentrations_M"] = np.array([lig_M], dtype=float)
        comps = row.get("components")
        if comps is not None and not (isinstance(comps, float) and math.isnan(comps)):
            new = []
            for c in list(np.atleast_1d(comps)):
                c = dict(c)
                if c.get("role") == "organic_extractant":
                    c["concentration_M"] = lig_M
                    c["concentration_raw"] = f"{lig_M:.6g} M"
                new.append(c)
            arr = np.empty(len(new), dtype=object)
            arr[:] = new
            row["components"] = arr
        row["metal_concentration_M"] = float(tracer_metal_M)
        if "log10_metal_M" in row:
            row["log10_metal_M"] = math.log10(tracer_metal_M)
        row[SG.TEMP_COL] = 25.0
        if "phase_ratio_org_aq" in row:
            row["phase_ratio_org_aq"] = 1.0
        if contact_time_min is not None and "contact_time_min" in row:
            row["contact_time_min"] = float(contact_time_min)
        for c in ("nitrate_concentration_M", "acid_concentration_organic_M", "shaking_time_min", "modifier_name",
                  "modifier_concentration_M", "complexant_name", "complexant_smiles_canonical", "complexant_concentration_M",
                  "holdback_name", "holdback_smiles_canonical", "holdback_concentration_M", "g19_bundle_exp_id",
                  "D_value", "D_raw"):
            if c in row:
                row[c] = np.nan
        for c in (SG.PUB_COL, I.PUB_GROUP_COL, FI.GROUP_COL, "g19_study_id"):
            if c in row:
                row[c] = PROCESS_PUB
        if "duplicate_group_id" in row:
            row["duplicate_group_id"] = f"PROCDG{k:04d}"
        row[I.TARGET_COL] = np.nan
        recs.append(row)
        labels.append(start + k)
    out = pd.DataFrame.from_records(recs, columns=list(templates[next(iter(templates))].index))
    out.index = pd.Index(labels, dtype=frame.index.dtype if len(frame) else "int64")
    out[FI.ROW_ID] = out[FI.ROW_ID].astype(object)
    return out


# ============================================================================================= #
# the fit
# ============================================================================================= #

def training_mask(frame: pd.DataFrame, variant: str) -> tuple[np.ndarray, int]:
    """``(mask over the frame's rows, number of actinide rows removed)``."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}")
    an = D.actinide_rows(frame)
    if variant == "WITHOUT":
        return ~an, int(an.sum())
    return np.ones(len(frame), dtype=bool), 0


def fit_context(corpus: Any, *, seed: int, isolation_check: Callable, guard_cache: dict | None = None) -> I.FitContext:
    return I.FitContext(systems=corpus.systems, components=corpus.comps, table=corpus.table, hidden_index=None,
                        v6_mask=corpus.v6, exclude_from_scoring=corpus.coext, isolation_check=isolation_check,
                        guard_cache={} if guard_cache is None else guard_cache, seed=int(seed))


def deployment_fit(corpus: Any, queries: pd.DataFrame, *, variant: str, config: BO.CatBoostConfig, iterations: int,
                   seed: int, design: Any, splitter_design: Any, guard: str, isolation_check: Callable,
                   arm_frame: pd.DataFrame, cv: pd.DataFrame | None, feature_factory: Callable | None = None,
                   fold_number: int = 0, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """One variant's deployment fit: inner-design splits on the training rows, the frozen configuration refitted per
    inner fold for the calibration residuals (``CrossFitResidualConformal``), the outer fit, the query predictions with
    the section 12 intervals, the section 13 support of every query and the signed calibration residuals.

    ``design`` is the ``interface``-level inner design (``SimultaneousInnerCells``) whose ``splits`` are drawn once;
    ``splitter_design`` the tuner-level design object handed to ``BoostedArm`` (unused for a frozen configuration but
    recorded); ``feature_factory`` overrides the registered B5 feature set (tests only)."""
    say = progress or (lambda s: None)
    t = corpus.table
    mask, n_removed = training_mask(corpus.frame, variant)
    if not np.array_equal(t.index, corpus.frame.index):
        raise AssertionError("the RowTable index differs from the frame index")
    rows = corpus.frame.loc[t.index[mask]]
    ctx = fit_context(corpus, seed=seed, isolation_check=isolation_check).for_training(rows)
    t0 = time.perf_counter()
    splits = design.splits(t, mask, ctx)
    if not splits:
        raise RuntimeError(f"{variant}: the inner design produced no split")
    for sp in splits:
        if (sp.train_mask & ~mask).any() or not mask[sp.cal_positions].all() or sp.train_mask[sp.cal_positions].any():
            raise AssertionError(f"{variant}: inner split {sp.unit} is not inside the training rows")
        if variant == "WITHOUT" and D.actinide_rows(corpus.frame.iloc[sp.cal_positions]).any():
            raise AssertionError(f"{variant}: an actinide row among the calibration rows")
        if corpus.v6.to_numpy(dtype=bool)[sp.cal_positions].any():
            raise AssertionError(f"{variant}: a V6_TARGET_ROWS row among the calibration rows")
    folds = sorted({int(sp.fold) for sp in splits})
    split_seconds = time.perf_counter() - t0
    say(f"{variant}: {int(mask.sum())} training rows ({n_removed} actinide rows removed), {len(splits)} inner splits "
        f"(folds {folds}; calibration rows {[int(len(s.cal_positions)) for s in splits]}) in {split_seconds:.1f} s")

    def make_arm() -> BO.BoostedArm:
        return BO.BoostedArm(ARM, fold_index=fold_number, design=splitter_design, inner_mode="full", config=config,
                             iterations=int(iterations), frame=arm_frame, cv=cv, feature_factory=feature_factory)

    sel = {j: {"selected": config.label, "iterations": int(iterations), "source": "frozen deployment configuration (mode)"}
           for j in folds}
    t1 = time.perf_counter()
    wrapper = D.CrossFitResidualConformal({j: make_arm() for j in folds}, D.FixedInnerSplits(splits, D.INNER_DESIGN_NAME),
                                          guard=guard, selections=sel)
    wrapper.fit_table(t, mask, ctx)
    cal_seconds = time.perf_counter() - t1
    say(f"{variant}: calibration on {len(wrapper.residuals)} rows in {cal_seconds:.1f} s; quantiles "
        f"{ {k: round(v, 4) for k, v in wrapper.quantiles.items()} }")
    t2 = time.perf_counter()
    outer = make_arm().fit_table(t, mask, ctx)
    D.assert_no_provenance_features(outer.fitted.columns, f"{ARM} deployment features")
    wrapper.fitted_arm = outer
    pred = wrapper.predict(queries)
    fit_seconds = time.perf_counter() - t2
    say(f"{variant}: outer fit ({outer.fitted.iterations} trees, {len(outer.fitted.columns)} feature columns) and "
        f"{len(pred)} predictions in {fit_seconds:.1f} s")
    t3 = time.perf_counter()
    support = query_support(corpus, mask, ctx, queries)
    sup_seconds = time.perf_counter() - t3
    say(f"{variant}: support of {len(support)} queries in {sup_seconds:.1f} s; statuses "
        f"{dict(Counter(support['domain_status']))}")
    detail = [{"fold": d["fold"], "labels": list(d["labels"]), "signed": np.asarray(d["signed"], dtype=float)}
              for d in wrapper.calibration_detail]
    from gen19ct.models import inner_design as ID
    return {"variant": variant, "n_train": int(mask.sum()), "n_actinide_rows_removed": n_removed,
            "pred": pred, "support": support, "calibration_detail": detail,
            "quantiles": {str(k): float(v) for k, v in wrapper.quantiles.items()},
            "n_calibration": int(len(wrapper.residuals)), "calibration_folds": folds,
            "splits": ID.splits_summary(splits).to_dict(orient="records"),
            "arm_record": outer.record(), "feature_columns": list(outer.fitted.columns),
            "seconds": {"splits": round(split_seconds, 1), "calibration": round(cal_seconds, 1),
                        "outer_fit_and_predict": round(fit_seconds, 1), "support": round(sup_seconds, 1)}}


# ============================================================================================= #
# section 13 support of the query rows (the discovery runner's fold_support_frame on query rows)
# ============================================================================================= #

def query_support(corpus: Any, mask: np.ndarray, ctx: I.FitContext, queries: pd.DataFrame) -> pd.DataFrame:
    from gen19ct.models import baselines as B

    t = corpus.table
    sup = corpus.frame[list(SG.REQUIRED_COLUMNS)]
    train = sup.loc[t.index[mask]]
    si = SG.SupportIndex(train, systems=corpus.systems)
    tau_in, tau_ext, tau_max, n_tau = ES.tau_thresholds(si, train)
    eng = B.make_engine(t, mask, ctx)
    mu, sd = eng._d_desc_scale(B._BASE)
    row_family = np.array([t.sys_family[s] if s >= 0 else None for s in t.sys], dtype=object)
    row_expert = np.array([t.sys_expert[s] if s >= 0 else None for s in t.sys], dtype=object)
    fam_counts, exp_counts = Counter(row_family[mask]), Counter(row_expert[mask])
    mp = np.flatnonzero(mask & (t.sys >= 0))
    sys_anion = set(zip(t.sys[mp].tolist(), t.anion[mp].tolist()))
    fam_anion = set(zip(row_family[mp].tolist(), t.anion[mp].tolist()))
    fam_systems: dict = {}
    for s_code in np.unique(t.sys[mp]):
        fam_systems.setdefault(t.sys_family[s_code], []).append(t.sys_labels[s_code])
    labels_of_system = {}
    if "system_label" in corpus.frame.columns:
        lab = corpus.frame.loc[mask, [SG.SYSTEM_COL, "system_label"]].dropna().drop_duplicates(SG.SYSTEM_COL)
        labels_of_system = dict(zip(lab[SG.SYSTEM_COL].astype(str), lab["system_label"].astype(str)))
    s4_cache: dict = {}
    recs = []
    for lab, q in queries.iterrows():
        m, s = str(q[SG.METAL_COL]), str(q[SG.SYSTEM_COL])
        anion = I.NA_ANION if SG._missing(q[SG.ACID_ANION_COL]) else str(q[SG.ACID_ANION_COL])
        acid, ext, tp = float(q[SG.LOG_ACID_COL]), float(q[SG.LOG_EXT_COL]), float(q[SG.TEMP_COL])
        cond = {SG.LOG_ACID_COL: acid, SG.LOG_EXT_COL: ext, SG.TEMP_COL: tp,
                SG.ACID_ANION_COL: None if anion == I.NA_ANION else anion}
        f = si.features(m, s, cond)
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
        comps = ES.support_components(f, s4=s4_cache[(m, s)], tau_ext=tau_ext, metal_series=SG.metal_properties(m)["series"])
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
        exact = int(f["exact_pair_rows"])
        if exact > 0:
            nearest = f"{m}|{labels_of_system.get(s, s)}"
        else:
            nl = f.get("nearest_ligand_system")
            nearest = f"{f.get('nearest_radius_metal')}|{labels_of_system.get(str(nl), nl)}"
        rec = {"query": lab, "row_id": q[FI.ROW_ID], "metal_state": m, "system_key": s, "exact_pair_rows": exact,
               "condition_distance_pair": f["condition_distance_pair"], "condition_distance_system": f["condition_distance_system"],
               "n_publications_system": f["n_publications_system"], "nearest_radius_metal": f.get("nearest_radius_metal"),
               "nearest_ligand_system": f.get("nearest_ligand_system")}
        rec.update({f"support_{k}": v for k, v in comps.items()})
        rec.update({"support_score": ES.support_score(comps), "domain_status": sa[0],
                    "domain_status_reading_neither": sb[0], "domain_status_ambiguous": sa[0] != sb[0],
                    "condition_extrapolated": sa[1], "nearest_support": nearest,
                    "tau_in": tau_in, "tau_ext": tau_ext, "tau_max": tau_max, "n_tau_rows": n_tau})
        recs.append(rec)
    return pd.DataFrame(recs).set_index("query")


# ============================================================================================= #
# the Pr / Nd residual correlation (section 14)
# ============================================================================================= #

def residual_correlation(calibration_detail: Sequence[Mapping[str, Any]], frame: pd.DataFrame, v6_mask: pd.Series,
                         *, metals: tuple[str, str] = ("Pr(III)", "Nd(III)"), min_pairs: int = MIN_RHO_PAIRS
                         ) -> dict[str, Any]:
    """Pearson correlation of the signed inner-calibration residuals over comparable pairs (module docstring reading)."""
    labels: list[Any] = []
    folds: list[int] = []
    resid: list[float] = []
    for d in calibration_detail:
        labels += list(d["labels"])
        folds += [int(d["fold"])] * len(d["labels"])
        resid += [float(x) for x in d["signed"]]
    if not labels:
        return {"rho": 0.0, "n_pairs": 0, "source": "no calibration residual", "flag": "independent draws (rho = 0)"}
    idx = pd.Index(labels)
    if v6_mask.reindex(idx).fillna(False).astype(bool).any():
        raise AssertionError("a V6_TARGET_ROWS row among the calibration residuals")
    rows = frame.loc[idx]
    df = pd.DataFrame({EM.METAL_STATE_COL: rows[SG.METAL_COL].to_numpy(dtype=object),
                       EM.SYSTEM_COL: rows[SG.SYSTEM_COL].to_numpy(dtype=object),
                       EM.PUB_GROUP_COL: rows[I.PUB_GROUP_COL].to_numpy(dtype=object),
                       EM.CONDITION_KEY_COL: N.condition_key(rows).to_numpy(dtype=object),
                       EM.Y_COL: pd.to_numeric(rows[I.TARGET_COL], errors="coerce").to_numpy(dtype=float),
                       "fold": np.asarray(folds, dtype=int), "resid": np.asarray(resid, dtype=float)}, index=idx)
    pairs = EP.comparable_pairs(df, fold_col="fold", carry_cols=("resid",))
    out: dict[str, Any] = {"n_calibration_rows": int(len(df)), "n_comparable_pairs_all": int(len(pairs))}
    want = set(metals)
    prnd = pairs[[{a, b} == want for a, b in zip(pairs["state_a"], pairs["state_b"])]] if len(pairs) else pairs
    lnln = pairs[pairs["category_class"] == "Ln-Ln"] if len(pairs) else pairs

    def corr(p: pd.DataFrame) -> float | None:
        if len(p) < 3:
            return None
        a, b = p["resid_a"].to_numpy(dtype=float), p["resid_b"].to_numpy(dtype=float)
        if np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    r_prnd, r_ln = corr(prnd), corr(lnln)
    out.update({"n_pairs_prnd": int(len(prnd)), "rho_prnd": r_prnd, "n_pairs_ln_ln": int(len(lnln)), "rho_ln_ln": r_ln,
                "min_pairs": int(min_pairs)})
    if r_prnd is not None and len(prnd) >= min_pairs:
        out.update({"rho": r_prnd, "n_pairs": int(len(prnd)), "source": "Pr(III)/Nd(III) comparable pairs of the "
                    "WITHOUT fit's inner calibration residuals (section 14)", "flag": None})
    elif r_ln is not None and len(lnln) >= min_pairs:
        out.update({"rho": r_ln, "n_pairs": int(len(lnln)),
                    "source": "Ln(III)-Ln(III) comparable pairs of the inner calibration residuals (Pr/Nd pairs too few)",
                    "flag": f"fewer than {min_pairs} Pr/Nd pairs ({len(prnd)}): the Ln-Ln correlation stands in"})
    else:
        out.update({"rho": 0.0, "n_pairs": int(len(prnd)), "source": "too few comparable pairs",
                    "flag": "independent draws (rho = 0): section 14 says these inflate logSF uncertainty"})
    out["rho"] = float(max(-0.999, min(0.999, out["rho"])))
    return out


# ============================================================================================= #
# the prediction table
# ============================================================================================= #

def prediction_table(request: pd.DataFrame, pred: pd.DataFrame, support: pd.DataFrame, queries: pd.DataFrame, *,
                     variant: str, system_key: str, label: str = LABEL) -> pd.DataFrame:
    """The adapter's layout (``gen18_adapter.PREDICTION_COLUMNS`` + member columns) for one variant."""
    if len(pred) != len(request) or len(support) != len(request):
        raise ValueError("prediction / support / request lengths differ")
    mean = pred["mean_logD"].to_numpy(dtype=float)
    q95 = pred["conformal_q95"].to_numpy(dtype=float)
    if not np.isfinite(mean).all() or not np.isfinite(q95).all() or (q95 < 0).any():
        raise ValueError("non-finite prediction or conformal quantile")
    sup = support.set_index(pd.RangeIndex(len(support)))
    out = request.drop(columns=[c for c in request.columns if c in GA.PREDICTION_COLUMNS[4:] or c == "note"
                                or c.startswith(GA.MEMBER_COLUMN_PREFIX) or c == GA.CONFORMAL_Q95_COLUMN]).copy()
    out = out.reset_index(drop=True)
    out["mean_logD"] = mean
    out["std_logD"] = q95 / Z_975
    out["lower_95"], out["upper_95"] = mean - q95, mean + q95
    out["domain_status"] = sup["domain_status"].to_numpy(dtype=object)
    out["support_score"] = sup["support_score"].to_numpy(dtype=float)
    out["nearest_support"] = sup["nearest_support"].to_numpy(dtype=object)
    out["arm"] = DEPLOYED
    for c in GA.MEMBER_COLUMNS:
        out[c] = mean
    out[GA.CONFORMAL_Q95_COLUMN] = Z_975
    out["conformal_q95_abs"] = q95
    for lv in (50, 80):
        out[f"lower_{lv}"] = pred[f"lower_{lv}"].to_numpy(dtype=float)
        out[f"upper_{lv}"] = pred[f"upper_{lv}"].to_numpy(dtype=float)
    out["conformal_n_calibration"] = pred["conformal_n_calibration"].to_numpy()
    out["arm_alias"] = ARM
    out["training_rows"] = f"{variant}_actinide"
    out["gen19_system_key"] = system_key
    out["query_row_id"] = queries[FI.ROW_ID].to_numpy(dtype=object)
    out["condition_distance_pair"] = sup["condition_distance_pair"].to_numpy(dtype=float)
    out["exact_pair_rows"] = sup["exact_pair_rows"].to_numpy()
    out["label"] = label
    out["members_note"] = MEMBERS_NOTE
    GA.PredictionTable.from_frame(out)          # the adapter accepts it (rectangular, vocabulary, member mean)
    return out


# ============================================================================================= #
# main
# ============================================================================================= #

def code_digest() -> dict[str, Any]:
    return D.code_digest(CODE_FILES)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=SELECTION_SEED, help="run seed of the inner design (default 104729)")
    ap.add_argument("--iterations-override", type=int, default=None,
                    help="SMOKE ONLY: replace the median iteration count (refused on the real out-root)")
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, check: Callable[[], int] | None = None,
         digests: Callable[[], Mapping[str, Any]] | None = None) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    real = out_root.resolve() == paths.G19_ROOT.resolve()
    if ns.iterations_override is not None and real:
        raise SystemExit("refused: --iterations-override is a smoke setting; the real inputs take the median iteration count")
    prereg = REG.refuse_unless_sealed(REGISTRY_STAGE, check, digests, expect_addenda=ns.expect_addenda)
    # the Run context opens HERE, before the corpus load and the fits, so the manifest's runtime_s is the measured wall
    # clock of the step: it used to open after both variants were fitted and recorded 0.477 s for an 18.5-minute run
    # (task X finding V-PROC-05); the per-step seconds stay in deployment_fit.json -> variants.*.seconds
    ctx = Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=int(ns.seed)) if not ns.no_manifest else _Null()
    t_start = time.perf_counter()
    with ctx as run:
        RD = _load_script("g19_run_discovery")
        req_path = request_path(out_root) if real else request_path(paths.G19_ROOT)
        if not req_path.exists():
            raise SystemExit(f"refused: the request {req_path} is missing (g19_run_process.py --exploratory writes it)")
        request = pd.read_csv(req_path)
        sids = sorted(set(request["system_id"].astype(str)))
        if sids != [GEN18_SYSTEM_ID]:
            raise SystemExit(f"refused: the request names systems {sids}, expected [{GEN18_SYSTEM_ID}]")
        records = selection_records(paths.G19_ROOT)
        conf = deployed_configuration(records)
        iterations = int(conf["iterations"]) if ns.iterations_override is None else int(ns.iterations_override)
        log(f"configuration: {conf['selected_config']} (counts {conf['counts']}), median iterations {conf['iterations']}"
            + (f" -> SMOKE override {iterations}" if ns.iterations_override is not None else "")
            + f"; {conf['n_records']} records verified {conf['records_verified']}")
        t0 = time.perf_counter()
        corpus = RD.load_corpus(RD.coextractant_ids())
        log(f"corpus: {len(corpus.frame)} MODEL rows in {time.perf_counter() - t0:.1f} s")
        paths.add_gen18_to_path()
        from gen18proc.systems import load_system

        entry = load_system(paths.GEN18_ROOT / "systems" / f"{GEN18_SYSTEM_ID}.json")
        system_key = gen19_system_key(entry, corpus.systems)
        metals = sorted(set(request["metal"].astype(str)))
        templates = template_rows(corpus.frame, system_key, metals=metals)
        ct = median_contact_time(corpus.frame, system_key)
        queries = query_rows(request, corpus.frame, system_key=system_key, templates=templates, contact_time_min=ct)
        arm_frame = pd.concat([corpus.frame, queries])
        if not arm_frame.index.is_unique:
            raise AssertionError("query labels collide with the frame")
        cv = N.condition_vector(arm_frame)
        log(f"queries: {len(queries)} rows for {metals} in system {system_key[:40]}... (template rows "
            f"{ {m: str(t[FI.ROW_ID]) for m, t in templates.items()} }, contact time {ct} min)")
        job = D.JobSpec(kind="fit", arm=ARM, design="V5", variant="primary", scheme="batched_max4", seed=int(ns.seed),
                        fold_seed=int(ns.seed), stage=D.STAGES["p_b5"], writes=(ARM,), registered=False,
                        purpose="exploratory deployment fit of M0 for the process case (section 14)")
        state = D.PlanState.read(RD.plan_state_path(paths.G19_ROOT))
        guard = RD.guard_mode_for(job, state)
        design = RD.simultaneous_inner_design(job, corpus)
        splitter_design = RD.inner_design_object(job, corpus)
        check_fn = RD.inner_isolation_check(job, corpus)
        results: dict[str, dict[str, Any]] = {}
        tables: dict[str, Path] = {}
        for variant in ns.variants:
            res = deployment_fit(corpus, queries, variant=variant, config=conf["config"], iterations=iterations,
                                 seed=int(ns.seed), design=design, splitter_design=splitter_design, guard=guard,
                                 isolation_check=check_fn, arm_frame=arm_frame, cv=cv, progress=log)
            table = prediction_table(request, res["pred"], res["support"], queries, variant=variant, system_key=system_key)
            p = write_csv(table, predictions_path(out_root, variant), float_format="%.10g")
            tables[variant] = p
            results[variant] = res
            log(f"{variant}: wrote {p}")
        rho: dict[str, Any] | None = None
        if DEPLOYED_VARIANT in results:
            rho = residual_correlation(results[DEPLOYED_VARIANT]["calibration_detail"], corpus.frame, corpus.v6)
            write_json(rho_path(out_root), {"schema": SCHEMA, "label": LABEL, **rho, "reading": READINGS["rho"]})
            log(f"rho: {rho['rho']:.3f} from {rho['n_pairs']} pairs ({rho['source']}); flag {rho.get('flag')}")
        codes = code_digest()
        wall = round(time.perf_counter() - t_start, 1)
        record = {"schema": SCHEMA, "label": LABEL, "deployed": DEPLOYED, "arm_alias": ARM, "deployed_variant": DEPLOYED_VARIANT,
                  "configuration": {k: v for k, v in conf.items() if k != "config"},
                  "iterations_used": iterations, "iterations_override": ns.iterations_override, "seed": int(ns.seed),
                  "guard": guard, "inner_design": design.describe() if hasattr(design, "describe") else str(design),
                  "system": {"gen18_system_id": GEN18_SYSTEM_ID, "gen19_system_key": system_key,
                             "templates": {m: str(t[FI.ROW_ID]) for m, t in templates.items()},
                             "tracer_metal_M": TRACER_METAL_M,
                             "tracer_metal_M_reading": "a rounded tracer value; the corpus median of the system's nitrate "
                                                       "lanthanide rows is 8e-5 M (READINGS['query_rows'])",
                             "contact_time_min": ct},
                  "request": {"path": str(req_path), "n_rows": int(len(request)), "metals": metals,
                              "digest": paths.digests(req_path)},
                  "variants": {v: {k: r[k] for k in ("n_train", "n_actinide_rows_removed", "quantiles", "n_calibration",
                                                     "calibration_folds", "splits", "arm_record", "seconds")}
                               | {"n_feature_columns": len(r["feature_columns"]),
                                  "domain_status_counts": dict(Counter(r["support"]["domain_status"])),
                                  "support_score": {"min": float(r["support"]["support_score"].min()),
                                                    "median": float(r["support"]["support_score"].median()),
                                                    "max": float(r["support"]["support_score"].max())},
                                  "mean_logD": {"min": float(r["pred"]["mean_logD"].min()),
                                                "max": float(r["pred"]["mean_logD"].max())},
                                  "table": str(tables[v])} for v, r in results.items()},
                  "wall_clock_seconds": wall,
                  "rho": rho, "readings": READINGS, "code_sha256": codes, "prereg_gate": prereg,
                  "confirmation_half_read": False, "v6_target_rows_scored": 0, "withheld_seed_read": False}
        write_json(fit_record_path(out_root), record)
        if run is not None:
            run.extra.update({"code_sha256": codes, "prereg_gate": prereg, "readings": READINGS, "label": LABEL,
                              "configuration": record["configuration"], "iterations_used": iterations,
                              "wall_clock_seconds": wall,
                              "confirmation_half_read": False, "v6_target_rows_scored": 0, "withheld_seed_read": False,
                              "learned_arm_fitted": True, "exploratory": True})
            run.inputs(req_path, *[Path(r["_path"]) for r in records], paths.ARCHIVE_MASTER)
            run.outputs(*tables.values(), fit_record_path(out_root), *([rho_path(out_root)] if rho else []))
    log(f"done: {list(tables.values())}; label {LABEL}")
    return 0


class _Null:
    """The no-manifest stand-in for :class:`gen19ct.manifest.Run` (``--no-manifest``)."""

    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
