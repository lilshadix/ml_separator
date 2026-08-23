"""Which experiment should be run first on a brand-new extractant? (brief §25)

This is the deployable end of gen8.  A chemist hands it a SMILES and a table of
experiments they *could* run; it hands back the one to run first, what it expects
to see everywhere else, and how much better that choice is than picking at random.
After the first measurement comes back it recommends the second and updates every
prediction.

Nothing in here is invented.  Every rule was measured first and is quoted with the
artefact it came from:

Every number quoted below is **re-derived at run time** from the run artefacts and
written into the output JSON, because those artefacts are regenerated whenever the
study is re-run and a literal in this file would quietly go stale.  The figures in
this docstring are what the artefacts yielded when it was written; the JSON is
authoritative.

**First point — MEDOID / CENTRAL.**  Measure the candidate nearest the centre of
the candidate pool in standardised condition space.  Measured on 143 held-out
ligands x 5 split seeds under protocol P2 (``runs/gen8_architecture/active_acquisition``):
macro MAE 0.649 for ``MEDOID`` and 0.651 for ``CENTRAL`` against 0.723 for
``RANDOM``, a paired-chemotype gain of 0.075 (BCa CI [0.035, 0.130], 101/143
ligands improved, 5/5 seeds).  Zero-shot is 0.995 and the non-deployable oracle
0.484, so one well-chosen measurement closes about two thirds of the gap.  The
geography behind it (``runs/gen8_architecture/cross_series``) is the same story cut
three ways: a low-acid point is 0.244 worse than a mid-acid one (CI [0.175, 0.323]),
an extreme lanthanide 0.095 worse than a central one (CI [0.060, 0.138]), and a
point in the bottom third of the model's own predicted range 0.143 worse than one in
the middle (CI [0.096, 0.197]).  All five seeds agree on all three.

**Second point — FARTHEST_FROM_EXISTING.**  Once the level has been anchored, the
remaining error is *shape*, and fitting a slope needs leverage rather than
typicality.  ``CENTRAL_THEN_SPREAD`` (medoid, then farthest-from-selected) with the
K3 adapter scores 0.558 at k = 2 against 0.598 for ``RANDOM`` and 0.611 for
``CENTRAL`` twice over — a paired gain over random of 0.041 (BCa CI [0.025, 0.058],
98/143 ligands, 5/5 seeds).  Note the reversal: ``FARTHEST_FROM_EXISTING`` is the
*worst* deployable first point and part of the best second one.  Order matters.

Stated plainly, because the earlier version of this file did not: at k = 2 this rule
is **not** the best point estimate in the field.  ``MAX_PREDICTIVE_VARIANCE`` (0.543),
``FARTHEST_FROM_EXISTING`` alone (0.553) and ``MAX_EXPECTED_VARIANCE_REDUCTION``
(0.555) all score below it, though none of the three differs from it by a paired
interval that excludes zero (best of them, +0.014, BCa [-0.002, +0.031]); at k = 3
and k = 5 the ordering reverses again and ``CENTRAL_THEN_SPREAD`` is level or ahead.
The full deployable leaderboard at every k is emitted as
``evidence.acquisition.deployable_policy_leaderboard`` so the choice can be audited
rather than taken on the two comparisons that flatter it.

**Calibration — OFFSET_K1 at k = 1, OFFSET_K3 at k >= 2.**  Both come straight from
:mod:`lanthanide_separation.gen8.adapters` and are not reimplemented here.  At
k = 2 the ridge-shrunk response-coefficient update is worth 0.066 over a plain
offset (CI [0.047, 0.084], 5/5 seeds), rising to 0.139 at k = 5.

**What is deliberately NOT used.**  The model's own uncertainty.  Be precise about
why, because the artefact does not say what an earlier draft of this file claimed:
at k = 1, ``MAX_ENSEMBLE_SD`` (0.707) and ``MIN_ENSEMBLE_SD`` (0.715) are *better*
than ``RANDOM`` (0.723) on the point estimate, while ``MAX_MODEL_DISAGREEMENT``
(0.740) and ``MAX_PREDICTIVE_VARIANCE`` (0.744) are worse.  What justifies ignoring
all four is the paired interval, not the ordering: no one of them separates from
``RANDOM`` (best case +0.016, BCa [-0.011, +0.049]), whereas the deployed rule beats
each of them by an interval that does exclude zero (vs ``MAX_ENSEMBLE_SD``: +0.059,
BCa [+0.029, +0.089]).  The per-candidate ensemble spread is still *reported* — it is
a real diagnostic — but it never enters the ranking, and the predictive intervals are
therefore constant across candidates within a stage rather than pretending a signal
that was tested and did not separate.

**A caveat on the intervals.**  The macro MAEs above weight every ligand equally.
The per-row prediction intervals this script reports are row-weighted and shrink much
less: on the same replay, row-weighted mean |error| goes 1.16 -> 1.00 -> 0.98 for
k = 0, 1, 2 while the ligand macro goes 1.00 -> 0.61 -> 0.56.  Both are true; they
answer different questions, and both are in the JSON.  That replay uses a
full-surface protocol (every candidate selectable, every unmeasured row scored),
which is *not* the acquisition study's P2 and is not offered as reproducing it.  The
actual reproduction check is ``uncertainty.p2_replay``: the same policy objects and
the same adapters, run under P2 with the study's own seed, which must land on
``primary_detail.parquet``'s macro MAEs to within 1e-9 — and does, at k = 0, 1 and 2.

The global model
----------------
The frozen gen7/gen8 arm ``REC_ecfp_plus_recovered`` — an ExtraTrees on
METAL + COND + ECFP + MASSACTION + RECOVERED with missingness indicators — refitted
through :mod:`lanthanide_separation.gen7.contenders`, i.e. the same class the
leaderboard ran, not a lookalike.  If the queried ligand is in the cohort its
entire Tanimoto chemotype is dropped from the training set first, so the demo is
under the same hold-out contract as every number quoted above.

Leakage contract
----------------
* No target of an unmeasured row is read anywhere in the recommendation path.  The
  acquisition policies come from :mod:`lanthanide_separation.gen8.kshot` and are
  handed a :class:`PolicyContext` whose ``truth`` field is ``None``.
* The only targets that ever enter are the values the user reports with
  ``--observed``, which are measurements they made.
* ``--demo-observe-truth`` substitutes the cohort's recorded value for a point the
  policy already chose, purely so the two-stage flow can be exercised without a
  laboratory.  It is recorded in the JSON as ``source: "cohort_truth (demo only)"``
  and it can never influence *which* point is chosen, because the choice is made
  before the value is looked up.
* Feature construction for an unseen ligand never touches ``log_D``.  Ligand
  denticity, when unknown, falls back to the training median rather than to the
  mechanistic proxy: ``mech__denticity_proxy`` correlates with the recorded
  ``DENTATE`` at r = 0.53 (MAE 0.78 over the 152 cohort ligands), which is not good
  enough to be worth the extra moving part.

Usage
-----
::

    .venv/bin/python scripts/gen8_recommend_experiment.py \\
        --ligand "<SMILES>" --candidate-conditions conditions.csv \\
        --out recommended_condition.json

    # ...then, once experiment 1 has been run:
    .venv/bin/python scripts/gen8_recommend_experiment.py \\
        --ligand "<SMILES>" --candidate-conditions conditions.csv \\
        --observed 1.23 --out recommended_condition.json

    # self-contained demonstration on a real held-out cohort ligand:
    .venv/bin/python scripts/gen8_recommend_experiment.py --demo

Candidate-conditions CSV
------------------------
One row per experiment you could run.  Header names are matched case- and
punctuation-insensitively; see :data:`COLUMN_ALIASES`.

===========================  ========  ==================================================
column                       required  meaning
===========================  ========  ==================================================
``metal``                    yes       lanthanide symbol, e.g. ``Nd``
``acid_concentration_M``     yes       aqueous acid molarity
``extractant_concentration_M`` yes     organic-phase extractant molarity
``acid``                     no        acid identity (default ``hno3``)
``diluent``                  no        organic diluent (default ``n_dodecane``)
``temperature_C``            no        default 25
``metal_concentration_mM``   no        left unknown if absent
``contact_time_min``         no        left unknown if absent
``additive``                 no        phase modifier, if any
``additive_concentration_M`` no        modifier molarity
``dentate`` / ``core_cn``    no        ligand denticity / core coordination number
``candidate_id``             no        your label; auto-assigned ``C001…`` if absent
``row_id``                   no        provenance only; used by ``--demo-observe-truth``
===========================  ========  ==================================================

Output JSON schema
------------------
``schema_version``   string, ``"gen8-recommend-1.0"``.
``query``            the ligand, its nearest training chemistry, what was excluded.
``model``            the arm, its blocks, the training rows it saw, its seed.
``stage``            how many measurements are in hand and which adapter that selects.
``recommendation``   the single next experiment: candidate id, conditions, prediction,
                     uncertainty, the rule that fired, and ``expected_improvement``.
``observations``     the measurements supplied so far, in the order they were taken.
``predictions``      one entry per candidate condition: calibrated ``predicted_log_D``,
                     the uncalibrated ``zero_shot_log_D``, ``ensemble_sd``, the
                     stage-appropriate ``uncertainty_68``/``uncertainty_90`` half-widths,
                     and flags for observed / recommended.
``evidence``         the measured tables the policy and the intervals rest on.
``warnings``         every assumption the run had to make, in plain language.
``field_documentation``  one line per field of ``predictions``.

The full JSON is self-describing; ``field_documentation`` travels with the file so a
recipient does not need this docstring.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lanthanide_separation.gen7.contenders import Tabular, extratrees  # noqa: E402
from lanthanide_separation.gen7.harness import (  # noqa: E402
    FOLD_SEED_OFFSET, Fold, FoldContext, MODEL_SEED_BASE, load_cohort,
)
from lanthanide_separation.gen7.recovered import solvent_descriptors  # noqa: E402
from lanthanide_separation.gen8.adapters import AdaptContext, RidgeOffset, ZeroShot  # noqa: E402
from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.kshot import (  # noqa: E402
    POLICY_AXES, PolicyContext, policy_farthest, policy_medoid, stable_hash,
)
from lanthanide_separation.gen8.protocols import _standardised_axes  # noqa: E402
from lanthanide_separation.gen8.report import md_table  # noqa: E402
from lanthanide_separation.levels import (  # noqa: E402
    LEVEL_TARGET_COLUMN, _attach_mass_action,
)

SCHEMA_VERSION = "gen8-recommend-1.0"

#: The frozen global arm.  Blocks and flags copied from ``gen7.suites._finalists``.
ARM_NAME = "REC_ecfp_plus_recovered"
ARM_BLOCKS: tuple[str, ...] = ("METAL", "COND", "ECFP", "MASSACTION", "RECOVERED")

#: ECFP settings that reproduce the dataset's ``ecfp_*`` columns bit-for-bit
#: (verified against the cohort: Morgan radius 2, 2048 bits, exact match).
ECFP_RADIUS = 2
ECFP_BITS = 2048

OOF_PATH = REPO_ROOT / "runs" / "gen7_architecture" / "finalists" / "oof_predictions.parquet"
ACQ_DETAIL = REPO_ROOT / "runs" / "gen8_architecture" / "active_acquisition" / "primary_detail.parquet"
GEOGRAPHY = REPO_ROOT / "runs" / "gen8_architecture" / "cross_series" / "stratum_paired_bootstrap.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "gen8_architecture" / "recommender"

#: Similarity bins for the empirical predictive intervals.  The fold plan blocks on
#: Tanimoto 0.7, so no held-out ligand can exceed ~0.70 and the table simply cannot
#: be measured above it; a query more similar than that is flagged as extrapolated.
SIMILARITY_BINS: tuple[tuple[float, float], ...] = ((0.0, 0.4), (0.4, 0.6), (0.6, 1.01))
CHEMOTYPE_THRESHOLD = 0.7


# --------------------------------------------------------------------------- #
# Candidate-conditions CSV
# --------------------------------------------------------------------------- #

#: canonical name -> accepted header spellings (matched after lower-casing and
#: stripping every non-alphanumeric character).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "candidate_id": ("candidateid", "id", "label", "name", "experiment"),
    "row_id": ("rowid",),
    "metal": ("metal", "metalsymbol", "element", "lanthanide"),
    "acid": ("acid", "acidtype", "acidname", "aqueousacid"),
    "acid_concentration_M": ("acidconcentrationm", "acidconcm", "acidm", "acidconc", "hno3m"),
    "extractant_concentration_M": ("extractantconcentrationm", "extractantconcm", "extractantm",
                                   "ligandconcentrationm", "ligandm", "extractantconc"),
    "diluent": ("diluent", "solvent", "organicphase", "organicdiluent"),
    "temperature_C": ("temperaturec", "tempc", "temperature", "temp"),
    "metal_concentration_mM": ("metalconcentrationmm", "metalconcmm", "metalmm"),
    "contact_time_min": ("contacttimemin", "contacttime"),
    # Deliberately NOT an alias of contact time: in the cohort the two are recorded
    # from different upstream fields and never co-occur on a row (0 of 5,248 rows
    # carry both), so folding them together would invent a value the model was
    # never trained on.
    "shaking_time_min": ("shakingtimemin", "shakingtime"),
    "additive": ("additive", "modifier", "phasemodifier"),
    "additive_concentration_M": ("additiveconcentrationm", "modifierconcentrationm",
                                 "phasemodifierconcentrationm"),
    "dentate": ("dentate", "denticity"),
    "core_cn": ("corecn", "coordinationnumber", "cn"),
}
REQUIRED_COLUMNS: tuple[str, ...] = ("metal", "acid_concentration_M", "extractant_concentration_M")
DEFAULT_ACID = "hno3"
DEFAULT_DILUENT = "n_dodecane"
DEFAULT_TEMPERATURE_C = 25.0
#: A few chemist-friendly spellings for the acid one-hots.
ACID_ALIASES: dict[str, str] = {
    "nitric": "hno3", "nitricacid": "hno3", "hno3": "hno3",
    "hydrochloric": "hcl", "hcl": "hcl",
    "sulfuric": "h2so4", "sulphuric": "h2so4", "h2so4": "h2so4",
    "perchloric": "hclo4", "hclo4": "hclo4",
}


def _slug(text: object) -> str:
    """``"n-Dodecane"`` -> ``"n_dodecane"``: the dataset's own column-name convention."""
    out = re.sub(r"[^0-9a-z]+", "_", str(text).strip().lower())
    return out.strip("_")


def _key(text: object) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(text).strip().lower())


def read_candidate_conditions(path: Path) -> pd.DataFrame:
    """Read and normalise the user's candidate table.  Never reads a target."""
    raw = pd.read_csv(path)
    if LEVEL_TARGET_COLUMN in raw.columns:
        raw = raw.drop(columns=[LEVEL_TARGET_COLUMN])
    lookup = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for name in (canonical,) + aliases:
            lookup[_key(name)] = canonical
    renamed: dict[str, str] = {}
    unknown: list[str] = []
    for column in raw.columns:
        canonical = lookup.get(_key(column))
        if canonical is None:
            unknown.append(str(column))
        else:
            renamed[str(column)] = canonical
    frame = raw.rename(columns=renamed)
    frame = frame[[c for c in frame.columns if c in COLUMN_ALIASES]]
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"candidate-conditions file {path} is missing required column(s) {missing}.\n"
            f"Recognised columns: {sorted(COLUMN_ALIASES)}\n"
            f"Unrecognised headers in your file: {unknown}")
    if not len(frame):
        raise SystemExit(f"candidate-conditions file {path} has no rows")
    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = [f"C{i + 1:03d}" for i in range(len(frame))]
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame["candidate_id"].duplicated().any():
        raise SystemExit("candidate_id values must be unique")
    frame.attrs["unrecognised_columns"] = unknown
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Ligand chemistry
# --------------------------------------------------------------------------- #

def _parse(smiles: str):
    """Parse one SMILES with RDKit's log **scoped**, not globally disabled.

    ``RDLogger.DisableLog`` is process-global and permanent: calling it here would
    silence parse and sanitisation warnings for every other module in the process,
    which is precisely the failure ``tests/test_gen8_mechanism.py`` exists to catch.
    ``rdBase.BlockLogs`` restores the previous state on exit.  We report the failure
    ourselves, with the offending string, which is more use than RDKit's stderr.
    """
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"rdkit could not parse the ligand SMILES: {smiles!r}")
    return mol


def ecfp_bits(smiles: str) -> np.ndarray:
    """The dataset's own fingerprint, recomputed for an arbitrary SMILES."""
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
    return np.asarray(generator.GetFingerprint(_parse(smiles)), dtype=float)


def canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(_parse(smiles))


def tanimoto_to_set(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Tanimoto of one bit vector against a matrix of them."""
    intersection = reference @ query
    union = reference.sum(axis=1) + query.sum() - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float),
                     where=union > 0)


# --------------------------------------------------------------------------- #
# Feature construction for an unseen ligand
# --------------------------------------------------------------------------- #

@dataclass
class CandidateFrame:
    frame: pd.DataFrame
    conditions: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)


def build_candidate_frame(cohort, smiles: str, conditions: pd.DataFrame,
                          *, cohort_ligand_row: pd.Series | None) -> CandidateFrame:
    """Turn (SMILES, candidate conditions) into feature rows the frozen arm accepts.

    Everything here is target-free by construction: fingerprint bits from rdkit,
    metal properties from a symbol lookup, condition one-hots from the user's
    labels, mass-action logarithms through :func:`levels._attach_mass_action` (the
    same function that built the cohort), and solvent physics through
    :func:`gen7.recovered.solvent_descriptors` (the same function that built the
    RECOVERED block).
    """
    frame_cohort = cohort.frame
    notes: list[str] = list(conditions.attrs.get("unrecognised_columns", []) and
                            [f"ignored unrecognised CSV column(s): "
                             f"{conditions.attrs['unrecognised_columns']}"] or [])
    assumptions: dict = {}
    n = len(conditions)
    out = pd.DataFrame(index=range(n))

    # -- ligand identity ----------------------------------------------------- #
    out["extractant"] = smiles
    bits = ecfp_bits(smiles)
    # One concat rather than 2,048 inserts: the same values, without fragmenting the
    # frame into 2,048 single-column blocks.
    out = pd.concat([out, pd.DataFrame(np.tile(bits, (n, 1)),
                                       columns=[f"ecfp_{i}" for i in range(ECFP_BITS)],
                                       index=out.index)], axis=1)

    # -- metal --------------------------------------------------------------- #
    metal_columns = ("Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal")
    metals = frame_cohort.drop_duplicates("metal_symbol").set_index("metal_symbol")[list(metal_columns)]
    symbols = [str(s).strip().capitalize() for s in conditions["metal"]]
    unknown = sorted({s for s in symbols if s not in metals.index})
    if unknown:
        raise SystemExit(f"unknown metal symbol(s) {unknown}; "
                         f"the cohort covers {sorted(metals.index)}")
    out["metal_symbol"] = symbols
    for column in metal_columns:
        out[column] = [float(metals.loc[s, column]) for s in symbols]

    # -- conditions: continuous ---------------------------------------------- #
    def numeric(name: str, default: float | None) -> np.ndarray:
        if name in conditions.columns:
            values = pd.to_numeric(conditions[name], errors="coerce").to_numpy(dtype=float)
        else:
            values = np.full(n, np.nan)
        if default is not None and not np.isfinite(values).all():
            filled = int((~np.isfinite(values)).sum())
            values = np.where(np.isfinite(values), values, default)
            notes.append(f"{name}: {filled} row(s) had no value; assumed {default}")
        return values

    out["cond__acid_concentration_M"] = numeric("acid_concentration_M", None)
    out["cond__extractant_concentration_M"] = numeric("extractant_concentration_M", None)
    out["cond__temperature_C"] = numeric("temperature_C", DEFAULT_TEMPERATURE_C)
    out["cond__metal_concentration_mM"] = numeric("metal_concentration_mM", None)
    out["cond__contact_time_min"] = numeric("contact_time_min", None)
    for column in ("cond__acid_concentration_M", "cond__extractant_concentration_M"):
        if not np.isfinite(out[column].to_numpy(dtype=float)).all():
            raise SystemExit(f"{column} must be given for every candidate row")

    # -- conditions: one-hot families ---------------------------------------- #
    cond_columns = tuple(cohort.blocks["COND"])
    fresh = [c for c in cond_columns if c not in out.columns]
    out = pd.concat([out, pd.DataFrame(0.0, index=out.index, columns=fresh)], axis=1)

    def assign_onehot(prefix: str, labels: Sequence[str], *, fallback: str | None,
                      family: str) -> list[str]:
        known = {c[len(prefix):] for c in cond_columns if c.startswith(prefix)}
        resolved: list[str] = []
        for label in labels:
            if label == "":
                resolved.append("")
                continue
            slug = ACID_ALIASES.get(_key(label), _slug(label)) if family == "acid" else _slug(label)
            if slug not in known:
                if fallback is not None and fallback in known:
                    notes.append(f"{family} {label!r} is not one of the cohort's "
                                 f"{len(known)} recorded {family}s; encoded as {fallback!r}")
                    slug = fallback
                else:
                    raise SystemExit(
                        f"{family} {label!r} is not one of the cohort's recorded {family}s: "
                        f"{sorted(known)}")
            resolved.append(slug)
        for i, slug in enumerate(resolved):
            if slug:
                out.loc[i, f"{prefix}{slug}"] = 1.0
        return resolved

    acids = [str(v) if pd.notna(v) and str(v).strip() else DEFAULT_ACID
             for v in (conditions["acid"] if "acid" in conditions.columns
                       else pd.Series([DEFAULT_ACID] * n))]
    if "acid" not in conditions.columns:
        assumptions["acid"] = DEFAULT_ACID
        notes.append(f"no acid column: assumed {DEFAULT_ACID} for every candidate")
    assign_onehot("cond__acid__", acids, fallback=None, family="acid")

    diluents = [str(v) if pd.notna(v) and str(v).strip() else DEFAULT_DILUENT
                for v in (conditions["diluent"] if "diluent" in conditions.columns
                          else pd.Series([DEFAULT_DILUENT] * n))]
    if "diluent" not in conditions.columns:
        assumptions["diluent"] = DEFAULT_DILUENT
        notes.append(f"no diluent column: assumed {DEFAULT_DILUENT} for every candidate")
    assign_onehot("cond__diluent__", diluents, fallback="other", family="diluent")

    additives = [str(v) if pd.notna(v) and str(v).strip() else ""
                 for v in (conditions["additive"] if "additive" in conditions.columns
                           else pd.Series([""] * n))]
    assign_onehot("cond__additive__", additives, fallback=None, family="additive")

    # -- ligand scalars the MASSACTION products need -------------------------- #
    # Neither is a pure ligand property.  ``coreCN`` is the core coordination number
    # and in this cohort it is set mostly by the *metal* — 9 across La–Gd, 8 from Tb
    # on, the standard lanthanide-contraction break — with a residual ligand
    # dependence (84 of 152 extractants carry both values).  ``DENTATE`` is nearly a
    # ligand constant (10 of 152 vary).  They are therefore looked up per
    # (extractant, metal) when the ligand is known, and per metal otherwise.
    ligand_metal = frame_cohort.groupby(["extractant", "metal_symbol"])[["DENTATE", "coreCN"]].median()
    by_metal = frame_cohort.groupby("metal_symbol")[["DENTATE", "coreCN"]].median()
    for name, csv_name in (("DENTATE", "dentate"), ("coreCN", "core_cn")):
        if csv_name in conditions.columns and pd.to_numeric(
                conditions[csv_name], errors="coerce").notna().any():
            out[name] = pd.to_numeric(conditions[csv_name], errors="coerce").to_numpy(dtype=float)
            assumptions[name] = "from candidate-conditions CSV"
        elif cohort_ligand_row is not None:
            values = []
            for symbol in symbols:
                key = (str(cohort_ligand_row["extractant"]), symbol)
                values.append(float(ligand_metal.loc[key, name]) if key in ligand_metal.index
                              else float(by_metal.loc[symbol, name]))
            out[name] = values
            assumptions[name] = "cohort record for this ligand, per metal"
        else:
            values = [float(by_metal.loc[symbol, name]) for symbol in symbols]
            out[name] = values
            assumptions[name] = ("training median for the metal — unknown for an unseen ligand"
                                 if name == "coreCN" else
                                 f"training median ({float(by_metal[name].median())}) — "
                                 f"unknown for an unseen ligand")
            notes.append(f"{name} unknown for this ligand; used the per-metal training median. "
                         f"It enters only the massact__logL_x_{name} product. Supply a "
                         f"{csv_name!r} column to override.")

    # -- mass action: the cohort's own function, not a copy of it ------------- #
    out = _attach_mass_action(out)

    # -- recovered variables -------------------------------------------------- #
    solvent = pd.DataFrame([solvent_descriptors(name) for name in diluents], index=out.index)
    for column in solvent.columns:
        out[column] = solvent[column].to_numpy(dtype=float)
    unparsed = int((solvent["rec__solvent_parsed"] == 0.0).sum())
    if unparsed:
        notes.append(f"{unparsed} candidate row(s) have a diluent the solvent-physics parser "
                     f"does not know; their rec__solvent_* features are left unknown")
    # ``rec__has_phase_modifier`` is reconstructible exactly: it agrees with the
    # COND additive one-hots on 5,248 of 5,248 cohort rows.
    out["rec__has_phase_modifier"] = np.array([1.0 if a else 0.0 for a in additives])
    out["rec__phase_modifier_concentration_M"] = numeric("additive_concentration_M", None)
    shaking = numeric("shaking_time_min", None)
    out["rec__shaking_time_min"] = shaking
    # ``rec__has_shaking_time`` is NOT a provenance unknown: in the cohort it equals
    # ``rec__shaking_time_min.notna()`` on all 5,248 rows and is never missing, so it
    # is fully determined by what the user supplied.  Leaving it NaN made every
    # prospective row disagree with the trained encoding (it happened to impute back
    # to 0.0 because 4,225 of 5,248 rows are 0, but that is luck, not correctness).
    out["rec__has_shaking_time"] = np.where(np.isfinite(shaking), 1.0, 0.0)
    # The remaining RECOVERED columns are provenance artefacts of the literature
    # source (did the paper's ligand name match the structure, how many names does
    # this structure have, is there an aqueous complexant implied by the name).  A
    # prospective experiment has none of them, so they are left unknown and the
    # arm's imputer substitutes the training median.  This is a real, unavoidable
    # train/deploy shift and is reported rather than papered over.
    absent_prospectively = ("rec__acid_concentration_organic_M", "rec__name_mismatch",
                            "rec__n_names_for_structure", "rec__aqueous_complexant")
    out = pd.concat([out, pd.DataFrame(np.nan, index=out.index,
                                       columns=list(absent_prospectively))], axis=1)
    notes.append("4 of the 23 RECOVERED columns are literature-provenance artefacts "
                 f"({', '.join(absent_prospectively)}) and cannot exist for a prospective "
                 "experiment; the arm's imputer fills them with the training median. "
                 "Three of the four are recorded on every cohort row, so this is a real "
                 "train/deploy shift; rec__acid_concentration_organic_M is recorded on "
                 "only 36 of 5,248 and is usually missing in training too")

    # -- everything the arm asks for and we could not build ------------------- #
    feature_columns = cohort.block_columns(ARM_BLOCKS)
    unbuilt = [c for c in feature_columns if c not in out.columns]
    out = pd.concat([out, pd.DataFrame(np.nan, index=out.index, columns=unbuilt)], axis=1)
    out = out.copy()   # de-fragment once, at the end
    if unbuilt:
        notes.append(f"{len(unbuilt)} arm feature column(s) could not be constructed and are "
                     f"left unknown (imputed): {unbuilt[:6]}{'…' if len(unbuilt) > 6 else ''}")

    out["candidate_id"] = conditions["candidate_id"].to_numpy()
    return CandidateFrame(frame=out, conditions=conditions, warnings=notes, assumptions=assumptions)


# --------------------------------------------------------------------------- #
# The frozen global model
# --------------------------------------------------------------------------- #

class _SpreadCapturingForest:
    """The frozen ExtraTrees, wrapped so the across-tree spread can be read out.

    ``Tabular`` builds its learner through a factory and never exposes it, so this
    thin delegate is inserted in place of the estimator.  ``predict`` still returns
    the forest's own ``predict`` — the mean over trees, bit-for-bit — and records
    the per-row standard deviation as a side effect.  Nothing about the fit changes.
    """

    def __init__(self, inner):
        self.inner = inner
        self.spread_: np.ndarray | None = None

    def fit(self, x, y, sample_weight=None):
        self.inner.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        per_tree = np.stack([tree.predict(x) for tree in self.inner.estimators_])
        self.spread_ = per_tree.std(axis=0)
        return self.inner.predict(x)


@dataclass
class GlobalModel:
    prediction: np.ndarray
    ensemble_sd: np.ndarray
    n_train_rows: int
    n_train_ligands: int
    n_train_chemotypes: int
    excluded_chemotype: str | None
    model_seed: int
    n_estimators: int


def fit_and_predict(cohort, candidates: pd.DataFrame, *, excluded_chemotype: str | None,
                    n_estimators: int) -> GlobalModel:
    """Refit ``REC_ecfp_plus_recovered`` on the cohort minus the query chemotype."""
    frame = cohort.frame
    mask = np.ones(len(frame), dtype=bool)
    if excluded_chemotype is not None:
        mask = frame["tanimoto_cluster"].astype(str).to_numpy() != str(excluded_chemotype)
    train = frame.loc[mask]
    if not len(train):
        raise SystemExit("excluding the query chemotype left no training rows")

    store: list[_SpreadCapturingForest] = []

    def factory(seed: int):
        forest = _SpreadCapturingForest(extratrees(seed, n=n_estimators))
        store.append(forest)
        return forest

    arm = Tabular(ARM_BLOCKS, factory, ARM_NAME, add_indicator=True)
    # There is no fold here — this is a deployment refit — so the seed is the gen7
    # fold-0 model seed, i.e. the frozen MODEL_SEED_BASE = 42 put through the same
    # formula every gen5/6/7/8 run used.  Recorded in the JSON so a rerun is exact.
    model_seed = int(MODEL_SEED_BASE + 0 + FOLD_SEED_OFFSET)
    fold = Fold(seed=0, fold=0, train_index=np.flatnonzero(mask),
                test_index=np.arange(len(candidates)), held_out_chemotypes=
                (str(excluded_chemotype),) if excluded_chemotype else ())
    context = FoldContext(fold=fold, cohort=cohort, feature_columns=(), model_seed=model_seed)
    prediction = arm.fit_predict(train, train[LEVEL_TARGET_COLUMN].to_numpy(dtype=float),
                                 candidates, context)
    spread = store[0].spread_ if store and store[0].spread_ is not None \
        else np.full(len(candidates), np.nan)
    return GlobalModel(
        prediction=np.asarray(prediction, dtype=float),
        ensemble_sd=np.asarray(spread, dtype=float),
        n_train_rows=int(len(train)),
        n_train_ligands=int(train["extractant"].nunique()),
        n_train_chemotypes=int(train["tanimoto_cluster"].nunique()),
        excluded_chemotype=str(excluded_chemotype) if excluded_chemotype else None,
        model_seed=model_seed, n_estimators=int(n_estimators))


# --------------------------------------------------------------------------- #
# Measured evidence: the policy gains and the predictive intervals
# --------------------------------------------------------------------------- #

#: The adapter the recommender deploys at each k, and the policy it deploys.
RECOMMENDED_POLICY = "CENTRAL_THEN_SPREAD"


def _adapter_for(k: int) -> str:
    return "OFFSET_K1" if int(k) <= 1 else "OFFSET_K3"


def _macro(detail: pd.DataFrame, policy: str, adapter: str, k: int) -> float:
    subset = detail[(detail["policy"] == policy) & (detail["adapter"] == adapter)
                    & (detail["k"] == k)]
    if not len(subset):
        return float("nan")
    per_ligand = subset.groupby(["split_seed", "extractant"])["mae"].mean()
    return float(per_ligand.groupby("split_seed").mean().mean())


def _paired(detail: pd.DataFrame, arms: dict[str, tuple[str, str, int]],
            comparisons: dict[str, tuple[str, str]], *, replicates: int) -> dict:
    """Paired chemotype bootstrap over named (policy, adapter, k) arms of one table.

    Every arm is drawn from the same run, so the ligands, repeats, candidate pools
    and evaluation rows are identical on both sides of every comparison by
    construction; :func:`paired_chemotype_bootstrap` then resamples chemotypes.
    """
    frames = []
    for name, (policy, adapter, k) in arms.items():
        block = detail[(detail["policy"] == policy) & (detail["adapter"] == adapter)
                       & (detail["k"] == int(k))].copy()
        if len(block):
            block["arm"] = name
            frames.append(block)
    if not frames:
        return {}
    table = paired_chemotype_bootstrap(pd.concat(frames, ignore_index=True), comparisons,
                                       replicates=replicates)
    counts = {"n_units", "units_improved", "seeds_positive", "n_seeds"}
    return {row["comparison"]: {
        key: (int(value) if key in counts else
              float(value) if isinstance(value, (int, float, np.floating, np.integer))
              else value)
        for key, value in row.items() if key != "comparison"}
        for _, row in table.iterrows()}


def _leaderboard(detail: pd.DataFrame, k: int) -> list[dict]:
    """Every *deployable* policy at this k under the adapter the recommender uses.

    The recommended policy was chosen from this field of 14, so the field travels
    with the recommendation: a reader can see where the deployed rule actually sits
    rather than only the two arms it is quoted against.
    """
    adapter = _adapter_for(k)
    deployable = detail[detail["deployable"].astype(bool) & (detail["policy"] != "NONE")]
    names = sorted(deployable["policy"].unique())
    scored = [(name, _macro(detail, name, adapter, k)) for name in names]
    scored = [(name, value) for name, value in scored if np.isfinite(value)]
    scored.sort(key=lambda item: item[1])
    return [{"rank": i + 1, "policy": name, "macro_mae": float(value),
             "is_recommended": bool(name == RECOMMENDED_POLICY)}
            for i, (name, value) in enumerate(scored)]


def build_policy_evidence(*, replicates: int = 2000) -> dict:
    """Re-measure the deltas the recommendation quotes, from the gen8 run artefact.

    Nothing is hard-coded: the numbers in this file's docstring are what this
    function returns on ``runs/gen8_architecture/active_acquisition/primary_detail.parquet``.
    """
    if not ACQ_DETAIL.exists():
        return {"available": False, "reason": f"{ACQ_DETAIL} not found"}
    columns = ["model", "split_seed", "extractant", "tanimoto_cluster", "policy", "adapter",
               "k", "mae", "deployable"]
    detail = pd.read_parquet(ACQ_DETAIL, columns=columns)
    detail = detail[detail["model"] == ARM_NAME]
    uncertainty_names = ("MAX_ENSEMBLE_SD", "MIN_ENSEMBLE_SD",
                         "MAX_MODEL_DISAGREEMENT", "MAX_PREDICTIVE_VARIANCE")
    uncertainty_arms = {"RANDOM": ("RANDOM", "OFFSET_K1", 1),
                        "RECOMMENDED": (RECOMMENDED_POLICY, "OFFSET_K1", 1)}
    uncertainty_arms.update({name: (name, "OFFSET_K1", 1) for name in uncertainty_names})
    uncertainty_paired = _paired(
        detail, uncertainty_arms,
        {**{f"{name}_vs_RANDOM": ("RANDOM", name) for name in uncertainty_names},
         **{f"RECOMMENDED_vs_{name}": (name, "RECOMMENDED") for name in uncertainty_names}},
        replicates=replicates)

    out: dict = {"available": True, "source": str(ACQ_DETAIL.relative_to(REPO_ROOT)),
                 "model": ARM_NAME,
                 # The policies this recommender deliberately does NOT use, measured
                 # from the same table so the refusal is auditable rather than asserted.
                 # Point estimates alone are NOT the justification: two of the four beat
                 # RANDOM on the point estimate, and the paired intervals below are what
                 # show that neither difference excludes zero while the gap to the
                 # deployed rule does.
                 "uncertainty_policies_at_k1": {
                     name: _macro(detail, name, "OFFSET_K1", 1) for name in
                     ("RANDOM", *uncertainty_names)},
                 "uncertainty_policies_at_k1_paired": uncertainty_paired,
                 "protocol": "P2 (disjoint candidate pool / evaluation set), 143 held-out "
                             "ligands x 5 split seeds, macro = one ligand one vote",
                 "zero_shot": _macro(detail, "NONE", "ZERO_SHOT_REF", 0),
                 "deployable_policy_leaderboard": {},
                 "by_k": {}}
    for k in (1, 2, 3, 5):
        adapter = _adapter_for(k)
        board = _leaderboard(detail, k)
        out["deployable_policy_leaderboard"][str(k)] = board
        # Where the deployed rule sits in the field it was chosen from, and what the
        # best alternative to it is.  Quoting only the arms it beats is what turns a
        # measured comparison into an advertisement.
        rank = next((row["rank"] for row in board if row["is_recommended"]), None)
        recommended_macro = _macro(detail, RECOMMENDED_POLICY, adapter, k)
        alternatives = [row for row in board if not row["is_recommended"]]
        # At k = 1 the composite rule *is* ``MEDOID`` — its first pick is that policy —
        # so MEDOID scores identically and is not an alternative in any useful sense.
        # Name the ties and compare against the best genuinely different rule.
        tied = [row["policy"] for row in alternatives
                if abs(row["macro_mae"] - recommended_macro) < 1e-12]
        distinct = [row for row in alternatives
                    if abs(row["macro_mae"] - recommended_macro) >= 1e-12]
        best_alternative = distinct[0]["policy"] if distinct else None
        entry = {
            "adapter": adapter,
            "policy": "MEDOID then FARTHEST_FROM_EXISTING (CENTRAL_THEN_SPREAD)",
            "recommended_macro_mae": recommended_macro,
            "random_macro_mae": _macro(detail, "RANDOM", adapter, k),
            "central_only_macro_mae": _macro(detail, "CENTRAL", adapter, k),
            "recommended_macro_mae_offset_only": _macro(detail, RECOMMENDED_POLICY,
                                                        "OFFSET_K1", k),
            "recommended_rank_among_deployable_policies": rank,
            "n_deployable_policies": len(board),
            "policies_scoring_identically": tied,
            "best_alternative_policy": best_alternative,
            "best_alternative_macro_mae": (distinct[0]["macro_mae"] if distinct
                                           else float("nan")),
        }
        # The oracle in this study is ORACLE[OFFSET_K1]: it chooses points to minimise
        # the *OFFSET_K1* error.  Scoring those choices under OFFSET_K3 is a different
        # quantity and is not a lower bound for it — at k = 5 it is actually worse than
        # the deployable policy.  Report both and say which one bounds what.
        oracle_same = _macro(detail, "ORACLE[OFFSET_K1]", adapter, k)
        oracle_own = _macro(detail, "ORACLE[OFFSET_K1]", "OFFSET_K1", k)
        entry.update({
            "oracle_policy": "ORACLE[OFFSET_K1] — points chosen with the target visible, "
                             "to minimise the OFFSET_K1 error",
            "oracle_macro_mae": oracle_same,
            "oracle_macro_mae_under_its_own_adapter": oracle_own,
            "oracle_is_a_valid_lower_bound": bool(
                np.isfinite(oracle_same) and np.isfinite(entry["recommended_macro_mae"])
                and oracle_same <= entry["recommended_macro_mae"]),
            "oracle_caveat": ("valid: this oracle's choices, scored under the adapter the "
                              "recommender deploys, beat the deployed policy"
                              if np.isfinite(oracle_same)
                              and oracle_same <= entry["recommended_macro_mae"] else
                              "NOT a lower bound at this k: the oracle selected its points "
                              f"to minimise OFFSET_K1 error and scores {oracle_same:.4f} "
                              f"under {adapter}, which the deployable policy already beats. "
                              "Treat it as a reference arm, not a ceiling."),
        })
        arms = {"RECOMMENDED": (RECOMMENDED_POLICY, adapter, k),
                "RANDOM": ("RANDOM", adapter, k)}
        comparisons = {"recommended_vs_random": ("RANDOM", "RECOMMENDED")}
        if k >= 2:
            arms["RECOMMENDED_OFFSET_ONLY"] = (RECOMMENDED_POLICY, "OFFSET_K1", k)
            comparisons["K3_vs_K1_adapter"] = ("RECOMMENDED_OFFSET_ONLY", "RECOMMENDED")
        if best_alternative:
            arms["BEST_ALTERNATIVE"] = (best_alternative, adapter, k)
            comparisons["recommended_vs_best_alternative"] = ("BEST_ALTERNATIVE", "RECOMMENDED")
        paired = _paired(detail, arms, comparisons, replicates=replicates)
        if paired:
            entry["paired"] = paired
        out["by_k"][str(k)] = entry
    out["policy_selection_note"] = (
        "CENTRAL_THEN_SPREAD was selected from the "
        f"{len(out['deployable_policy_leaderboard'].get('1', []))} deployable policies in "
        "this table.  Its gain over RANDOM at k = 1 is large enough to survive that "
        "multiplicity; at k = 2 it is NOT the best point estimate in the field (see "
        "deployable_policy_leaderboard and paired.recommended_vs_best_alternative), and "
        "it is deployed at every k because it is the only rule whose k = 1 behaviour is "
        "measured to be best and whose later behaviour is never measurably worse.")
    if GEOGRAPHY.exists():
        geo = pd.read_csv(GEOGRAPHY)
        out["geography"] = [
            {"comparison": r["comparison"], "cost_of_the_worse_stratum": float(r["point"]),
             "bca_low": float(r["bca_low"]), "bca_high": float(r["bca_high"]),
             "n_ligands": int(r["n_units"]), "ligands_improved": int(r["units_improved"]),
             "seeds_positive": int(r["seeds_positive"]), "n_seeds": int(r["n_seeds"])}
            for _, r in geo.iterrows()]
        out["geography_source"] = str(GEOGRAPHY.relative_to(REPO_ROOT))
    return out


def p2_replay_of_the_deployed_policy() -> dict:
    """Replay the deployed rule under the acquisition study's own P2 protocol.

    This is the check that the deployed code path *is* the measured one.  It runs
    the same policy functions and the same adapters this script deploys, over the
    same out-of-fold predictions, under the same pool/evaluation split — and it must
    land on ``primary_detail.parquet``'s own macro MAEs.  Not "approximately": the
    split is a pure function of (seed, repeat, ligand, n_rows), so agreement should
    be to floating-point noise.  Any drift here means the recommender has stopped
    deploying the policy the numbers were measured on.

    Every constant it needs is read from the run artefact or from the signature of
    :func:`evaluate_fewshot` rather than written here, so re-running the study with
    different settings cannot silently invalidate the comparison.
    """
    import inspect

    from lanthanide_separation.gen8.evaluate import evaluate_fewshot
    from lanthanide_separation.gen8.protocols import make_p2_split

    cohort_path = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
    if not OOF_PATH.exists() or not cohort_path.exists():
        return {"available": False, "reason": "out-of-fold predictions or cohort cache missing"}
    defaults = inspect.signature(evaluate_fewshot).parameters
    seed = int(defaults["seed"].default)
    repeats = int(defaults["repeats"].default)
    min_rows = int(defaults["min_rows"].default)
    run_config = ACQ_DETAIL.parent / "primary_run.json"
    if run_config.exists():
        recorded = json.loads(run_config.read_text())
        repeats = int(recorded.get("repeats", repeats))
        min_rows = int(recorded.get("min_rows", min_rows))

    oof = pd.read_parquet(OOF_PATH)
    oof = oof[oof["model"] == ARM_NAME].copy()
    cohort_frame = pd.read_parquet(cohort_path)
    feature_columns = [c for c in cohort_frame.columns if c != LEVEL_TARGET_COLUMN]
    merged = oof.merge(cohort_frame[feature_columns], on="row_id", how="left",
                       validate="many_to_one", suffixes=("", "__cohort"))
    assert len(merged) == len(oof), "cohort join changed the row count"

    records: list[dict] = []
    for (split_seed, fold), fold_block in merged.groupby(["split_seed", "fold"], sort=True):
        for ligand, block in fold_block.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            n = len(block)
            if n < min_rows:
                continue
            truth = block[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
            prediction = block["prediction"].to_numpy(dtype=float)
            axes = _standardised_axes(block)
            for repeat in range(repeats):
                rng = np.random.default_rng((seed, repeat, stable_hash(str(ligand))))
                split = make_p2_split(n, rng)
                if len(split.pool) < 1 or len(split.evaluation) < 2:
                    continue
                context = PolicyContext(
                    block=block, prediction=prediction, pool=split.pool,
                    evaluation=split.evaluation, axes=axes,
                    uncertainty=np.full(n, np.nan), disagreement=np.full(n, np.nan),
                    rng=np.random.default_rng((seed, repeat, stable_hash(RECOMMENDED_POLICY))),
                    truth=None)
                selected: list[int] = []
                for _ in range(min(2, len(split.pool))):
                    selected.append(policy_medoid(context, selected) if not selected
                                    else policy_farthest(context, selected))
                record = {"split_seed": int(split_seed), "extractant": str(ligand),
                          "k0": float(np.abs(prediction[split.evaluation]
                                             - truth[split.evaluation]).mean())}
                for k, mode in ((1, "K1"), (2, "K3")):
                    if len(selected) < k:
                        continue
                    taken = np.asarray(selected[:k], dtype=int)
                    adjusted = RidgeOffset(mode=mode).predict(
                        block, prediction, taken, truth[taken],
                        AdaptContext(int(split_seed), int(fold), str(ligand)))
                    record[f"k{k}"] = float(np.abs(adjusted[split.evaluation]
                                                   - truth[split.evaluation]).mean())
                records.append(record)
    if not records:
        return {"available": False, "reason": "no ligand met the protocol's row minimum"}
    table = pd.DataFrame(records)
    replay = {}
    for k in (0, 1, 2):
        per_ligand = table.groupby(["split_seed", "extractant"])[f"k{k}"].mean()
        replay[f"k{k}"] = float(per_ligand.groupby("split_seed").mean().mean())
    return {"available": True, "protocol": "P2, replayed from the frozen out-of-fold "
                                           "predictions with this script's own policy and "
                                           "adapter objects",
            "seed": seed, "repeats": repeats, "min_rows": min_rows,
            "n_ligands": int(table["extractant"].nunique()),
            "macro_mae": replay}


def build_uncertainty_calibration() -> dict:
    """Empirical |error| quantiles on genuinely held-out chemotypes, before and after
    the first one or two measurements — the honest width of a prediction interval.

    Computed from the frozen arm's out-of-fold predictions by replaying the *same*
    policy and the *same* adapters this script deploys: medoid first, farthest
    second, ``OFFSET_K1`` then ``OFFSET_K3``.  Row-weighted, because a prediction
    interval is a statement about a row; the headline macro MAEs elsewhere are
    ligand-weighted and are a different quantity.
    """
    if not OOF_PATH.exists():
        return {"available": False, "reason": f"{OOF_PATH} not found"}
    oof = pd.read_parquet(OOF_PATH, columns=["row_id", "extractant", "log_D", "prediction",
                                             "split_seed", "nn_train_tanimoto", "model"])
    oof = oof[oof["model"] == ARM_NAME].copy()
    axes_columns = [c for c in POLICY_AXES]
    cohort_axes = pd.read_parquet(REPO_ROOT / "runs" / "gen7_architecture" / "cache" /
                                  "cohort.parquet", columns=["row_id"] + axes_columns)
    oof = oof.merge(cohort_axes, on="row_id", how="left", validate="many_to_one")

    records: list[pd.DataFrame] = []
    macro: list[dict] = []
    for (seed, _), block in oof.groupby(["split_seed", "extractant"], sort=False):
        block = block.reset_index(drop=True)
        n = len(block)
        if n < 4:
            continue
        truth = block["log_D"].to_numpy(dtype=float)
        prediction = block["prediction"].to_numpy(dtype=float)
        axes = _standardised_axes(block)
        context = PolicyContext(block=block, prediction=prediction, pool=np.arange(n),
                                evaluation=np.arange(n), axes=axes,
                                uncertainty=np.full(n, np.nan), disagreement=np.full(n, np.nan),
                                rng=np.random.default_rng(0), truth=None)
        selected: list[int] = []
        errors = {0: np.abs(truth - prediction)}
        for step, adapter in ((0, RidgeOffset(mode="K1")), (1, RidgeOffset(mode="K3"))):
            pick = policy_medoid(context, selected) if not selected \
                else policy_farthest(context, selected)
            selected.append(int(pick))
            adjusted = adapter.predict(block, prediction, np.asarray(selected, dtype=int),
                                       truth[np.asarray(selected, dtype=int)],
                                       AdaptContext(0, 0, str(block["extractant"].iloc[0])))
            error = np.abs(truth - adjusted)
            error[np.asarray(selected, dtype=int)] = np.nan   # a measured row is not predicted
            errors[step + 1] = error
        records.append(pd.DataFrame({
            "nn": block["nn_train_tanimoto"].to_numpy(dtype=float),
            "k0": errors[0], "k1": errors[1], "k2": errors[2]}))
        macro.append({"split_seed": seed,
                      **{f"k{k}": float(np.nanmean(errors[k])) for k in (0, 1, 2)}})
    table = pd.concat(records, ignore_index=True)
    # Scored one-ligand-one-vote, this full-surface replay is *not* the acquisition
    # study's protocol and must not be presented as reproducing it: here the whole
    # candidate grid is selectable and every unmeasured row is scored, whereas P2
    # selects from a capped random half and scores the disjoint other half.  The two
    # differ in both directions (better at k = 1, marginally worse at k = 2), so no
    # single-cause explanation of the gap is offered.  The real reproduction check —
    # same protocol, same seeds, exact agreement demanded — is p2_replay below.
    macro_frame = pd.DataFrame(macro)
    cross_check = macro_frame.groupby("split_seed")[["k0", "k1", "k2"]].mean().mean().to_dict()
    p2_replay = p2_replay_of_the_deployed_policy()
    reference = {}
    if ACQ_DETAIL.exists():
        detail = pd.read_parquet(ACQ_DETAIL, columns=["model", "split_seed", "extractant",
                                                      "policy", "adapter", "k", "mae"])
        detail = detail[detail["model"] == ARM_NAME]
        reference = {"k0": _macro(detail, "NONE", "ZERO_SHOT_REF", 0),
                     "k1": _macro(detail, RECOMMENDED_POLICY, "OFFSET_K1", 1),
                     "k2": _macro(detail, RECOMMENDED_POLICY, "OFFSET_K3", 2)}
    if reference and p2_replay.get("available"):
        deltas = {k: abs(p2_replay["macro_mae"][k] - reference[k]) for k in ("k0", "k1", "k2")}
        p2_replay["acquisition_artefact_macro_mae"] = reference
        p2_replay["max_abs_difference"] = float(max(deltas.values()))
        p2_replay["reproduces_the_acquisition_artefact"] = bool(max(deltas.values()) < 1e-9)

    def quantiles(values: pd.Series) -> dict:
        clean = values.dropna()
        if clean.empty:
            return {"n": 0}
        return {"n": int(clean.size), "median": float(clean.median()),
                "q68": float(clean.quantile(0.68)), "q90": float(clean.quantile(0.90)),
                "mean": float(clean.mean())}

    bins = []
    # NOTE on the bin edges: 0.4 carries over from gen5's near-neighbour result and
    # 0.7 is the fold plan's own chemotype threshold, but 0.6 is a round number chosen
    # here to keep the top bin populated.  It is a presentation choice, not a fitted
    # one — no comparison in this file depends on it — but it was not pre-registered,
    # so read the interval widths as descriptive rather than as a tested quantity.
    for low, high in SIMILARITY_BINS:
        subset = table[(table["nn"] >= low) & (table["nn"] < high)]
        bins.append({"nn_low": low, "nn_high": high,
                     **{f"k{k}": quantiles(subset[f"k{k}"]) for k in (0, 1, 2)}})
    return {"available": True, "model": ARM_NAME,
            "source": str(OOF_PATH.relative_to(REPO_ROOT)),
            "weighting": "row-weighted absolute error on held-out chemotypes",
            "weighting_note": "these are per-ROW intervals and they shrink far less with k "
                              "than the headline macro MAE does; the row-weighted and "
                              "ligand-macro means of this same replay are both reported "
                              "below.  The macro weights every ligand equally, so it is "
                              "carried by the small-surface ligands where one measurement "
                              "covers most of the surface; one row of a long titration series "
                              "benefits much less.  Quote the macro when comparing methods, "
                              "the row interval when quoting a prediction.",
            "row_weighted_mean_abs_error": {f"k{k}": quantiles(table[f"k{k}"]).get("mean")
                                            for k in (0, 1, 2)},
            "ligand_macro_of_this_replay": {k: float(v) for k, v in cross_check.items()},
            "ligand_macro_of_this_replay_note":
                "full-surface protocol (every candidate selectable, every unmeasured row "
                "scored) — a different protocol from the acquisition study's P2, so a gap "
                "against it is expected and is not evidence of drift.  The drift check is "
                "p2_replay.",
            "p2_replay": p2_replay,
            "policy_replayed": "MEDOID then FARTHEST_FROM_EXISTING; OFFSET_K1 then OFFSET_K3",
            "max_measurable_similarity": float(table["nn"].max()),
            "bins_note": "similarity bin edges are a presentation choice, not a fitted one: "
                         "0.4 carries over from gen5's near-neighbour result and 0.7 is the "
                         "fold plan's chemotype threshold, but 0.6 was picked here to keep "
                         "the top bin populated.  No comparison depends on it; read the "
                         "widths as descriptive.  The top bin is labelled up to 1.01 but "
                         "contains no row above max_measurable_similarity.",
            "bins": bins,
            "overall": {f"k{k}": quantiles(table[f"k{k}"]) for k in (0, 1, 2)}}


def cached_json(path: Path, builder: Callable[[], dict], *, refresh: bool) -> dict:
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    payload = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    return payload


def interval_for(calibration: dict, similarity: float, k: int) -> dict:
    """Half-widths of the 68 % and 90 % prediction interval for this query and stage."""
    if not calibration.get("available"):
        return {"available": False}
    key = f"k{min(int(k), 2)}"
    chosen = calibration["bins"][-1]
    for entry in calibration["bins"]:
        if entry["nn_low"] <= similarity < entry["nn_high"]:
            chosen = entry
            break
    stats = chosen.get(key, {})
    return {"available": bool(stats.get("n")),
            "q68": stats.get("q68"), "q90": stats.get("q90"), "median": stats.get("median"),
            "n_reference_rows": stats.get("n"),
            "similarity_bin": [chosen["nn_low"], chosen["nn_high"]],
            "extrapolated_similarity": bool(
                similarity > calibration.get("max_measurable_similarity", 1.0)),
            "extrapolated_k": bool(int(k) > 2),
            "basis": "empirical quantiles of |log D error| over held-out chemotypes; "
                     "constant across candidates because gen8 measured the model's "
                     "per-row uncertainty to carry no usable ranking signal"}


# --------------------------------------------------------------------------- #
# The recommendation itself
# --------------------------------------------------------------------------- #

FIRST_RULE = "MEDOID"
LATER_RULE = "FARTHEST_FROM_EXISTING"


def _rule_for(step: int) -> tuple[str, str]:
    if step == 0:
        return FIRST_RULE, (
            "measure the candidate closest to the centre of the candidate pool in "
            "standardised condition space (log acid, log extractant, lanthanide index, "
            "temperature).  One measurement can only move the ligand's overall level, so "
            "it should be taken where the level is most representative — not at an extreme.")
    return LATER_RULE, (
        "the level is already anchored by the measurement(s) in hand, so the remaining "
        "error is shape; measure the candidate farthest from everything measured so far, "
        "which is what buys leverage for the acid and extractant response coefficients.")


def run_policy(block: pd.DataFrame, prediction: np.ndarray, ensemble_sd: np.ndarray,
               *, explicit: list[tuple[int, float, str]], queued: list[float],
               truth_supplier: Callable[[int], float] | None, n_truth: int) -> dict:
    """Replay the deterministic sequential design and stop at the next unmeasured point.

    ``explicit`` are (index, value, source) triples the user pinned to a named
    candidate; they are entered first, exactly as taken.  ``queued`` are bare values
    assigned to the policy's own picks in order.  The recommendation is the point the
    policy proposes once every supplied measurement has been consumed.
    """
    n = len(block)
    axes = _standardised_axes(block)
    context = PolicyContext(block=block, prediction=prediction, pool=np.arange(n),
                            evaluation=np.arange(n), axes=axes, uncertainty=ensemble_sd,
                            disagreement=np.full(n, np.nan), rng=np.random.default_rng(0),
                            truth=None)   # <- no targets reach any policy
    selected: list[int] = []
    values: list[float] = []
    sources: list[str] = []
    rules: list[str] = []
    for index, value, source in explicit:
        selected.append(int(index))
        values.append(float(value))
        sources.append(source)
        rules.append("user-specified candidate")
    pending = list(queued)
    remaining_truth = int(n_truth)
    while True:
        step = len(selected)
        if step >= n:
            raise SystemExit("every candidate has been measured; nothing left to recommend")
        rule, _ = _rule_for(step)
        pick = policy_medoid(context, selected) if step == 0 else policy_farthest(context, selected)
        if pending:
            selected.append(int(pick))
            values.append(float(pending.pop(0)))
            sources.append("user-supplied measurement")
            rules.append(rule)
            continue
        if remaining_truth > 0 and truth_supplier is not None:
            selected.append(int(pick))
            values.append(float(truth_supplier(int(pick))))
            sources.append("cohort_truth (demo only)")
            rules.append(rule)
            remaining_truth -= 1
            continue
        return {"selected": selected, "values": values, "sources": sources,
                "rules": rules, "recommended": int(pick), "recommended_rule": rule}


def calibrate(block: pd.DataFrame, prediction: np.ndarray, selected: Sequence[int],
              observed: Sequence[float], evidence: dict | None = None
              ) -> tuple[np.ndarray, str, str]:
    """Apply the adaptation mode gen8 measured to be best at this k.

    Every number in the explanation is read out of the run artefact rather than
    written here, because the artefact is regenerated when the study is re-run and a
    literal in this file would quietly go stale.
    """
    k = len(selected)
    evidence = evidence or {}
    entry = evidence.get("by_k", {}).get(str(max(k, 1)), {})
    if k == 0:
        adapter = ZeroShot()
        zero = evidence.get("zero_shot")
        why = ("no measurement yet, so the frozen global model stands as it is"
               + (f" (macro MAE {zero:.3f} on held-out chemotypes)" if zero else ""))
    elif k == 1:
        adapter = RidgeOffset(mode="K1")
        why = ("one measurement supports one degree of freedom: the ligand's overall level. "
               "OFFSET_K1 shifts every prediction by the measured residual.")
    else:
        adapter = RidgeOffset(mode="K3")
        gain = entry.get("paired", {}).get("K3_vs_K1_adapter", {}).get("point")
        why = ("from two measurements the ridge-shrunk response-coefficient update also moves "
               "the lanthanide, acid and extractant slopes, with an unpenalised intercept"
               + (f"; measured worth over a plain offset at k={k}: {gain:.3f} macro MAE"
                  if gain else ""))
    adjusted = adapter.predict(block, prediction, np.asarray(selected, dtype=int),
                               np.asarray(observed, dtype=float),
                               AdaptContext(0, 0, str(block["extractant"].iloc[0])))
    return np.asarray(adjusted, dtype=float), adapter.name, why


# --------------------------------------------------------------------------- #
# Geography of the chosen point, within this candidate pool
# --------------------------------------------------------------------------- #

def _geography_of(index: int, block: pd.DataFrame, prediction: np.ndarray) -> dict:
    """Where does the recommended point sit, by the gen8 stratum definitions?

    The tercile/extremity helpers are imported from the script that *measured* the
    strata, so the label here and the number quoted beside it cannot drift apart.
    """
    from gen8_calibration_geography import ACID, EXTR, _extremity, _tercile

    out: dict = {}
    def column(name: str) -> np.ndarray:
        return block[name].to_numpy(dtype=float) if name in block.columns \
            else np.full(len(block), np.nan)
    out["acid_position"] = str(_tercile(column(ACID))[index])
    out["extractant_position"] = str(_tercile(column(EXTR))[index])
    out["metal_position"] = str(_extremity(column("lanthanide_index"))[index])
    out["prediction_position"] = str(_tercile(np.asarray(prediction, dtype=float))[index])
    return out


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

def pick_demo_ligand(cohort) -> str:
    """A real cohort ligand with a candidate space worth designing over.

    Deterministic: among ligands carrying at least 12 rows, 3 distinct acid
    concentrations and 3 distinct metals, take the one with the median row count
    (ties broken by SMILES) so the demo neither flatters itself with the easiest
    ligand nor picks the largest.
    """
    frame = cohort.frame
    summary = frame.groupby("extractant").agg(
        n=("row_id", "size"),
        n_acid=("cond__acid_concentration_M", "nunique"),
        n_metal=("metal_symbol", "nunique"))
    eligible = summary[(summary["n"] >= 12) & (summary["n_acid"] >= 3) & (summary["n_metal"] >= 3)]
    if not len(eligible):
        eligible = summary[summary["n"] >= 8]
    if not len(eligible):
        raise SystemExit("no cohort ligand is large enough for a demonstration")
    ordered = eligible.sort_values(["n", "n_acid"], kind="mergesort")
    ordered = ordered.reindex(sorted(ordered.index, key=lambda s: (int(ordered.loc[s, "n"]), s)))
    return str(ordered.index[len(ordered) // 2])


def write_demo_conditions(cohort, ligand: str, path: Path, *, max_rows: int) -> Path:
    """Emit a plausible candidate-conditions CSV: this ligand's real experiment grid."""
    frame = cohort.frame
    block = frame[frame["extractant"].astype(str) == ligand].copy()
    cond_columns = tuple(cohort.blocks["COND"])

    def onehot_label(prefix: str) -> list[str]:
        names = [c for c in cond_columns if c.startswith(prefix)]
        values = block[names].to_numpy(dtype=float)
        out = []
        for row in values:
            hit = np.flatnonzero(row > 0)
            out.append(names[int(hit[0])][len(prefix):] if hit.size else "")
        return out

    table = pd.DataFrame({
        "candidate_id": [f"C{i + 1:03d}" for i in range(len(block))],
        "row_id": block["row_id"].to_numpy(),
        "metal": block["metal_symbol"].astype(str).to_numpy(),
        "acid": onehot_label("cond__acid__"),
        "acid_concentration_M": block["cond__acid_concentration_M"].to_numpy(dtype=float),
        "extractant_concentration_M": block["cond__extractant_concentration_M"].to_numpy(dtype=float),
        "diluent": onehot_label("cond__diluent__"),
        "additive": onehot_label("cond__additive__"),
        "temperature_C": block["cond__temperature_C"].to_numpy(dtype=float),
        "metal_concentration_mM": block["cond__metal_concentration_mM"].to_numpy(dtype=float),
        "contact_time_min": block["cond__contact_time_min"].to_numpy(dtype=float),
        # A separate upstream field from contact time — 0 of 5,248 rows carry both —
        # and emitted so the reconstruction check exercises it rather than silently
        # tolerating a column the demo never supplies.
        "shaking_time_min": block["rec__shaking_time_min"].to_numpy(dtype=float),
    })
    if len(table) > max_rows:
        # deterministic thinning, seeded on the ligand so the demo is reproducible
        rng = np.random.default_rng(stable_hash(str(ligand)))
        keep = np.sort(rng.choice(len(table), size=max_rows, replace=False))
        table = table.iloc[keep].reset_index(drop=True)
        table["candidate_id"] = [f"C{i + 1:03d}" for i in range(len(table))]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def reconstruction_check(cohort, built: pd.DataFrame, row_ids: Sequence[int]) -> dict:
    """Do the features we build for a cohort ligand match the cohort's own row?

    This is the correctness test that matters: if the candidate frame reproduces the
    cohort's feature row exactly, then the prediction a chemist gets for a new ligand
    was produced by the same arithmetic as every number in the gen8 report.  Columns
    that *cannot* match are enumerated rather than hidden.
    """
    frame = cohort.frame.set_index("row_id")
    columns = list(cohort.block_columns(ARM_BLOCKS))
    reference = frame.loc[list(row_ids), columns].apply(pd.to_numeric, errors="coerce")
    ours = built[columns].apply(pd.to_numeric, errors="coerce")
    a, b = reference.to_numpy(dtype=float), ours.to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    delta = np.where(both_nan, 0.0, np.abs(a - b))
    delta = np.where(np.isnan(delta), np.inf, delta)   # one side NaN = a real mismatch
    per_column = delta.max(axis=0)
    agreeing = per_column <= 1e-9
    mismatched = [columns[i] for i in np.flatnonzero(~agreeing)]

    # How much of that agreement is actually load-bearing?  Most of the 2,146 columns
    # are ECFP bits that are zero for every ligand in the cohort, so a count of
    # reproduced columns flatters itself badly if quoted on its own.  Report the two
    # subsets that can actually discriminate: columns that vary across the cohort at
    # all, and — the strictest — columns that vary across *this ligand's own rows*,
    # which is where a mis-encoded condition or a wrong mass-action product would show.
    cohort_varying = np.array(
        [cohort.frame[name].nunique(dropna=True) > 1 for name in columns])
    within_varying = np.array(
        [int(pd.Series(a[:, i]).nunique(dropna=True)) > 1 for i in range(len(columns))])
    return {"n_columns": len(columns),
            "n_reproduced_exactly": int(agreeing.sum()),
            "n_mismatched": len(mismatched),
            "mismatched_columns": mismatched[:20],
            "max_abs_difference_over_reproduced_columns": float(
                per_column[agreeing].max()) if agreeing.any() else None,
            "discriminating_power": {
                "n_columns_varying_across_the_cohort": int(cohort_varying.sum()),
                "n_cohort_varying_reproduced": int((cohort_varying & agreeing).sum()),
                "n_columns_varying_within_this_ligand": int(within_varying.sum()),
                "n_within_ligand_varying_reproduced": int((within_varying & agreeing).sum()),
                "note": "the headline count is dominated by columns that are the same "
                        "constant for every row in the cohort (mostly unset ECFP bits); "
                        "these two subsets are the part of the check that can fail.  The "
                        "ligand-specific fingerprint is verified separately, bit for bit, "
                        "by tests/test_gen8_recommender.py."},
            "note": "the RECOVERED literature-provenance columns cannot be reconstructed "
                    "prospectively and are the expected mismatches; everything else must "
                    "agree to 1e-9 or the deployed features are not the trained features"}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

SECTION_DOCUMENTATION = {
    "query": "the ligand, its nearest training chemistry, and whether it counts as a "
             "novel chemotype (Tanimoto < 0.7 to every training ligand)",
    "model": "the frozen arm, its feature blocks, the rows it was refitted on, its seed, "
             "and whether the recipe was left frozen",
    "stage": "how many measurements are in hand and which calibration adapter that selects",
    "recommendation": "the single next experiment, with the rule that chose it and the "
                      "measured improvement that rule buys over choosing at random",
    "observations": "the measurements supplied so far, in the order they were taken",
    "predictions": "one entry per candidate condition; see field_documentation",
    "uncertainty": "how the prediction intervals were calibrated and whether this query "
                   "falls inside the range they were measured over",
    "evidence": "the run artefacts the policy and the intervals rest on, re-derived at "
                "run time rather than quoted from memory",
    "assumptions": "values the run had to assume because the candidate table did not "
                   "supply them",
    "warnings": "every assumption and caveat, in plain language",
    "demo": "present only for --demo: which cohort ligand was used and whether the "
            "features built here reproduce that ligand's own cohort feature row",
}

FIELD_DOCUMENTATION = {
    "candidate_id": "your label for this candidate experiment",
    "conditions": "the experiment, echoed back after normalisation",
    "zero_shot_log_D": "the frozen global model's prediction, before any calibration",
    "predicted_log_D": "the prediction after calibrating on the measurements supplied",
    "ensemble_sd": "spread of the 400 ExtraTrees over this row; a diagnostic only — "
                   "gen8 measured that ranking candidates by it is no better than random",
    "uncertainty_68": "half-width of a 68 % interval, from the empirical error distribution "
                      "on held-out chemotypes at this ligand's similarity and this k",
    "uncertainty_90": "the same at 90 %",
    "is_observed": "true if this candidate's value was supplied with --observed",
    "is_recommended": "true for the single next experiment",
    "distance_to_pool_centre": "L1 medoid cost in standardised condition space; the first "
                               "rule minimises it",
    "distance_to_measured": "Euclidean distance to the nearest measured candidate; the "
                            "second and later rules maximise it",
}


def _jsonable(value):
    # bool before int: ``bool`` subclasses ``int`` in Python, so testing int first
    # turned every documented boolean flag into 0/1 in the emitted JSON.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, str):
        return value
    return str(value)


def print_summary(payload: dict) -> None:
    q, m, s, r = payload["query"], payload["model"], payload["stage"], payload["recommendation"]
    print("=" * 78)
    print("gen8 experiment recommender")
    print("=" * 78)
    print(f"ligand                 {q['ligand_smiles']}")
    print(f"nearest training ligand  Tanimoto {q['nearest_training_tanimoto']}  "
          f"({q['chemotype_status']})")
    print(f"candidates             {q['n_candidates']}")
    print(f"model                  {m['arm']}  ({m['n_train_rows']} rows, "
          f"{m['n_train_ligands']} ligands, {m['n_train_chemotypes']} chemotypes"
          + (f", chemotype {m['excluded_chemotype']} withheld" if m["excluded_chemotype"] else "")
          + ")")
    print(f"measurements in hand   {s['k_observed']}   ->  adapter {s['adapter']}")
    print()
    if payload["observations"]:
        print("-- measurements supplied ---------------------------------------------------")
        print(md_table(pd.DataFrame([
            {"candidate_id": o["candidate_id"], "rule": o["rule"],
             "observed_log_D": o["observed_log_D"], "source": o["source"],
             **{k: v for k, v in o["conditions"].items()
                if k in ("metal", "acid_concentration_M", "extractant_concentration_M")}}
            for o in payload["observations"]])))
        print()
    print("-- RECOMMENDED NEXT EXPERIMENT ---------------------------------------------")
    print(f"  candidate  {r['candidate_id']}     rule: {r['rule']}")
    for key, value in r["conditions"].items():
        if value not in (None, "", "nan"):
            print(f"    {key:<30} {value}")
    print(f"  predicted log D        {r['predicted_log_D']}"
          f"  (+/- {r['uncertainty_68']} at 68 %, +/- {r['uncertainty_90']} at 90 %)")
    print(f"  why                    {r['why']['rule_fired']}")
    improvement = r["why"].get("expected_improvement", {})
    if improvement.get("available"):
        print(f"  expected macro MAE     {improvement['recommended_macro_mae']} with this rule "
              f"vs {improvement['random_macro_mae']} choosing at random "
              f"(gain {improvement['gain_over_random']}, "
              f"CI [{improvement['gain_bca_low']}, {improvement['gain_bca_high']}], "
              f"{improvement['seeds_positive']}/{improvement['n_seeds']} seeds)")
        rank = improvement.get("recommended_rank_among_deployable_policies")
        if rank:
            print(f"  rank among deployable  {rank} of "
                  f"{improvement.get('n_deployable_policies')} policies measured; best "
                  f"alternative {improvement.get('best_alternative_policy')} at "
                  f"{improvement.get('best_alternative_macro_mae')}")
            tied = improvement.get("policies_scoring_identically") or []
            if tied:
                print(f"    scoring identically  {', '.join(tied)} (same rule at this k)")
            versus = improvement.get("vs_best_alternative") or {}
            low, high = versus.get("bca_low"), versus.get("bca_high")
            if versus.get("point") is not None and low is not None and high is not None:
                separated = not (low <= 0 <= high)
                print(f"    vs that alternative  {versus['point']:+.4f} "
                      f"CI [{low:+.4f}, {high:+.4f}] "
                      f"({'a measurable difference' if separated else 'no measurable difference'})")
        print(f"  oracle (not deployable) {improvement['oracle_macro_mae']}"
              + ("" if improvement.get("oracle_is_a_valid_lower_bound")
                 else "   <- NOT a lower bound at this k; see oracle_caveat"))
    geography = r["why"].get("geography_of_this_point", {})
    if geography:
        print("  this point sits at:    " + ", ".join(f"{k}={v}" for k, v in geography.items()))
    print()
    table = pd.DataFrame(payload["predictions"])
    show = ["candidate_id", "metal", "acid_concentration_M", "extractant_concentration_M",
            "zero_shot_log_D", "predicted_log_D", "ensemble_sd", "is_observed", "is_recommended"]
    flat = pd.DataFrame([{**{k: p["conditions"].get(k) for k in
                             ("metal", "acid_concentration_M", "extractant_concentration_M")},
                          **{k: p.get(k) for k in show if k not in
                             ("metal", "acid_concentration_M", "extractant_concentration_M")}}
                         for p in payload["predictions"]])
    print(f"-- predictions for all {len(flat)} candidates ---------------------------------")
    ordered = flat.sort_values("predicted_log_D", ascending=False)
    if len(ordered) > 24:
        print(md_table(ordered.head(12), columns=show))
        print(f"  ... {len(ordered) - 24} rows omitted ...")
        print(md_table(ordered.tail(12), columns=show))
    else:
        print(md_table(ordered, columns=show))
    if payload["warnings"]:
        print()
        print("-- assumptions and warnings ------------------------------------------------")
        for line in payload["warnings"]:
            print(f"  * {line}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_observations(tokens: Sequence[str], conditions: pd.DataFrame
                       ) -> tuple[list[tuple[int, float, str]], list[float]]:
    """``--observed 1.2`` (next policy point) or ``--observed C007=1.2`` (that candidate)."""
    ids = {str(v): i for i, v in enumerate(conditions["candidate_id"])}
    explicit: list[tuple[int, float, str]] = []
    queued: list[float] = []
    for token in tokens:
        text = str(token)
        if "=" in text:
            name, value = text.split("=", 1)
            name = name.strip()
            if name not in ids:
                raise SystemExit(f"--observed refers to unknown candidate_id {name!r}")
            explicit.append((ids[name], float(value), "user-supplied measurement (pinned)"))
        else:
            queued.append(float(text))
    return explicit, queued


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recommend the next solvent-extraction experiment for a new extractant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ligand", help="extractant SMILES")
    parser.add_argument("--candidate-conditions", type=Path,
                        help="CSV of experiments that could be run")
    parser.add_argument("--observed", nargs="*", default=[],
                        help="measured log D values, in the order the recommender proposed "
                             "them; or CANDIDATE_ID=VALUE to pin one")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON (default runs/gen8_architecture/recommender/"
                             "recommended_condition.json)")
    parser.add_argument("--demo", action="store_true",
                        help="generate a candidate-conditions CSV for a real held-out cohort "
                             "ligand and run end to end")
    parser.add_argument("--demo-ligand", default=None, help="override the demo's ligand SMILES")
    parser.add_argument("--demo-max-candidates", type=int, default=40)
    parser.add_argument("--demo-observe-truth", type=int, default=0,
                        help="demo only: feed back the cohort's recorded value for this many "
                             "recommended points, to exercise the second-experiment path")
    parser.add_argument("--n-estimators", type=int, default=400,
                        help="trees in the frozen arm; 400 is the frozen recipe and anything "
                             "else departs from it")
    parser.add_argument("--refresh-evidence", action="store_true",
                        help="recompute the cached policy-gain and interval tables")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    out_dir = args.out.parent if args.out else DEFAULT_OUT_DIR
    out_path = args.out or (DEFAULT_OUT_DIR / "recommended_condition.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.demo and (not args.ligand or not args.candidate_conditions):
        raise SystemExit("--ligand and --candidate-conditions are required (or use --demo)")

    if not args.quiet:
        print("loading the frozen cohort ...", flush=True)
    cohort = load_cohort()
    frame = cohort.frame

    demo: dict | None = None
    if args.demo:
        ligand = args.demo_ligand or pick_demo_ligand(cohort)
        conditions_path = out_dir / "demo_candidate_conditions.csv"
        write_demo_conditions(cohort, ligand, conditions_path, max_rows=args.demo_max_candidates)
        demo = {"ligand": ligand, "conditions_csv": str(conditions_path)}
        if not args.quiet:
            print(f"demo: ligand {ligand}\n      wrote {conditions_path}", flush=True)
    else:
        ligand = args.ligand
        conditions_path = args.candidate_conditions

    smiles = canonical_smiles(ligand)
    conditions = read_candidate_conditions(Path(conditions_path))

    # -- is this ligand in the cohort, and what must be withheld? ------------- #
    cohort_rows = frame[frame["extractant"].astype(str) == smiles]
    in_cohort = bool(len(cohort_rows))
    excluded_chemotype = str(cohort_rows["tanimoto_cluster"].iloc[0]) if in_cohort else None
    cohort_ligand_row = cohort_rows.iloc[0] if in_cohort else None

    # -- similarity to the chemistry the model will actually be trained on ---- #
    ecfp_columns = [f"ecfp_{i}" for i in range(ECFP_BITS)]
    training_mask = np.ones(len(frame), dtype=bool) if excluded_chemotype is None else \
        (frame["tanimoto_cluster"].astype(str).to_numpy() != excluded_chemotype)
    training_ligands = frame.loc[training_mask].drop_duplicates("extractant")
    reference = training_ligands[ecfp_columns].to_numpy(dtype=float)
    query_bits = ecfp_bits(smiles)
    similarities = tanimoto_to_set(query_bits, reference)
    best = int(np.argmax(similarities)) if similarities.size else -1
    nn_tanimoto = float(similarities[best]) if best >= 0 else float("nan")
    nn_ligand = str(training_ligands["extractant"].iloc[best]) if best >= 0 else None
    chemotype_status = "novel chemotype (nn < 0.7 — the regime every quoted number was " \
                       "measured in)" if nn_tanimoto < CHEMOTYPE_THRESHOLD else \
        "NOT a novel chemotype: a training ligand sits at Tanimoto >= 0.7, so the quoted " \
        "held-out-chemotype numbers are pessimistic for this query"

    # -- features, model, calibration ---------------------------------------- #
    built = build_candidate_frame(cohort, smiles, conditions, cohort_ligand_row=cohort_ligand_row)
    block = built.frame
    warnings_out = list(built.warnings)
    if not args.quiet:
        print(f"fitting {ARM_NAME} on {int(training_mask.sum())} rows ...", flush=True)
    model = fit_and_predict(cohort, block, excluded_chemotype=excluded_chemotype,
                            n_estimators=args.n_estimators)
    zero_shot = model.prediction

    evidence_policy = cached_json(DEFAULT_OUT_DIR / "policy_evidence.json",
                                  build_policy_evidence, refresh=args.refresh_evidence)
    calibration = cached_json(DEFAULT_OUT_DIR / "uncertainty_calibration.json",
                              build_uncertainty_calibration, refresh=args.refresh_evidence)

    explicit, queued = parse_observations(args.observed, conditions)
    truth_supplier = None
    n_truth = 0
    if args.demo_observe_truth:
        if not args.demo or "row_id" not in conditions.columns:
            raise SystemExit("--demo-observe-truth needs --demo (its CSV carries row_id)")
        truth_by_row = frame.set_index("row_id")[LEVEL_TARGET_COLUMN]
        row_ids = conditions["row_id"].to_numpy()
        truth_supplier = lambda i: float(truth_by_row.loc[row_ids[i]])  # noqa: E731
        n_truth = int(args.demo_observe_truth)

    plan = run_policy(block, zero_shot, model.ensemble_sd, explicit=explicit, queued=queued,
                      truth_supplier=truth_supplier, n_truth=n_truth)
    selected, values = plan["selected"], plan["values"]
    adjusted, adapter_name, adapter_why = calibrate(block, zero_shot, selected, values,
                                                    evidence_policy)

    k = len(selected)
    interval = interval_for(calibration, nn_tanimoto, k)
    recommended = plan["recommended"]
    _, rule_description = _rule_for(k)

    # geometry of the pool, reported so the choice is auditable
    axes = _standardised_axes(block)
    medoid_cost = np.abs(axes[:, None, :] - axes[None, :, :]).sum(axis=(1, 2))
    if selected:
        taken = axes[np.asarray(selected, dtype=int)]
        distance_measured = np.linalg.norm(axes[:, None, :] - taken[None, :, :], axis=2).min(axis=1)
    else:
        distance_measured = np.full(len(block), np.nan)

    condition_fields = [c for c in ("metal", "acid", "acid_concentration_M",
                                    "extractant_concentration_M", "diluent", "additive",
                                    "temperature_C", "metal_concentration_mM",
                                    "contact_time_min") if c in conditions.columns]

    def conditions_of(index: int) -> dict:
        row = conditions.iloc[index]
        return {name: _jsonable(row[name]) for name in condition_fields}

    k_evidence = evidence_policy.get("by_k", {}).get(str(k + 1), {})
    paired = k_evidence.get("paired", {}).get("recommended_vs_random", {})
    improvement = {"available": bool(k_evidence)}
    if not k_evidence:
        measured_at = sorted(int(key) for key in evidence_policy.get("by_k", {}))
        improvement["reason"] = (
            f"the acquisition study measured k = {measured_at}; with {k} measurement(s) in "
            f"hand this would be the k = {k + 1} point, which was not measured, so no "
            f"expected improvement is quoted rather than one being interpolated")
    if k_evidence:
        improvement.update({
            "k_after_this_measurement": k + 1,
            "recommended_macro_mae": _jsonable(k_evidence.get("recommended_macro_mae")),
            "random_macro_mae": _jsonable(k_evidence.get("random_macro_mae")),
            "oracle_macro_mae": _jsonable(k_evidence.get("oracle_macro_mae")),
            "oracle_policy": k_evidence.get("oracle_policy"),
            "oracle_is_a_valid_lower_bound": _jsonable(
                k_evidence.get("oracle_is_a_valid_lower_bound")),
            "oracle_caveat": k_evidence.get("oracle_caveat"),
            "recommended_rank_among_deployable_policies": _jsonable(
                k_evidence.get("recommended_rank_among_deployable_policies")),
            "n_deployable_policies": _jsonable(k_evidence.get("n_deployable_policies")),
            "policies_scoring_identically": k_evidence.get("policies_scoring_identically", []),
            "best_alternative_policy": k_evidence.get("best_alternative_policy"),
            "best_alternative_macro_mae": _jsonable(k_evidence.get("best_alternative_macro_mae")),
            "vs_best_alternative": _jsonable(
                k_evidence.get("paired", {}).get("recommended_vs_best_alternative", {})),
            "zero_shot_macro_mae": _jsonable(evidence_policy.get("zero_shot")),
            "gain_over_random": _jsonable(paired.get("point")),
            "gain_bca_low": _jsonable(paired.get("bca_low")),
            "gain_bca_high": _jsonable(paired.get("bca_high")),
            "ligands_improved": _jsonable(paired.get("units_improved")),
            "n_ligands": _jsonable(paired.get("n_units")),
            "seeds_positive": _jsonable(paired.get("seeds_positive")),
            "n_seeds": _jsonable(paired.get("n_seeds")),
            "units": "macro MAE in log D units, one held-out ligand one vote",
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": {
            "ligand_smiles": smiles,
            "ligand_as_given": ligand,
            "ligand_in_cohort": in_cohort,
            "n_candidates": int(len(conditions)),
            "nearest_training_ligand": nn_ligand,
            "nearest_training_tanimoto": _jsonable(nn_tanimoto),
            "chemotype_status": chemotype_status,
            "candidate_conditions_file": str(conditions_path),
        },
        "model": {
            "arm": ARM_NAME,
            "blocks": list(ARM_BLOCKS),
            "recipe": "ExtraTrees(n_estimators=%d, max_features=0.30, min_samples_leaf=2), "
                      "median imputation with missingness indicators, ECFP-cluster-balanced "
                      "sample weights, predictions clipped to the training range"
                      % model.n_estimators,
            "is_frozen_recipe": bool(model.n_estimators == 400),
            "n_train_rows": model.n_train_rows,
            "n_train_ligands": model.n_train_ligands,
            "n_train_chemotypes": model.n_train_chemotypes,
            "excluded_chemotype": model.excluded_chemotype,
            "model_seed": model.model_seed,
            "zero_shot_macro_mae_on_held_out_chemotypes": _jsonable(
                evidence_policy.get("zero_shot")),
        },
        "stage": {
            "k_observed": k,
            "adapter": adapter_name,
            "adapter_reason": adapter_why,
            "next_rule": plan["recommended_rule"],
        },
        "recommendation": {
            "candidate_id": str(conditions["candidate_id"].iloc[recommended]),
            "rule": plan["recommended_rule"],
            "rule_description": rule_description,
            "conditions": conditions_of(recommended),
            "predicted_log_D": _jsonable(adjusted[recommended]),
            "zero_shot_log_D": _jsonable(zero_shot[recommended]),
            "ensemble_sd": _jsonable(model.ensemble_sd[recommended]),
            "uncertainty_68": _jsonable(interval.get("q68")),
            "uncertainty_90": _jsonable(interval.get("q90")),
            "why": {
                "rule_fired": plan["recommended_rule"],
                "rule_description": rule_description,
                "expected_improvement": improvement,
                "geography_of_this_point": _geography_of(recommended, block, zero_shot),
                "geography_evidence": evidence_policy.get("geography", []),
                "not_used": "the model's own uncertainty.  At k = 1 no uncertainty policy "
                            "differs from RANDOM by a margin that excludes zero — two of "
                            "the four (MAX_ENSEMBLE_SD, MIN_ENSEMBLE_SD) do have the better "
                            "point estimate, so the refusal rests on the paired intervals, "
                            "not on the ordering — while the deployed rule beats each of "
                            "them by an interval that does exclude zero.  The per-candidate "
                            "ensemble_sd is reported as a diagnostic and never ranked on. "
                            "The measured comparison is in "
                            "evidence.acquisition.uncertainty_policies_at_k1_paired.",
                "uncertainty_policies_at_k1": _jsonable(
                    evidence_policy.get("uncertainty_policies_at_k1", {})),
                "uncertainty_policies_at_k1_paired": _jsonable(
                    evidence_policy.get("uncertainty_policies_at_k1_paired", {})),
            },
        },
        "observations": [
            {"candidate_id": str(conditions["candidate_id"].iloc[i]),
             "conditions": conditions_of(i), "observed_log_D": _jsonable(v),
             "source": src, "rule": rule}
            for i, v, src, rule in zip(selected, values, plan["sources"], plan["rules"])
        ],
        "predictions": [
            {"candidate_id": str(conditions["candidate_id"].iloc[i]),
             "conditions": conditions_of(i),
             "zero_shot_log_D": _jsonable(zero_shot[i]),
             "predicted_log_D": _jsonable(adjusted[i]),
             "ensemble_sd": _jsonable(model.ensemble_sd[i]),
             "uncertainty_68": _jsonable(interval.get("q68")),
             "uncertainty_90": _jsonable(interval.get("q90")),
             "is_observed": bool(i in set(selected)),
             "is_recommended": bool(i == recommended),
             "distance_to_pool_centre": _jsonable(medoid_cost[i]),
             "distance_to_measured": _jsonable(distance_measured[i])}
            for i in range(len(block))
        ],
        "uncertainty": {**_jsonable(interval),
                        "source": calibration.get("source"),
                        "policy_replayed": calibration.get("policy_replayed")},
        "evidence": {"acquisition": _jsonable(evidence_policy),
                     "interval_calibration": _jsonable(
                         {k2: v for k2, v in calibration.items() if k2 != "bins"})},
        "assumptions": _jsonable(built.assumptions),
        "warnings": warnings_out,
        "section_documentation": SECTION_DOCUMENTATION,
        "field_documentation": FIELD_DOCUMENTATION,
        "runtime_seconds": None,
    }

    if demo is not None:
        payload["demo"] = {
            **demo,
            "reconstruction_check": _jsonable(reconstruction_check(
                cohort, block, conditions["row_id"].to_numpy()))
            if "row_id" in conditions.columns else None,
            "note": "the ligand's whole Tanimoto chemotype was removed from the training set, "
                    "so this demonstration is under the same hold-out contract as the "
                    "measured numbers quoted above",
        }
    if in_cohort:
        removed = 1.0 - model.n_train_rows / len(frame)
        warnings_out.append(
            f"the queried ligand is in the cohort; its entire Tanimoto chemotype "
            f"({excluded_chemotype}) was removed from the training set before fitting "
            f"({removed:.0%} of the cohort's rows)")
        if removed > 0.35:
            warnings_out.append(
                f"that chemotype is unusually large, so this model saw only "
                f"{model.n_train_rows} of {len(frame)} rows — a harder training set than the "
                f"average cross-validation fold, and the predictions here are correspondingly "
                f"weaker than the headline macro MAE")
    if nn_tanimoto >= CHEMOTYPE_THRESHOLD:
        warnings_out.append(
            f"a training ligand sits at Tanimoto {nn_tanimoto:.3f} >= {CHEMOTYPE_THRESHOLD}; "
            f"every accuracy number quoted here was measured on ligands with no such "
            f"neighbour and is therefore pessimistic for this query")
    if interval.get("extrapolated_similarity"):
        warnings_out.append(
            "this ligand is more similar to training chemistry than any held-out ligand the "
            "interval table could be measured on; the reported interval is the closest "
            "measured bin and is not calibrated for this case")
    if interval.get("extrapolated_k"):
        warnings_out.append(
            f"prediction intervals were measured at k = 0, 1 and 2 only; with {k} "
            f"measurements in hand the k = 2 interval is being reused and is conservative")
    if not evidence_policy.get("available"):
        warnings_out.append("the gen8 acquisition run artefact was not found, so the expected "
                            "improvement over a random choice could not be quoted")
    payload["runtime_seconds"] = round(time.time() - started, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    if not args.quiet:
        print_summary(payload)
        print()
        print(f"wrote {out_path}  ({payload['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
