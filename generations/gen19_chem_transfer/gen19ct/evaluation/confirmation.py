"""``evaluation/confirmation.py`` -- the ONE registered confirmation run (pre-registration section 15).

Section 15 registers exactly one run: *"One run on the withheld seeds scores those claims on the **confirmation half**
of each design and runs **V6 once**"*, *"A claim is **confirmed** only if it passes R19 with 5 of 5 confirmation
seeds"*, *"Whatever confirmation returns is the result. No sixth claim is added and nothing is re-run"*.

POST-HOC addendum 5 item 1 makes that run the **CORE** run, compute-driven:

* the frozen claims of ``decisions/CONFIRMATION_PLAN.md`` (at most five) on the confirmation half with the 5 withheld
  seeds, **R19 item 4 exactly as registered (5 of 5 seeds)**;
* the single V6 run of section 3.4 with **S2(a), S2(b), S2(c)**;
* **S1(c)** under its addendum-2 confirmation rule and **S1(d)** calibration;
* it does **NOT** run the section 11 V6 actinide deltas nor any section 8 power check: both are reported ``NOT_RUN``
  with their cost and their consequence (:data:`V6_ACTINIDE_DELTAS_NOT_RUN`, :data:`POWER_CHECK_NOT_RUN`; a contrast
  with no power check is UNDECIDED, never a null -- addendum 4 item 4);
* **R19 item 6 refits run on seed 104729 only**, as in discovery (addendum 1 item 4).

The withheld seeds
------------------
Five seeds committed as ``sha256(canonical JSON{n, salt, schema, seeds})`` =
``65e8ae8c...f82`` (``manifests/confirmation_seeds_sha256.txt``), stored OUTSIDE the repository.  They enter this code
**only** through the ``--seed-store PATH`` argument of ``scripts/g19_run_confirmation.py``, at run time, and only as a
:class:`SeedStore` whose ``repr`` is redacted.  Nothing written by this module names a seed: a fold, a record, a table
and a log line carry an **opaque seed index 1..5** (:meth:`SeedStore.index_of`) and the commitment digest, and
``decisions/CONFIRMATION.md`` reveals the seeds only as the ``--verify-seeds`` verdict (:data:`SEED_DISCLOSURE`).
:func:`scan_for_seed_leak` re-reads every file the run wrote and fails the run if a seed's decimal form appears.

What this module is, and is not
-------------------------------
It holds the run's **rules**: the gates, the plan it reads, the job and fold inventory, the cost model, the
confirmation-half scoring population, the statistics of R19 item 4 / S1 / S2, the ``NOT_RUN`` records and the writers.
The **fitting** is ``scripts/g19_run_confirmation.py`` (the stage's runner), which owns the confirmation-half
``prepare_fold`` -- discovery's refuses a confirmation-half fold by construction
(``g19_run_discovery.prepare_fold``: *"a confirmation-half fold is never fitted in discovery"*) -- and reuses the
discovery runners, tuning and record schema unchanged.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen19ct import paths
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import registry as REG
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import io as FI
from gen19ct.manifest import write_json

SCHEMA = "gen19.confirmation_record.v1"
DECISION_SCHEMA = "gen19.confirmation_decisions.v1"
#: the registry stage every record of this run carries (``registry.STAGES``)
STAGE = "confirmation"
CONF_REL = "evaluation/confirmation"
PLAN_REL = "decisions/CONFIRMATION_PLAN.md"
REPORT_REL = "decisions/CONFIRMATION.md"
SEED_COMMITMENT_REL = "manifests/confirmation_seeds_sha256.txt"
#: section 15: five withheld seeds, and R19 item 4 needs 5 of 5 of them
N_SEEDS = 5
N_SEEDS_REQUIRED = ET.MIN_SEEDS_POSITIVE_CONFIRMATION
#: section 15: "at most 5 claims are frozen in decisions/CONFIRMATION_PLAN.md"
MAX_CLAIMS = 5
#: the discovery seed R19 item 6's refits run on, at confirmation as in discovery (addendum 1 item 4, addendum 5 item 1)
ITEM6_REFIT_SEED = 104729
#: section 9 gamma5 / eta5 / section 8 epsilon, and the S2(a) sign count of section 9
GAMMA5, ETA5 = 0.05, 0.02
EPSILON = ET.EPSILON
S2A_MIN_SIGN_SYSTEMS, S2A_N_SYSTEMS = 11, 13
S2B_MARGIN = 0.02
#: section 9 S1(d) / S2(c) coverage bands
S1D_BANDS: dict[str, tuple[float, float]] = {"50": (0.40, 0.60), "80": (0.70, 0.90), "95": (0.88, 0.99)}
S1D_CATEGORY_80 = (0.65, 0.92)
S1D_CATEGORY_MIN_CELLS = 20
S2C_BANDS: dict[str, tuple[float, float]] = {"80": (0.70, 0.90)}
S2C_MIN_95 = 0.88
NOT_RUN = "NOT_RUN"
NOT_EVALUATED = "NOT_EVALUATED"
UNDECIDED = "UNDECIDED"

SEED_DISCLOSURE = (
    "decisions/CONFIRMATION.md reveals the withheld seeds ONLY as the verdict of "
    "`scripts/g19_seal_prereg.py --verify-seeds --seed-store <path>` against the commitment of section 15 "
    "(manifests/confirmation_seeds_sha256.txt), which prints no seed. The VALUES are never written to the repository, "
    "never logged and never put in a fold id, a record, a table or a figure: per fold and per record they appear as an "
    "opaque seed index 1..5 (confirmation.SeedStore.index_of, the store's own order). Section 15's sentence 'They are "
    "revealed in decisions/CONFIRMATION.md together with the --verify-seeds verdict' is satisfied by the verdict: the "
    "commitment is what makes the seeds checkable, and printing the values would let a later run reproduce a "
    "confirmation fit -- POST-HOC addendum 5 item 1 keeps this run single. A POST-HOC addendum recording that reading "
    "of section 15 is REQUESTED (it narrows 'revealed' to 'verified'); until one is written the values stay withheld, "
    "which is the conservative side of a rule written to stop seed luck")

READINGS: dict[str, str] = {
    "seed_entry": "the 5 withheld seeds enter this code ONLY through --seed-store PATH at run time, verified against "
                  "the section 15 commitment by scripts/g19_seal_prereg.py --verify-seeds (which never prints them); "
                  "no default path exists, no environment variable is read and nothing is cached: a second run needs "
                  "the store again",
    "seed_disclosure": SEED_DISCLOSURE,
    "half": "every row this run scores is in the CONFIRMATION half of its design (feasibility_halves.csv; "
            "assert_confirmation_rows), the half section 15 says contributed to no ladder decision, claim or "
            "preferred-model choice. Discovery's own prepare_fold refuses a confirmation-half fold, so the runner owns "
            "its own; nothing here ever scores a selection-half row and nothing in discovery ever scores a "
            "confirmation-half row",
    "v6_scope": "section 3.1 carves V6_TARGET_ROWS out of every design, and section 3.4 runs V6 ONCE, here. So this "
                "run is the only place a V6_TARGET_ROWS row is scored, and only by a V6 job: assert_v6_scope refuses a "
                "V6 row in any other design's scored set and refuses a V6 job that scores a row outside V6_TARGET_ROWS",
    "item4": "R19 item 4 as registered at confirmation: Delta > 0 in 5 of 5 withheld seeds (section 8 item 4; "
             "addendum 3 item 1 restates it; addendum 5 item 1 keeps it exactly). A claim with fewer than 5 scored "
             "seeds is NOT confirmed and is reported with the seeds it has -- never rounded up",
    "item6_seed": "R19 item 6's strict and HNO3-only REFITS run on seed 104729 only, as in discovery (addendum 1 item "
                  "4, kept by addendum 5 item 1); the four scoring-filter sensitivities are re-scorings of the primary "
                  "fits and are evaluated on all 5 withheld seeds. The report prints which sensitivity was decided on "
                  "how many seeds, and the plan's 'conservative reading' (all 5 seeds) is NOT taken: addendum 5 item 1 "
                  "chose the core run",
    "s1c_seed_combination": "S1(c) at confirmation (section 9 S1(c) as redefined, addendum 2): each Delta_Y is the MEAN "
                            "over the 5 withheld seeds of the per-seed cell-pair-macro Delta_Y; its interval is the "
                            "percentile interval of a system-cluster bootstrap (10,000 resamples, seed 19) of that seed "
                            "mean with THE SAME resampled systems applied to every seed; and Delta_Y > 0 in 5 of 5 "
                            "seeds. Pass: min_Y Delta_Y >= gamma5 = 0.05 and every Delta_Y's interval excludes 0, "
                            "together with the selection-half counterweight min_Y Delta_Y >= -0.02 already satisfied",
    "s2a": "S2(a): the sign of the per-system median PREDICTED logSF_Nd/Pr equals the sign of the observed median in "
           ">= 11 of the 13 V6 systems (section 9, fixed pre-seal), and pair-level direction accuracy (|observed| >= "
           "0.1) beats every yardstick Y in {HEAVIER, B3x-derived, B3i-derived, B8} by >= 0.05 under the paired rule of "
           "S1(c), pooled AND in the HNO3 pairs. A system whose observed or predicted median is 0 or undefined counts "
           "as NOT agreeing (the conservative side of a sign count)",
    "v6_deltas_not_run": "POST-HOC addendum 5 item 1: the section 11 V6 actinide deltas are NOT_RUN in this run, with "
                         "their cost and their consequence recorded (V6_ACTINIDE_DELTAS_NOT_RUN)",
    "power_not_run": "POST-HOC addendum 5 items 1 and 4: no section 8 power check runs here; the debt is inventoried, "
                     "not discharged, and every contrast owing one is UNDECIDED, never a null (POWER_CHECK_NOT_RUN)",
    "idempotence": "section 15's 'nothing is re-run' as a lock: the runner refuses to start when "
                   "evaluation/confirmation/decisions/confirmation.json exists unless --resume, and --resume may only "
                   "COMPLETE unfitted folds -- it re-scores no claim under a code digest different from the one the "
                   "lock records, and it never widens the claim list (lock_verdict)",
}

#: POST-HOC addendum 5 item 1, with the cost that drove it and the consequence of not running it
V6_ACTINIDE_DELTAS_NOT_RUN = {
    "status": NOT_RUN,
    "what": "the section 11 V6 actinide deltas (H3 at confirmation): arms M0 (= B5, the deployed configuration of "
            "addendum 3 item 2) and B6, transforms WITH / WITHOUT / ACT_PERMUTED at the frozen per-fold "
            "hyperparameters, deltas WITH - WITHOUT and WITH - PERMUTED with R19 and TOST (epsilon = 0.05) on "
            "macro MAE log D, rank accuracy, logSF MAE and calibration",
    "registered_by": "section 11 ('V6, at confirmation only') and CONFIRMATION_PLAN section 3.4",
    "not_run_by": "POST-HOC addendum 5 item 1 (compute-driven, chosen by the user): 'It does NOT run the section 11 V6 "
                  "actinide deltas nor any section 8 power check; both are reported NOT_RUN with their cost and their "
                  "consequence'",
    "cost_hours_serial": 30.3,
    "cost_basis": "CONFIRMATION_PLAN section 6: M0 = B5 WITH (574.3 s) + WITHOUT (495.8) + PERMUTED (579.1) over 13 x 5 "
                  "= 65 folds = 29.8 h serial, plus the B6 reference's three transforms at about 30 s/fold = 0.5 h; "
                  "measured per-fold means from evaluation/h3/records/<arm>/<transform>/<design>/s104729/*.json steps",
    "consequence": "H3 keeps the discovery-side verdict: UNDECIDED for both arms, with no power check, so it is never "
                   "reported as a null (addendum 4 item 4). Delta logSF MAE for H3 stays NOT_DEFINED -- V6 is the only "
                   "design that could have supplied it (CONFIRMATION_PLAN section 3.4 item 2), so the quantity now "
                   "exists nowhere and no later run may create it: section 3.4 says V6 runs ONCE and addendum 5 spends "
                   "that run on S2. The section 11 consequence is unchanged: actinide rows enter the deployed Ln "
                   "configuration only on a *helps* verdict, and there is none",
}
POWER_CHECK_NOT_RUN = {
    "status": NOT_RUN,
    "what": "the section 8 signal-injection power check over the H3 contrasts (y' = y + kappa*s, kappa in "
            "{0.1, 0.25, 0.5, 1.0}, every endpoint of the contrast refitted at its selected hyperparameters)",
    "not_run_by": "POST-HOC addendum 5 items 1 and 4: 'the power-check debt is inventoried, not discharged'",
    "inventory": "20 of 20 H3 contrasts owe the check and 0 have one: 6,272 injected refits, about 436 h of serial "
                 "compute at the measured per-fold cost (evaluation/h3/h3_summary.json -> power_debt; the totals are an "
                 "upper bound because each contrast is priced separately and a shared WITH leg is counted twice, "
                 "h3.POWER_COST_BASIS). The de-duplicated figure for the deployed arm's two V5 contrasts is 336 refits, "
                 "about 50 h serial",
    "consequence": "every one of those contrasts is reported UNDECIDED (no registered power check) in the words "
                   "addendum 4 item 4 fixes, NEVER as a null and never as 'no effect'; the inventory and its cost are "
                   "part of this report so the omission is visible rather than implicit",
}

# --------------------------------------------------------------------------------------------- #
# measured unit costs (the --dry-run estimate)
# --------------------------------------------------------------------------------------------- #
#: measured per-fold seconds (point + intervals), CONFIRMATION_PLAN section 6, from the record ``steps`` of
#: ``evaluation/discovery/<arm>/<design>/s104729/*.json`` and the frozen H3 legs; the frozen-configuration factor 0.2785
#: is measured (``evaluation/h3/decisions/cost_estimate.json -> h3_over_discovery_ratio``)
UNIT_SECONDS: dict[str, float] = {
    "M1@V5__primary__batched_max4": 684.8, "M2@V5__primary__batched_max4": 303.6,
    "M1@V5__strict__batched_max4": 568.1, "M2@V5__strict__batched_max4": 279.8,
    "M1@V5__hno3_only__batched_max4": 680.2, "M2@V5__hno3_only__batched_max4": 283.6,
    "M1@V5PAIR__primary__batched": 686.4, "M2@V5PAIR__primary__batched": 286.9,
    "B6@V5__primary__exact": 9.6, "B6@V2__element__exact": 9.0, "B6:ACT_PERMUTED@V2__element__exact": 2.6,
    "M1@V6__prnd__exact": 0.2785 * 684.8, "M2@V6__prnd__exact": 0.2785 * 303.6, "B8@V6__prnd__exact": 0.2785 * 699.6,
    "B3i": 0.0, "B0": 0.0, "B3x": 0.0, "FLAT": 0.0, "HEAVIER": 0.0,
}
#: the measured parallel efficiency of this machine at 2 workers (discovery: 149.91 h serial in 76.5955 h of wall clock)
PARALLEL_EFFICIENCY_2_WORKERS = 1.96
#: confirmation-half fold counts **per seed**, from ``folds/INDEX.json -> designs.*.by_half.C`` and ``placeholders``
#: (CONFIRMATION_PLAN sections 2.2, 2.3 and 3.1).  A batched file holds the colourings of every seed, so its ``by_half.C``
#: count is divided by the 5 discovery seeds that built it: V5-primary 135/5 = 27 (the plan's "expect 27 +- 1 per seed"),
#: strict 37/5 = 8 (rounded up), HNO3-only 135/5 = 27; V5-PAIR was built for seed 104729 alone, so its 38 IS per seed;
#: the exact V5 and V2 files are seed-INDEPENDENT (111 and 11 folds), and an arm is still refitted on them per withheld
#: seed because its conformal inner folds are drawn with that seed (section 15 resolution).  V6 has no fold design yet:
#: 13 systems, one fold each, built in this run by the section 3.4 component-aware hiding.
FOLD_COUNTS_CONFIRMATION: dict[str, int] = {
    "V5__primary__batched_max4": 27, "V5__strict__batched_max4": 8, "V5__hno3_only__batched_max4": 27,
    "V5PAIR__primary__batched": 38, "V5__primary__exact": 111, "V2__element__exact": 11, "V1__copy__exact": 52,
    "V6__prnd__exact": 13,
}


# --------------------------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------------------------- #

def conf_root(out_root: Path | str) -> Path:
    return Path(out_root) / CONF_REL


def decisions_path(out_root: Path | str) -> Path:
    return conf_root(out_root) / "decisions" / "confirmation.json"


def tables_dir(out_root: Path | str) -> Path:
    return Path(out_root) / "tables"


def report_path(out_root: Path | str) -> Path:
    return Path(out_root) / REPORT_REL


def plan_path(out_root: Path | str) -> Path:
    return Path(out_root) / PLAN_REL


def fold_paths(out_root: Path | str, arm: str, design_dir: str, seed_index: int, fold_id: str) -> tuple[Path, Path]:
    """Prediction parquet and record JSON of one confirmation fold.  The directory carries the OPAQUE seed index, never
    a seed: ``evaluation/confirmation/records/<arm>/<design_dir>/i<index>/<fold_id>.{parquet,json}``."""
    d = conf_root(out_root) / "records" / arm / design_dir / f"i{int(seed_index)}"
    return d / f"{fold_id}.parquet", d / f"{fold_id}.json"


def folds_dir(out_root: Path | str) -> Path:
    """Where the withheld-seed fold files live.  Separate from ``folds/`` so no registered discovery design is touched,
    and named by seed INDEX only."""
    return conf_root(out_root) / "folds"


# --------------------------------------------------------------------------------------------- #
# the withheld seeds
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SeedStore:
    """The 5 withheld seeds, held in memory for the length of one run and never printed.

    ``digest`` is the commitment of section 15, which IS the public name of this object; ``values`` is private and its
    ``repr`` is redacted, so a traceback, a log line, a ``json.dumps(default=str)`` or an f-string cannot leak a seed.
    Code asks for :meth:`indices` and passes an index around; only the fitting call site asks for :meth:`seed`.
    """

    digest: str
    path: str
    values: tuple[int, ...] = field(repr=False)
    verified: bool = True

    def __post_init__(self) -> None:
        if len(self.values) != N_SEEDS:
            raise ValueError(f"the seed store holds {len(self.values)} seeds, section 15 registers {N_SEEDS}")
        if len(set(self.values)) != N_SEEDS:
            raise ValueError("the seed store holds a repeated seed")

    def __repr__(self) -> str:      # pragma: no cover - exercised by the leak test
        return f"SeedStore(digest={self.digest[:12]}..., n={len(self.values)}, values=<withheld>)"

    __str__ = __repr__

    def indices(self) -> tuple[int, ...]:
        """The opaque indices 1..5 every output is labelled with (the store's own order)."""
        return tuple(range(1, len(self.values) + 1))

    def seed(self, index: int) -> int:
        """The seed of ``index`` (1-based).  The ONLY accessor; call it at the fitting site, never in a log line."""
        if int(index) not in self.indices():
            raise ValueError(f"seed index {index!r} is not in {self.indices()}")
        return int(self.values[int(index) - 1])

    def index_of(self, seed: int) -> int:
        return int(self.values.index(int(seed))) + 1

    def public(self) -> dict[str, Any]:
        """What may be written to a file: the commitment, the count, the verdict -- no value."""
        return {"commitment_sha256": self.digest, "n_seeds": len(self.values), "verified_against_commitment":
                bool(self.verified), "seed_indices": list(self.indices()), "values": "withheld",
                "disclosure": SEED_DISCLOSURE}


def committed_digest(root: Path | None = None) -> str:
    p = (paths.G19_ROOT if root is None else Path(root)) / SEED_COMMITMENT_REL
    if not p.exists():
        raise SystemExit(f"refused: no confirmation-seed commitment at {p} (section 15)")
    return p.read_text(encoding="utf-8").strip().split()[0]


def load_seed_store(store_path: Path | str, *, root: Path | None = None, seal=None, prereg_paths=None) -> SeedStore:
    """Verify ``store_path`` against the section 15 commitment and return a :class:`SeedStore`.

    ``seal`` is ``scripts/g19_seal_prereg.py`` as a module (the runner passes it; tests pass a stub exposing
    ``verify_seed_store`` and ``load_committed_seeds``).  A store that does not verify raises: the run never starts on
    unverified seeds, and no seed is printed on either path.
    """
    p = Path(store_path)
    if not p.exists():
        raise SystemExit(f"refused: --seed-store {p} does not exist; the 5 withheld seeds enter only through it")
    if seal is None:      # pragma: no cover - the runner always passes the module
        raise ValueError("load_seed_store needs the seal module (scripts/g19_seal_prereg.py)")
    pp = prereg_paths if prereg_paths is not None else (
        seal.default_paths() if root is None else seal.PreregPaths(root=Path(root), repo_root=paths.REPO_ROOT))
    ok, msg = seal.verify_seed_store(pp, p)
    if not ok:
        raise SystemExit(f"refused: the seed store does not verify against the section 15 commitment: {msg}")
    seeds = tuple(int(s) for s in seal.load_committed_seeds(pp, p))
    return SeedStore(digest=committed_digest(root), path=str(p), values=seeds, verified=True)


def seed_token(index: int) -> str:
    """The opaque label a written file carries where a seed would otherwise appear."""
    return f"i{int(index)}"


def scrub(obj: Any, store: SeedStore) -> Any:
    """Replace every withheld seed -- as an int, and as a substring of any string, including a fold id or a path -- by
    its opaque index token, recursively.

    Every record, table row, decision block and log line goes through this before it is written, so a seed cannot leak
    through a field nobody thought about (``job.seed``, a batched fold id ``b104729_3``, a directory name).
    :func:`scan_for_seed_leak` then re-reads the files and fails the run if one got through anyway: the scrub is the
    intent, the scan is the proof.
    """
    if isinstance(obj, Mapping):
        return {scrub(k, store): scrub(v, store) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        out = [scrub(v, store) for v in obj]
        return type(obj)(out) if not isinstance(obj, set) else set(out)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)) and int(obj) in store.values:
        return seed_token(store.index_of(int(obj)))
    if isinstance(obj, str):
        out = obj
        for s in store.values:
            out = out.replace(str(s), seed_token(store.index_of(s)))
        return out
    return obj


def scan_for_seed_leak(store: SeedStore, targets: Iterable[Path | str], *, extra_text: Iterable[str] = ()
                       ) -> dict[str, Any]:
    """Re-read every file the run wrote (and any captured log text) and refuse if a seed's decimal form appears.

    The check is deliberately crude and total: a withheld seed is a 6-digit integer, so ``str(seed)`` is searched as a
    substring of the raw bytes of every listed file, whatever its format.  Its own report records only the COUNT of
    files scanned and the verdict -- never which seed, never where.
    """
    hits: list[str] = []
    needles = [str(s).encode() for s in store.values]
    n = 0
    for t in targets:
        p = Path(t)
        if not p.exists():
            continue
        for f in ([p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())):
            n += 1
            b = f.read_bytes()
            if any(x in b for x in needles):
                hits.append(str(f))
    for txt in extra_text:
        if any(x.decode() in str(txt) for x in needles):
            hits.append("<log text>")
    return {"files_scanned": n, "ok": not hits, "files_containing_a_withheld_seed": sorted(set(hits)),
            "rule": "no withheld seed's decimal form may appear in any file this run wrote or in its log"}


# --------------------------------------------------------------------------------------------- #
# the plan (decisions/CONFIRMATION_PLAN.md)
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Claim:
    """One frozen claim of the plan: the contrast R19 is applied to on the confirmation half."""

    claim_id: str
    label: str
    family: str
    candidate: str
    comparator: str
    design: str
    margin: float
    stem: str = ""
    comparator_stem: str = ""
    transform: str = ""
    comparator_transform: str = ""
    discovery_point: float | None = None

    @property
    def key(self) -> str:
        return f"{self.label}@{self.design}"


_CLAIM_ROW = re.compile(r"^\|\s*\*\*(C\d)\*\*\s*\|(.*)$")


def parse_claims(text: str) -> list[Claim]:
    """The frozen claims of ``decisions/CONFIRMATION_PLAN.md`` section 2, read from its table rows.

    A row is ``| **C<k>** | <claim> | <family> | <candidate> | <comparator> | <margin> | <designs> | <discovery Delta> |
    <verdict> |``.  Only the fields this run needs are parsed, and every one of them is checked: a claim whose margin,
    design or arms cannot be read is a refusal, never a default.
    """
    out: list[Claim] = []
    for line in text.splitlines():
        m = _CLAIM_ROW.match(line.strip())
        if not m:
            continue
        cid, rest = m.group(1), m.group(2)
        cells = [c.strip() for c in rest.split("|")]
        if len(cells) < 8:
            raise ValueError(f"{cid}: the plan's claim row has {len(cells)} fields, expected at least 8")
        label, family, cand, comp, margin_s, designs, point_s = cells[0], cells[1], cells[2], cells[3], cells[4], cells[5], cells[6]

        def plain(s: str) -> str:
            return re.sub(r"\*\*|`", "", s).strip()

        def clean(s: str) -> str:
            return plain(s).split("—")[0].split("(")[0].strip()

        cand_c, comp_c = clean(cand), clean(comp)
        # the margin cell may carry its NAME as well as its value ("δ5 = 0.10568948790529951", "0.05 (registered: ...)"),
        # and a name contains digits: take the longest numeric literal, never the first one
        nums = re.findall(r"[0-9]*\.?[0-9]+", plain(margin_s).replace(",", ""))
        mm = max(nums, key=len) if nums else None
        if mm is None:
            raise ValueError(f"{cid}: no margin in {margin_s!r}")
        design = "V5" if "V5-primary" in designs or "V5-PAIR" not in designs and "V5" in designs else ""
        for d in ("V5-PAIR", "V2", "V1", "V6"):
            if d in designs:
                design = d
        if not design:
            raise ValueError(f"{cid}: no design in {designs!r}")
        pt = None
        pm = re.search(r"([+-]?[0-9]*\.?[0-9]+)", re.sub(r"\*\*", "", point_s))
        if pm:
            pt = float(pm.group(1))
        cand_arm, cand_tf = (cand_c.split(":") + [""])[:2]
        comp_arm, comp_tf = (comp_c.split(":") + [""])[:2]
        out.append(Claim(claim_id=cid, label=clean(label), family=plain(family), candidate=cand_arm,
                         comparator=comp_arm, design=design.replace("-", ""), margin=float(mm),
                         transform=cand_tf, comparator_transform=comp_tf, discovery_point=pt))
    return out


def read_plan(path: Path | str) -> dict[str, Any]:
    """The plan as this run reads it: its claims, its digest and the ``<= 5`` check of section 15."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"refused: no frozen plan at {p}; section 15 freezes the claims BEFORE the run")
    text = p.read_text(encoding="utf-8")
    claims = parse_claims(text)
    if not claims:
        raise SystemExit(f"refused: {p} freezes no claim (section 15: the plan gives the arm, comparator, metric and "
                         "designs of each claim)")
    if len(claims) > MAX_CLAIMS:
        raise SystemExit(f"refused: {p} freezes {len(claims)} claims; section 15 allows at most {MAX_CLAIMS}")
    return {"path": str(p), "sha256": hashlib.sha256(text.encode("utf-8").replace(b"\r\n", b"\n")).hexdigest(),
            "n_claims": len(claims), "claims": claims,
            "claim_ids": [c.claim_id for c in claims], "claim_keys": [c.key for c in claims]}


def claim_of(plan: Mapping[str, Any], claim_id: str) -> Claim:
    for c in plan["claims"]:
        if c.claim_id == claim_id or c.key == claim_id or c.label == claim_id:
            return c
    raise SystemExit(f"refused: {claim_id!r} is not a frozen claim of the plan ({[c.claim_id for c in plan['claims']]}); "
                     "section 15: 'No sixth claim is added and nothing is re-run'")


# --------------------------------------------------------------------------------------------- #
# the job inventory and the cost estimate
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfJob:
    """One (arm, design file, withheld-seed index) unit of work.  ``seed_index`` is opaque; no seed appears here."""

    purpose: str
    arm: str
    design: str
    stem: str
    seed_index: int
    n_folds: int
    transform: str = ""
    tuned: bool = True
    note: str = ""

    @property
    def key(self) -> str:
        t = f":{self.transform}" if self.transform else ""
        return f"{self.arm}{t}@{self.stem}/i{self.seed_index}"

    @property
    def unit_seconds(self) -> float:
        for k in (f"{self.arm}:{self.transform}@{self.stem}" if self.transform else "", f"{self.arm}@{self.stem}",
                  self.arm):
            if k and k in UNIT_SECONDS:
                return float(UNIT_SECONDS[k])
        return float("nan")


def enumerate_jobs(plan: Mapping[str, Any], *, n_seeds: int = N_SEEDS,
                   fold_counts: Mapping[str, int] | None = None) -> list[ConfJob]:
    """Every job of the single registered run, in the order of ``CONFIRMATION_PLAN`` section 5.

    Claims C1-C3 share one fit set (M1 then M2 on the confirmation V5-primary batched folds); C4 is B6 WITH tuned plus
    ACT_PERMUTED frozen on the V2 folds; R19 item 6's two refits run on ONE seed (addendum 1 item 4 / addendum 5 item
    1); S1(c) adds the V5-PAIR folds; V6 runs once with the frozen configurations.
    """
    fc = dict(FOLD_COUNTS_CONFIRMATION if fold_counts is None else fold_counts)
    ids = {c.claim_id for c in plan["claims"]}
    jobs: list[ConfJob] = []
    seeds = list(range(1, int(n_seeds) + 1))
    if ids & {"C1", "C2", "C3"}:
        for i in seeds:
            for arm in ("M1", "M2"):            # M2 keeps each outer fold's retained M1 hyperparameters (addendum 1 6(d))
                jobs.append(ConfJob("claims C1-C3 (core)", arm, "V5", "V5__primary__batched_max4", i,
                                    fc.get("V5__primary__batched_max4", 0),
                                    note="M2's record verifies against M1's record of the same fold"))
        if "C3" in ids:
            for i in seeds:
                jobs.append(ConfJob("claim C3 comparator B6r0", "B6", "V5", "V5__primary__exact", i,
                                    fc.get("V5__primary__exact", 0), note="B6r0 is read from the B6 fit (rank 0)"))
        for i in seeds:                          # the closed-form comparators and their multi-seed conformal intervals
            for arm in ("B3i", "B0"):
                jobs.append(ConfJob("claim comparators (closed form)", arm, "V5", "V5__primary__exact", i,
                                    fc.get("V5__primary__exact", 0), note="section 15: deterministic point predictions, "
                                                                          "conformal inner folds drawn with each seed"))
        for arm in ("M1", "M2"):                 # item 6 refits, seed 104729 only
            for stem in ("V5__strict__batched_max4", "V5__hno3_only__batched_max4"):
                jobs.append(ConfJob("R19 item 6 refits (discovery seed 104729 only)", arm, "V5", stem, 0,
                                    fc.get(stem, 0), note="addendum 1 item 4 scopes these refits to seed 104729; "
                                                          "addendum 5 item 1 keeps the core run"))
    if "C4" in ids:
        for i in seeds:
            jobs.append(ConfJob("claim C4 (H3 control contrast)", "B6", "V2", "V2__element__exact", i,
                                fc.get("V2__element__exact", 0), transform="WITH"))
            jobs.append(ConfJob("claim C4 (H3 control contrast)", "B6", "V2", "V2__element__exact", i,
                                fc.get("V2__element__exact", 0), transform="ACT_PERMUTED", tuned=False,
                                note="refitted at the WITH run's selected hyperparameters of the same fold (section 11)"))
    for i in seeds:                              # S1(c): M1 then M2 on the batched V5-PAIR folds, yardsticks refitted there
        for arm in ("M1", "M2"):
            jobs.append(ConfJob("S1(c) direction and logSF (V5-PAIR)", arm, "V5PAIR", "V5PAIR__primary__batched", i,
                                fc.get("V5PAIR__primary__batched", 0)))
        jobs.append(ConfJob("S1(c) yardstick refits (closed form)", "B3x", "V5PAIR", "V5PAIR__primary__batched", i,
                            fc.get("V5PAIR__primary__batched", 0), note="B3x and B3i refitted on exactly M2's folds"))
    for i in seeds:                              # V6, once, at the frozen configurations
        for arm in ("M1", "M2", "B8"):
            jobs.append(ConfJob("the single V6 run (S2)", arm, "V6", "V6__prnd__exact", i, fc.get("V6__prnd__exact", 0),
                                tuned=False, note="frozen configurations, no re-tuning (section 3.4)"))
    return jobs


def cost_estimate(jobs: Sequence[ConfJob]) -> dict[str, Any]:
    """Serial hours and wall hours at 2 workers from the measured unit costs (:data:`UNIT_SECONDS`)."""
    rows = []
    for j in jobs:
        s = j.unit_seconds * j.n_folds
        rows.append({"key": j.key, "purpose": j.purpose, "arm": j.arm, "transform": j.transform, "stem": j.stem,
                     "seed_index": j.seed_index, "n_folds": j.n_folds, "unit_seconds": j.unit_seconds,
                     "serial_hours": s / 3600.0 if np.isfinite(s) else float("nan")})
    tot = float(np.nansum([r["serial_hours"] for r in rows]))
    by_purpose: dict[str, float] = {}
    for r in rows:
        if np.isfinite(r["serial_hours"]):
            by_purpose[r["purpose"]] = by_purpose.get(r["purpose"], 0.0) + r["serial_hours"]
    return {"jobs": rows, "n_jobs": len(rows), "n_folds": int(sum(j.n_folds for j in jobs)),
            "serial_hours": tot, "wall_hours_2_workers": tot / PARALLEL_EFFICIENCY_2_WORKERS,
            "by_purpose_serial_hours": dict(sorted(by_purpose.items())),
            "unknown_unit_cost": sorted({r["key"] for r in rows if not np.isfinite(r["unit_seconds"])}),
            "basis": "measured per-fold means (point + intervals) of the discovery and H3 records on this machine; the "
                     "parallel efficiency 1.96x at 2 workers is measured (discovery: 149.91 h serial in 76.5955 h wall). "
                     "Treat +-30 % as the honest band: discovery itself overran its 60 h budget by 28 %",
            "not_in_this_estimate": {"v6_actinide_deltas": V6_ACTINIDE_DELTAS_NOT_RUN["cost_hours_serial"],
                                     "power_check": POWER_CHECK_NOT_RUN["inventory"]}}


# --------------------------------------------------------------------------------------------- #
# the confirmation-half scoring population
# --------------------------------------------------------------------------------------------- #

def confirmation_scored_ids(fold: FI.Fold) -> tuple[str, ...]:
    """The fold's scored rows of the CONFIRMATION half -- the mirror of ``discovery.selection_scored_ids``."""
    return tuple(r for r in fold.scored_row_ids if str(fold.row_half.get(r, fold.half)) == D.CONFIRMATION)


def assert_confirmation_rows(ids: Iterable[str], row_half: Mapping[str, str] | pd.Series, what: str) -> None:
    """Raise unless every row id's registered half is the CONFIRMATION half (this run scores no other row)."""
    ids = list(ids)
    get = row_half.get if isinstance(row_half, Mapping) else (lambda r, d=None: row_half.get(r, d))
    bad = [r for r in ids if get(r, "NA") != D.CONFIRMATION]
    if bad:
        halves = sorted({str(get(r, "NA")) for r in bad})
        raise AssertionError(f"{what}: {len(bad)} scored row(s) outside the confirmation half ({halves}; first "
                             f"{bad[0]!r}); the confirmation run scores the confirmation half only")


def assert_v6_scope(labels: pd.Index, v6_mask: pd.Series, design: str, what: str) -> None:
    """Section 3.4 read in both directions (:data:`READINGS` ``v6_scope``).

    A V6 job scores ONLY ``V6_TARGET_ROWS``; every other design scores NONE of them.  This is the one run where a V6
    row may be scored at all, so the assertion has to be two-sided -- an "is it carved out?" check alone would let the
    carve-out leak into V5 here.
    """
    m = v6_mask.reindex(labels)
    if m.isna().any():
        raise AssertionError(f"{what}: {int(m.isna().sum())} scored row(s) are not in the V6 mask")
    inside = int(m.astype(bool).sum())
    if str(design).upper().replace("-", "") == "V6":
        if inside != len(labels):
            raise AssertionError(f"{what}: a V6 job scored {len(labels) - inside} row(s) outside V6_TARGET_ROWS")
    elif inside:
        raise AssertionError(f"{what}: {inside} V6_TARGET_ROWS row(s) scored under design {design}; section 3.1 carves "
                             "them out of every design and section 3.4 runs V6 once, as its own job")


# --------------------------------------------------------------------------------------------- #
# R19 at confirmation
# --------------------------------------------------------------------------------------------- #

def item4(seed_deltas: Mapping[int, float] | Sequence[float], *, required: int = N_SEEDS_REQUIRED,
          n_seeds: int = N_SEEDS) -> dict[str, Any]:
    """R19 item 4 as registered at confirmation: ``Delta > 0`` in 5 of 5 withheld seeds.

    ``seed_deltas`` is keyed by the OPAQUE seed index (or a sequence in index order).  Fewer than ``n_seeds`` scored
    seeds is a FAIL with the count -- never a pass on the seeds that happen to exist.
    """
    if isinstance(seed_deltas, Mapping):
        items = {int(k): float(v) for k, v in seed_deltas.items()}
    else:
        items = {i + 1: float(v) for i, v in enumerate(seed_deltas)}
    pos = sorted(i for i, v in items.items() if np.isfinite(v) and v > 0)
    ok = len(items) == int(n_seeds) and len(pos) >= int(required)
    return {"item": 4, "name": "seed_sign_agreement", "status": "PASS" if ok else "FAIL",
            "n_seeds_scored": len(items), "n_seeds_expected": int(n_seeds), "n_positive": len(pos),
            "required": int(required), "seed_indices_positive": pos,
            "detail": f"{len(pos)} of {len(items)} scored seeds positive ({n_seeds} expected); confirmation needs "
                      f"{required} of {n_seeds}", "reading": READINGS["item4"]}


def score_claim(claim: Claim, *, point: float, bootstraps: Mapping[str, ET.BootstrapResult],
                seed_deltas: Mapping[int, float], sensitivity_deltas: Mapping[str, float | str],
                deterministic: bool = False) -> dict[str, Any]:
    """R19 (section 8, ``stage='confirmation'``) plus TOST for one frozen claim.

    ``ET.r19`` evaluates items 1-6 with the confirmation reading of item 4 (5 of 5); :func:`item4` is computed beside it
    so the record carries the seed count explicitly, and the two must agree.
    """
    res = ET.r19(design=claim.design, stage="confirmation", point=float(point), margin=float(claim.margin),
                 bootstraps=bootstraps, seed_deltas=[seed_deltas[i] for i in sorted(seed_deltas)],
                 deterministic=bool(deterministic), sensitivity_deltas=sensitivity_deltas, contrast=claim.key)
    i4 = item4(seed_deltas)
    r19_i4 = next((it for it in res.items if it["item"] == 4), {})
    if not deterministic and r19_i4.get("status") != i4["status"]:
        raise AssertionError(f"{claim.key}: transfer.r19 item 4 {r19_i4.get('status')} disagrees with "
                             f"confirmation.item4 {i4['status']}")
    primary = ET.REGISTERED_CLUSTER_UNITS[claim.design][0]
    t = ET.tost(bootstraps[primary], epsilon=EPSILON) if primary in bootstraps else {"verdict": "NOT_COMPUTED"}
    confirmed = res.verdict == "PASS"
    return {"claim_id": claim.claim_id, "claim": claim.key, "family": claim.family, "candidate": claim.candidate,
            "comparator": claim.comparator, "design": claim.design, "margin": claim.margin, "point": float(point),
            "half": D.CONFIRMATION, "stage": "confirmation", "r19_verdict": res.verdict, "items": list(res.items),
            "item4": i4, "tost": t, "confirmed": bool(confirmed),
            "confirmed_rule": "section 15: a claim is confirmed only if it passes R19 with 5 of 5 confirmation seeds",
            "seed_deltas_by_index": {int(k): float(v) for k, v in sorted(seed_deltas.items())},
            "discovery_point_selection_half": claim.discovery_point}


def bh_adjust(p_by_key: Mapping[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg within one family (section 8: printed beside raw p, deciding nothing)."""
    items = [(k, float(v)) for k, v in p_by_key.items() if np.isfinite(v)]
    m = len(items)
    out: dict[str, float] = {k: float("nan") for k in p_by_key}
    if not m:
        return out
    items.sort(key=lambda kv: kv[1])
    prev = 1.0
    for rank in range(m, 0, -1):
        k, p = items[rank - 1]
        prev = min(prev, p * m / rank)
        out[k] = min(1.0, prev)
    return out


# --------------------------------------------------------------------------------------------- #
# S1(c) at confirmation -- the seed mean and its system-cluster bootstrap
# --------------------------------------------------------------------------------------------- #

def seed_mean_system_bootstrap(per_seed_by_system: Mapping[int, Mapping[str, float]], *,
                              n_resamples: int = ET.N_RESAMPLES, seed: int = ET.BOOTSTRAP_SEED) -> dict[str, Any]:
    """The S1(c) / S2(a) confirmation statistic: the mean over the withheld seeds of each seed's system-macro Delta,
    with a system-cluster percentile bootstrap in which **the same resampled systems are applied to every seed**.

    ``per_seed_by_system`` maps the opaque seed index to {system: per-system Delta}.  Systems must be the same set in
    every seed (they are: the confirmation half's scored systems do not depend on a colouring).
    """
    idx = sorted(per_seed_by_system)
    if not idx:
        raise ValueError("no seed given")
    systems = sorted(per_seed_by_system[idx[0]])
    for i in idx:
        if sorted(per_seed_by_system[i]) != systems:
            raise ValueError(f"seed index {i} carries a different system set")
    mat = np.array([[float(per_seed_by_system[i][s]) for s in systems] for i in idx], dtype=float)   # seeds x systems
    per_seed = np.nanmean(mat, axis=1)
    point = float(np.nanmean(per_seed))
    rng = np.random.default_rng(int(seed))
    n = len(systems)
    draws = np.empty(int(n_resamples), dtype=float)
    for b in range(int(n_resamples)):
        take = rng.integers(0, n, size=n)
        draws[b] = float(np.nanmean(np.nanmean(mat[:, take], axis=1)))
    lo, hi = (float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5)))
    return {"point": point, "per_seed": {int(i): float(v) for i, v in zip(idx, per_seed)},
            "percentile_95": [lo, hi], "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
            "n_systems": n, "n_seeds": len(idx), "n_resamples": int(n_resamples), "bootstrap_seed": int(seed),
            "construction": "system-cluster percentile bootstrap of the SEED MEAN, the same resampled systems applied "
                            "to every seed (section 9 S1(c) as redefined; BCa is not registered here)",
            "reading": READINGS["s1c_seed_combination"]}


def s1c_confirmation(direction_by_yardstick: Mapping[str, Mapping[int, Mapping[str, float]]], *,
                     logsf_gain: Mapping[str, float], selection_min_delta: float | None,
                     gamma5: float = GAMMA5, eta5: float = ETA5) -> dict[str, Any]:
    """S1(c) at confirmation: ``min_Y Delta_Y >= gamma5``, every ``Delta_Y``'s interval excluding 0, ``Delta_Y > 0`` in
    5 of 5 seeds, the logSF MAE gains over FLAT and the B3i-derived value ``>= eta5``, and the selection-half
    counterweight ``min_Y Delta_Y >= -0.02`` (already satisfied in discovery)."""
    per: dict[str, Any] = {}
    for y, per_seed in sorted(direction_by_yardstick.items()):
        st = seed_mean_system_bootstrap(per_seed)
        st["item4"] = item4({i: v for i, v in st["per_seed"].items()})
        per[y] = st
    mins = {y: st["point"] for y, st in per.items()}
    min_y = min(mins.values()) if mins else float("nan")
    binding = min(mins, key=lambda k: mins[k]) if mins else None
    dir_ok = bool(mins and min_y >= gamma5 and all(st["excludes_zero"] for st in per.values())
                  and all(st["item4"]["status"] == "PASS" for st in per.values()))
    mag = {k: float(v) for k, v in logsf_gain.items()}
    mag_ok = bool(mag) and all(np.isfinite(v) and v >= eta5 for v in mag.values())
    cw_ok = None if selection_min_delta is None else bool(float(selection_min_delta) >= -0.02)
    verdict = "PASS" if (dir_ok and mag_ok and cw_ok) else ("UNDECIDED" if cw_ok is None else "FAIL")
    return {"per_yardstick": per, "min_delta": min_y, "binding_yardstick": binding, "gamma5": gamma5, "eta5": eta5,
            "direction_pass": dir_ok, "logsf_gain": mag, "logsf_pass": mag_ok,
            "counterweight_selection_half_min_delta": selection_min_delta, "counterweight_pass": cw_ok,
            "verdict": verdict, "reading": READINGS["s1c_seed_combination"],
            "rule": "section 9 S1(c) as redefined 2026-09-15 and its addendum-2 confirmation rule; S1(c) passes only if "
                    "both halves hold, and if V5-PAIR were dropped S1 would be UNDECIDED, never passed"}


# --------------------------------------------------------------------------------------------- #
# S2 -- the single V6 run
# --------------------------------------------------------------------------------------------- #

def s2a_sign_count(observed_median: Mapping[str, float], predicted_median: Mapping[str, float], *,
                   n_systems: int = S2A_N_SYSTEMS, required: int = S2A_MIN_SIGN_SYSTEMS) -> dict[str, Any]:
    """S2(a) first half: the sign of the per-system median predicted logSF_Nd/Pr equals the observed sign in >= 11 of
    the 13 systems.  A zero or non-finite median on either side does NOT agree (:data:`READINGS` ``s2a``)."""
    systems = sorted(set(observed_median) | set(predicted_median))
    rows, agree = [], []
    for s in systems:
        o, p = float(observed_median.get(s, float("nan"))), float(predicted_median.get(s, float("nan")))
        ok = bool(np.isfinite(o) and np.isfinite(p) and o != 0 and p != 0 and np.sign(o) == np.sign(p))
        rows.append({"system": s, "observed_median_logsf": o, "predicted_median_logsf": p, "sign_agrees": ok})
        if ok:
            agree.append(s)
    return {"n_systems_scored": len(systems), "n_systems_expected": int(n_systems), "n_sign_agree": len(agree),
            "required": int(required), "systems_agreeing": agree,
            "status": "PASS" if (len(systems) == int(n_systems) and len(agree) >= int(required)) else "FAIL",
            "per_system": rows, "reading": READINGS["s2a"]}


def s2b_magnitude(*, logsf_mae: float, flat_logsf_mae: float, lookup_logsf_mae: float,
                  gain_interval_excludes_zero: bool | None, logd_macro_mae: float, v5_lookup_comparator_mae: float,
                  margin: float = S2B_MARGIN) -> dict[str, Any]:
    """S2(b): logSF MAE <= FLAT - 0.02 with the 13-system percentile interval of the gain over FLAT excluding 0, also
    <= the lookup-derived logSF MAE, and the macro log D MAE of the hidden Pr / Nd rows <= the V5 lookup comparator."""
    gain = float(flat_logsf_mae) - float(logsf_mae)
    beats_flat = bool(np.isfinite(gain) and gain >= float(margin))
    beats_lookup = bool(np.isfinite(logsf_mae) and logsf_mae <= float(lookup_logsf_mae))
    logd_ok = bool(np.isfinite(logd_macro_mae) and logd_macro_mae <= float(v5_lookup_comparator_mae))
    parts = {"logsf_mae": float(logsf_mae), "flat_logsf_mae": float(flat_logsf_mae), "gain_over_flat": gain,
             "margin": float(margin), "beats_flat_by_margin": beats_flat,
             "gain_interval_excludes_zero": gain_interval_excludes_zero,
             "lookup_derived_logsf_mae": float(lookup_logsf_mae), "at_most_lookup": beats_lookup,
             "logd_macro_mae_hidden_pr_nd": float(logd_macro_mae),
             "v5_lookup_comparator_mae": float(v5_lookup_comparator_mae), "logd_at_most_lookup": logd_ok}
    ok = beats_flat and beats_lookup and logd_ok and bool(gain_interval_excludes_zero)
    parts["status"] = "PASS" if ok else ("UNDECIDED" if gain_interval_excludes_zero is None else "FAIL")
    return parts


def coverage_bands(coverage: Mapping[str, float], bands: Mapping[str, tuple[float, float]], *,
                   min_95: float | None = None) -> dict[str, Any]:
    """S1(d) / S2(c): every registered coverage band, with the value that decides each one."""
    rows, fails = [], []
    for lvl, (lo, hi) in sorted(bands.items()):
        v = float(coverage.get(lvl, float("nan")))
        ok = bool(np.isfinite(v) and lo <= v <= hi)
        rows.append({"level": lvl, "coverage": v, "band": [lo, hi], "in_band": ok})
        if not ok:
            fails.append(f"{lvl}: {v:.4g} outside [{lo}, {hi}]")
    if min_95 is not None:
        v = float(coverage.get("95", float("nan")))
        ok = bool(np.isfinite(v) and v >= float(min_95))
        rows.append({"level": "95", "coverage": v, "band": [float(min_95), 1.0], "in_band": ok})
        if not ok:
            fails.append(f"95: {v:.4g} below {min_95}")
    return {"per_level": rows, "status": "FAIL" if fails else "PASS", "detail": "; ".join(fails) or "every band met"}


def s1d_calibration(coverage: Mapping[str, float], by_category: Mapping[str, Mapping[str, float]] | None = None
                    ) -> dict[str, Any]:
    """S1(d): the three macro bands over the confirmation half's V5-primary cells, plus 80 % in [0.65, 0.92] in every
    domain-status category with >= 20 scored cells (a category with fewer is printed with its count and decides
    nothing)."""
    out = {"macro": coverage_bands(coverage, S1D_BANDS), "by_category": [], "min_cells": S1D_CATEGORY_MIN_CELLS}
    fails = [] if out["macro"]["status"] == "PASS" else ["macro bands"]
    for cat, c in sorted((by_category or {}).items()):
        n = int(c.get("n_cells", 0))
        v = float(c.get("80", float("nan")))
        decides = n >= S1D_CATEGORY_MIN_CELLS
        ok = bool(np.isfinite(v) and S1D_CATEGORY_80[0] <= v <= S1D_CATEGORY_80[1])
        out["by_category"].append({"category": cat, "n_cells": n, "coverage_80": v, "band": list(S1D_CATEGORY_80),
                                   "decides": decides, "in_band": ok})
        if decides and not ok:
            fails.append(f"{cat}: 80 % {v:.4g}")
    out["status"] = "FAIL" if fails else "PASS"
    out["detail"] = "; ".join(fails) or "every registered band met"
    return out


# --------------------------------------------------------------------------------------------- #
# the idempotence lock (section 15: "nothing is re-run")
# --------------------------------------------------------------------------------------------- #

def lock_verdict(out_root: Path | str, *, resume: bool, code_digest: str, claim_ids: Sequence[str]) -> dict[str, Any]:
    """Whether this invocation may proceed (:data:`READINGS` ``idempotence``).

    First run: ``decisions/confirmation.json`` absent -> proceed.  Second run: refuse, unless ``--resume``, and then
    only to COMPLETE unfitted folds -- the recorded code digest must be the live one and the claim list must not grow,
    so no claim is ever re-scored under different code and no sixth claim appears.
    """
    p = decisions_path(out_root)
    if not p.exists():
        return {"ok": True, "mode": "first_run", "decisions": str(p), "reading": READINGS["idempotence"]}
    prev = json.loads(p.read_text(encoding="utf-8"))
    prev_code = str(prev.get("code_digest", ""))
    prev_ids = list(prev.get("claim_ids") or [])
    if not resume:
        return {"ok": False, "mode": "already_run", "decisions": str(p),
                "reason": f"{p} exists: the confirmation run is registered to happen ONCE (section 15: 'Whatever "
                          "confirmation returns is the result. No sixth claim is added and nothing is re-run'). Pass "
                          "--resume to COMPLETE unfitted folds; nothing else is permitted.",
                "reading": READINGS["idempotence"]}
    if prev_code and prev_code != str(code_digest):
        return {"ok": False, "mode": "resume_refused_code_changed", "decisions": str(p),
                "recorded_code_digest": prev_code, "live_code_digest": str(code_digest),
                "reason": "--resume may only complete unfitted folds: the code digest of the recorded run differs from "
                          "the live one, so scoring a claim now would score it under different code. Nothing is "
                          "re-scored and nothing is deleted.", "reading": READINGS["idempotence"]}
    extra = sorted(set(claim_ids) - set(prev_ids))
    if extra:
        return {"ok": False, "mode": "resume_refused_claims_grew", "decisions": str(p), "new_claims": extra,
                "reason": f"--resume may not widen the claim list; {extra} are not in the recorded run ({prev_ids}).",
                "reading": READINGS["idempotence"]}
    return {"ok": True, "mode": "resume", "decisions": str(p), "recorded_code_digest": prev_code,
            "claim_ids": prev_ids, "may_only": "complete unfitted folds", "reading": READINGS["idempotence"]}


# --------------------------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------------------------- #

def gate(out_root: Path | str, *, seed_store: SeedStore | None, code_digest: str, resume: bool = False,
         check: Any = None, digests: Any = None, expect_addenda: int | None = None,
         plan: Mapping[str, Any] | None = None, registry_path: Path | None = None) -> dict[str, Any]:
    """The five gates of the confirmation runner, in order, each refusing with what is missing:

    (a) the seal check and the registered below-footer digest of stage ``confirmation``
        (``registry.refuse_unless_sealed``);
    (b) the registry holds that stage and this code is the code it was registered with
        (``registry.refuse_unless_writable``);
    (c) ``decisions/CONFIRMATION_PLAN.md`` is present and freezes at most 5 claims;
    (d) a ``--seed-store`` that verifies against the committed digest;
    (e) the idempotence lock.
    """
    out_root = Path(out_root)
    prereg = REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=expect_addenda, path=registry_path)
    entry = REG.registered(STAGE, registry_path)
    if entry is None:
        raise SystemExit(f"refused: the registry holds no '{STAGE}' stage; register it before the run "
                         f"(python -m gen19ct.evaluation.registry register --stage {STAGE} --note ...)")
    writable = REG.refuse_unless_writable(STAGE, str(code_digest), registry_path)
    pl = dict(read_plan(plan_path(out_root))) if plan is None else dict(plan)
    if seed_store is None:
        raise SystemExit("refused: the confirmation run needs --seed-store PATH; the 5 withheld seeds enter only "
                         "through it (section 15)")
    if not seed_store.verified or seed_store.digest != committed_digest(out_root):
        raise SystemExit("refused: the seed store does not verify against manifests/confirmation_seeds_sha256.txt")
    lock = lock_verdict(out_root, resume=resume, code_digest=code_digest,
                        claim_ids=[c.claim_id for c in pl["claims"]])
    if not lock["ok"]:
        raise SystemExit("refused: " + str(lock.get("reason")))
    return {"prereg": prereg, "registry_entry": entry, "writable": writable, "lock": lock,
            "plan": {k: v for k, v in pl.items() if k != "claims"},
            "claims": [c.claim_id for c in pl["claims"]], "seed_store": seed_store.public(),
            "stage": STAGE, "code_digest": str(code_digest), "readings": dict(READINGS),
            "not_run": {"v6_actinide_deltas": V6_ACTINIDE_DELTAS_NOT_RUN, "power_check": POWER_CHECK_NOT_RUN}}


# --------------------------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------------------------- #

def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def write_decisions(out_root: Path | str, body: Mapping[str, Any]) -> Path:
    p = decisions_path(out_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, {"schema": DECISION_SCHEMA, "written_utc": _now(), **dict(body)})
    return p


def confirmation_tables(decisions: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """``tables/confirmation_*.csv`` -- one row per claim, per R19 item and per NOT_RUN entry."""
    claims = decisions.get("claims") or []
    rows = []
    for c in claims:
        sd = c.get("seed_deltas_by_index") or {}
        rows.append({"claim_id": c.get("claim_id"), "claim": c.get("claim"), "family": c.get("family"),
                     "design": c.get("design"), "candidate": c.get("candidate"), "comparator": c.get("comparator"),
                     "half": c.get("half"), "margin": c.get("margin"), "point": c.get("point"),
                     "r19_verdict": c.get("r19_verdict"), "confirmed": c.get("confirmed"),
                     "n_seeds_positive": (c.get("item4") or {}).get("n_positive"),
                     "n_seeds_scored": (c.get("item4") or {}).get("n_seeds_scored"),
                     "tost_verdict": (c.get("tost") or {}).get("verdict"),
                     "seed_deltas_by_index": "; ".join(f"i{k}={v:.6g}" for k, v in sorted(sd.items())),
                     "discovery_point_selection_half": c.get("discovery_point_selection_half")})
    items = [{"claim_id": c.get("claim_id"), "claim": c.get("claim"), "item": it.get("item"), "name": it.get("name"),
              "status": it.get("status"), "detail": it.get("detail")}
             for c in claims for it in (c.get("items") or [])]
    nr = [{"what": k, **{kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float))}}
          for k, v in sorted((decisions.get("not_run") or {}).items())]
    return {"confirmation_claims": pd.DataFrame(rows), "confirmation_r19_items": pd.DataFrame(items),
            "confirmation_not_run": pd.DataFrame(nr)}


def confirmation_report(decisions: Mapping[str, Any]) -> str:
    """``decisions/CONFIRMATION.md`` -- the result, the seeds as a VERDICT only, and what was not run."""
    ss = decisions.get("seed_store") or {}
    lines = ["# CONFIRMATION -- the single registered confirmation run (pre-registration section 15)", "",
             f"*Written {decisions.get('written_utc', _now())} by `scripts/g19_run_confirmation.py`. "
             "POST-HOC addendum 5 item 1 makes this the CORE run: the frozen claims on the confirmation half with the 5 "
             "withheld seeds, the single V6 run with S2(a)-(c), S1(c) and S1(d). Whatever it returns is the result "
             "(section 15).*", "",
             "## The withheld seeds", "",
             f"- commitment (section 15): `{ss.get('commitment_sha256', '?')}`",
             f"- `scripts/g19_seal_prereg.py --verify-seeds`: **"
             f"{'VERIFIED' if ss.get('verified_against_commitment') else 'NOT VERIFIED'}** "
             f"({ss.get('n_seeds', '?')} seeds).",
             f"- the VALUES are not printed here. {ss.get('disclosure', SEED_DISCLOSURE)}", "",
             "## The frozen claims on the confirmation half", "",
             "| claim | design | Delta | margin | R19 | item 4 (seeds) | TOST | confirmed |", "|---|---|---|---|---|---|---|---|"]
    for c in decisions.get("claims") or []:
        i4 = c.get("item4") or {}
        lines.append(f"| {c.get('claim_id')} {c.get('claim')} | {c.get('design')} | "
                     f"{c.get('point', float('nan')):+.6g} | {c.get('margin')} | **{c.get('r19_verdict')}** | "
                     f"{i4.get('n_positive')} of {i4.get('n_seeds_scored')} | "
                     f"{(c.get('tost') or {}).get('verdict')} | **{'yes' if c.get('confirmed') else 'no'}** |")
    for key, title in (("s1", "## S1 -- the scientific claim (H1, V5)"), ("s2", "## S2 -- the applied claim (V6, once)")):
        block = decisions.get(key)
        if block:
            lines += ["", title, "", "```json", json.dumps(block, indent=1, default=str)[:4000], "```"]
    lines += ["", "## What this run did NOT do, with its cost and its consequence", ""]
    for k, v in sorted((decisions.get("not_run") or {}).items()):
        lines += [f"- **{k}: {v.get('status', NOT_RUN)}** -- {v.get('what')}.",
                  f"  - not run by: {v.get('not_run_by')}",
                  f"  - cost: {v.get('cost_hours_serial', v.get('inventory'))}",
                  f"  - consequence: {v.get('consequence')}"]
    leak = decisions.get("seed_leak_scan") or {}
    lines += ["", "## Seed hygiene", "",
              f"- files scanned for a withheld seed's decimal form: {leak.get('files_scanned', '?')}; "
              f"verdict **{'clean' if leak.get('ok') else 'LEAK'}**"
              + ("" if leak.get("ok") else f" -- {leak.get('files_containing_a_withheld_seed')}"), ""]
    return "\n".join(lines)
