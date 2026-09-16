"""``folds/io.py`` -- the common outer / inner fold representation of every Gen19 hold-out design.

A :class:`Fold` names the rows a design REMOVES from training and the subset of them that is SCORED:

``design``, ``variant``, ``scheme``  e.g. ``("V5", "primary", "exact")``, ``("V1", "copy", "grouped10")``
``fold_id``                         unique inside one design file
``half``                            ``"S"`` / ``"C"`` (selection / confirmation, ``feasibility_halves.csv``) or
                                    ``"NA"`` (no half, or scored units of both halves in one fold -- the per-row
                                    half is then carried by ``row_half``)
``seed``                            the discovery seed that drew the fold, ``None`` for deterministic designs
``hidden_row_ids``                  ``canonical_measurement_id`` of every row removed from training (sorted)
``scored_row_ids``                  the hidden rows that are scored: known metal state, not ``Sr(III)`` and not in
                                    ``V6_TARGET_ROWS`` (:func:`scorable_mask`; ``registered.assert_not_scored``
                                    is run on them before a fold is written) -- plus the design's own rule
``unit_type`` / ``units``           the scoring unit(s) of the fold (cell, publication group, metal state, ...)
``batch_id``                        batch label of batched designs, else ``None``
``row_unit`` / ``row_half``         per hidden row: the unit that hid it and that unit's half
``meta``                            design-specific facts (carve-out, publication group, cell-pair class, ...)

Training rows are never stored: they are the universe (MODEL rows, or an outer training set for an inner
fold) minus ``hidden_row_ids`` -- :func:`training_ids` recomputes and asserts the partition.

Hashes.  ``fold_hash`` = SHA-256 of the sorted ``role,row_id`` lines (LF, trailing newline); the design hash is
SHA-256 of the sorted ``fold_id,fold_hash`` lines; ``assignment_sha256`` is the SHA-256 of the LF-written
assignment CSV ``row_id,fold_id,role,unit,half`` sorted by ``row_id`` then ``fold_id`` (the pre-registration
section 16 "fold hash"), computed in memory from the same records the parquet stores.

Serialisation (:func:`write_design` / :func:`read_design`): one parquet per design + variant + scheme in long
format (``fold_id, row_id, role, unit, half``; roles ``hidden_scored`` / ``hidden_unscored``) and one JSON index
with the per-fold counts, units, meta and hashes.  :func:`read_design` re-derives every fold hash and refuses a
file that does not reproduce its index.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.data import leakage as L
from gen19ct.folds import registered as FR

SCHEMA = "gen19.folds.v1"
ROW_ID = "canonical_measurement_id"
ROLES: tuple[str, ...] = ("hidden_scored", "hidden_unscored")
HALF_VALUES: tuple[str, ...] = ("S", "C", "NA")
#: brief / pre-registration section 15: the programme's public split seeds
DISCOVERY_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
#: pre-registration section 2: the implausible oxidation state kept in training and never scored
UNSCORED_STATES: tuple[str, ...] = ("Sr(III)",)
#: pre-registration section 2: every publication unit of folds, clusters and inner folds
GROUP_COL = "group_cross_publication_copy"
PUB_COMPONENTS_CSV = paths.DATA_AUDIT_DIR / "leakage_publication_components.csv"
HALVES_CSV = paths.DATA_AUDIT_DIR / "feasibility_halves.csv"
#: pre-registration section 2 guard settings
NEAR_DUP_SIG = 6
NEAR_DUP_VALUE_TOL = L.COPY_TOLERANCE_LOG_D
#: independent random streams drawn from one seed: ``numpy.random.SeedSequence([seed, tag, ...])``
STREAM_TAGS: dict[str, int] = {"V0": 0, "V1_grouped": 1, "V5_batch": 5, "V5P_batch": 6, "V5PAIR_batch": 7,
                               "inner_V1": 11, "inner_V2": 12, "inner_V5_groups": 15, "inner_V5_subsample": 16,
                               "inner_V5_batch": 17}
#: columns the guards, the hiding rules and the fold builders read (the slim frame of :func:`slim_frame`)
FRAME_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys((
    ROW_ID, "log_D", "duplicate_group_id", "g19_publication_id", "g19_metal", "g19_ox", "g19_metal_state",
    "extractant_system_key", "acid_primary") + L.KEY_CATEGORICAL + L.KEY_NUMERIC))


# --------------------------------------------------------------------------------------------- #
# the Fold record
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Fold:
    design: str
    variant: str
    scheme: str
    fold_id: str
    half: str
    seed: int | None
    hidden_row_ids: tuple[str, ...]
    scored_row_ids: tuple[str, ...]
    unit_type: str
    units: tuple[str, ...]
    batch_id: str | None = None
    row_unit: Mapping[str, str] = field(default_factory=dict, compare=False, repr=False)
    row_half: Mapping[str, str] = field(default_factory=dict, compare=False, repr=False)
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.half not in HALF_VALUES:
            raise ValueError(f"{self.fold_id}: half {self.half!r} not in {HALF_VALUES}")
        if list(self.hidden_row_ids) != sorted(set(self.hidden_row_ids)):
            raise ValueError(f"{self.fold_id}: hidden_row_ids must be sorted and unique")
        if list(self.scored_row_ids) != sorted(set(self.scored_row_ids)):
            raise ValueError(f"{self.fold_id}: scored_row_ids must be sorted and unique")
        if not set(self.scored_row_ids) <= set(self.hidden_row_ids):
            raise ValueError(f"{self.fold_id}: a scored row is not hidden")

    @property
    def unscored_row_ids(self) -> tuple[str, ...]:
        s = set(self.scored_row_ids)
        return tuple(r for r in self.hidden_row_ids if r not in s)

    @property
    def fold_hash(self) -> str:
        return fold_hash(self.hidden_row_ids, self.scored_row_ids)

    def role_of(self) -> dict[str, str]:
        s = set(self.scored_row_ids)
        return {r: ("hidden_scored" if r in s else "hidden_unscored") for r in self.hidden_row_ids}

    def records(self) -> list[tuple[str, str, str, str, str]]:
        """Long-format ``(fold_id, row_id, role, unit, half)`` rows, sorted by row id."""
        role = self.role_of()
        default_unit = self.units[0] if self.units else ""
        return [(self.fold_id, r, role[r], str(self.row_unit.get(r, default_unit)), str(self.row_half.get(r, self.half)))
                for r in self.hidden_row_ids]


def make_fold(*, design: str, variant: str, scheme: str, fold_id: str, half: str, seed: int | None,
              hidden: Iterable[str], scored: Iterable[str], unit_type: str, units: Sequence[str],
              batch_id: str | None = None, row_unit: Mapping[str, str] | None = None,
              row_half: Mapping[str, str] | None = None, meta: Mapping[str, Any] | None = None) -> Fold:
    """:class:`Fold` with sorted, de-duplicated id tuples (plain ``str`` ids)."""
    h = tuple(sorted({str(x) for x in hidden}))
    s = tuple(sorted({str(x) for x in scored}))
    return Fold(design=design, variant=variant, scheme=scheme, fold_id=str(fold_id), half=half,
                seed=None if seed is None else int(seed), hidden_row_ids=h, scored_row_ids=s, unit_type=unit_type,
                units=tuple(str(u) for u in units), batch_id=batch_id, row_unit=dict(row_unit or {}),
                row_half=dict(row_half or {}), meta=dict(meta or {}))


# --------------------------------------------------------------------------------------------- #
# hashes
# --------------------------------------------------------------------------------------------- #

def _sha(lines: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in lines).encode("utf-8")).hexdigest()


def fold_hash(hidden_row_ids: Iterable[str], scored_row_ids: Iterable[str]) -> str:
    """SHA-256 of the sorted ``role,row_id`` lines of one fold."""
    s = set(scored_row_ids)
    return _sha(sorted(f"{'hidden_scored' if r in s else 'hidden_unscored'},{r}" for r in set(hidden_row_ids)))


def design_hash(folds: Sequence[Fold]) -> str:
    """SHA-256 of the sorted ``fold_id,fold_hash`` lines of one design file."""
    ids = [f.fold_id for f in folds]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fold_id in one design")
    return _sha(sorted(f"{f.fold_id},{f.fold_hash}" for f in folds))


def assignment_frame(folds: Sequence[Fold]) -> pd.DataFrame:
    recs = [r for f in folds for r in f.records()]
    return pd.DataFrame(recs, columns=["fold_id", "row_id", "role", "unit", "half"])


def assignment_sha256(folds: Sequence[Fold]) -> str:
    """SHA-256 of the LF assignment CSV ``row_id,fold_id,role,unit,half`` sorted by row id, then fold id."""
    a = assignment_frame(folds).sort_values(["row_id", "fold_id"], kind="mergesort")
    head = "row_id,fold_id,role,unit,half"
    body = (a["row_id"] + "," + a["fold_id"] + "," + a["role"] + "," + a["unit"].map(_csv_field) + "," + a["half"])
    return _sha([head, *body.tolist()])


def _csv_field(v: str) -> str:
    v = str(v)
    return '"' + v.replace('"', '""') + '"' if any(ch in v for ch in ',"\n') else v


# --------------------------------------------------------------------------------------------- #
# corpus helpers shared by the builders
# --------------------------------------------------------------------------------------------- #

def seed_rng(seed: int, stream: str, *extra: int) -> np.random.Generator:
    """The random stream ``stream`` of ``seed`` (``SeedSequence([seed, STREAM_TAGS[stream], *extra])``)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), STREAM_TAGS[stream], *[int(e) for e in extra]]))


def slim_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns the fold builders and guards read, same index; row ids must be unique."""
    missing = [c for c in FRAME_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"fold frame: columns missing {missing}")
    out = frame[list(FRAME_COLUMNS) + [c for c in (GROUP_COL,) if c in frame.columns]].copy()
    if not out[ROW_ID].is_unique or not out.index.is_unique:
        raise ValueError("fold frame: canonical_measurement_id and the index must be unique")
    return out


@lru_cache(maxsize=1)
def _group_map() -> dict[str, str]:
    pc = pd.read_csv(PUB_COMPONENTS_CSV, usecols=["g19_publication_id", GROUP_COL])
    return dict(zip(pc["g19_publication_id"], pc[GROUP_COL]))


def publication_groups(frame: pd.DataFrame, column: str = GROUP_COL) -> pd.Series:
    """Registered publication group of every row (``leakage_publication_components.csv``); raises on a gap."""
    if column in frame.columns and column == GROUP_COL:
        g = frame[column]
    elif column == GROUP_COL:
        g = frame["g19_publication_id"].map(_group_map())
    else:
        pc = pd.read_csv(PUB_COMPONENTS_CSV, usecols=["g19_publication_id", column])
        g = frame["g19_publication_id"].map(dict(zip(pc["g19_publication_id"], pc[column])))
    if g.isna().any():
        raise RuntimeError(f"{column}: {int(g.isna().sum())} row(s) without a publication group")
    return g.astype(str)


@lru_cache(maxsize=1)
def _halves_table() -> pd.DataFrame:
    return pd.read_csv(HALVES_CSV)


def registered_halves(design: str) -> dict[str, str]:
    """``unit -> 'S'|'C'`` of ``feasibility_halves.csv`` for ``design`` in ``{"V1", "V5_system", "V2_state"}``."""
    h = _halves_table()
    sub = h[h["design"] == design]
    if sub.empty:
        raise KeyError(f"no halves for design {design!r}")
    return dict(zip(sub["unit"].astype(str), sub["half"].astype(str)))


@lru_cache(maxsize=1)
def _registered_v6_ids() -> frozenset[str]:
    from gen19ct.data import load
    m = load.load_model_rows(copy=False)
    mask = FR.v6_target_mask(m, FR.v6_system_set(m))
    return frozenset(m.loc[mask.to_numpy(dtype=bool), ROW_ID].astype(str))


def registered_v6_ids() -> frozenset[str]:
    """``V6_TARGET_ROWS`` as ``canonical_measurement_id`` strings, defined once on the full MODEL tier."""
    return _registered_v6_ids()


def v6_mask_for(frame: pd.DataFrame, v6_ids: Iterable[str] | None = None) -> pd.Series:
    """Boolean ``V6_TARGET_ROWS`` mask on ``frame.index`` (ids from :func:`registered_v6_ids` by default)."""
    ids = registered_v6_ids() if v6_ids is None else frozenset(str(x) for x in v6_ids)
    return frame[ROW_ID].astype(str).isin(ids)


def scorable_mask(frame: pd.DataFrame, v6_mask: pd.Series) -> pd.Series:
    """Rows a fold may score: a known metal state, not an unscored state (Sr(III)), not ``V6_TARGET_ROWS``."""
    st = frame[SG.METAL_COL]
    return st.notna() & ~st.isin(UNSCORED_STATES) & ~v6_mask.reindex(frame.index).fillna(False).astype(bool)


def assert_scoring_clean(fold: Fold, frame: pd.DataFrame, v6_mask: pd.Series) -> None:
    """``registered.assert_not_scored`` on the fold's scored rows, plus the Sr(III) / X(?) exclusions."""
    if not fold.scored_row_ids:
        return
    idx = frame.index[frame[ROW_ID].isin(set(fold.scored_row_ids))]
    if len(idx) != len(fold.scored_row_ids):
        raise AssertionError(f"{fold.fold_id}: scored ids missing from the frame")
    FR.assert_not_scored(idx, v6_mask, f"{fold.design}/{fold.variant}/{fold.fold_id}")
    st = frame.loc[idx, SG.METAL_COL]
    if st.isna().any() or st.isin(UNSCORED_STATES).any():
        raise AssertionError(f"{fold.fold_id}: scores an X(?) or Sr(III) row")


def training_ids(fold: Fold, universe_ids: Iterable[str]) -> np.ndarray:
    """Universe ids minus the hidden ids (sorted); asserts every hidden id is in the universe."""
    uni = np.asarray(sorted({str(x) for x in universe_ids}), dtype=object)
    hid = set(fold.hidden_row_ids)
    keep = np.fromiter((u not in hid for u in uni), dtype=bool, count=len(uni))
    if int((~keep).sum()) != len(hid):
        raise AssertionError(f"{fold.fold_id}: {len(hid) - int((~keep).sum())} hidden id(s) outside the universe")
    return uni[keep]


def training_frame(fold: Fold, frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` without the fold's hidden rows (partition asserted)."""
    hidden = frame[ROW_ID].astype(str).isin(set(fold.hidden_row_ids))
    if int(hidden.sum()) != len(fold.hidden_row_ids):
        raise AssertionError(f"{fold.fold_id}: hidden ids missing from the frame")
    return frame[~hidden.to_numpy()]


def index_of(frame: pd.DataFrame, ids: Iterable[str]) -> pd.Index:
    ids = set(str(x) for x in ids)
    return frame.index[frame[ROW_ID].astype(str).isin(ids).to_numpy()]


# --------------------------------------------------------------------------------------------- #
# seeded helpers used by several designs
# --------------------------------------------------------------------------------------------- #

def greedy_balance(weights: Mapping[str, float], n_bins: int, rng: np.random.Generator) -> dict[str, int]:
    """Seeded greedy balance: units in a seeded random order, each to the currently lightest bin (ties to the
    lowest bin index).  Every unit stays whole; the seed changes the assignment, the balance stays within
    the heaviest unit."""
    units = sorted(weights)
    order = [units[i] for i in rng.permutation(len(units))]
    load = np.zeros(int(n_bins), dtype=float)
    out: dict[str, int] = {}
    for u in order:
        b = int(np.argmin(load))
        out[u] = b
        load[b] += float(weights[u])
    return out


# --------------------------------------------------------------------------------------------- #
# summaries and serialisation
# --------------------------------------------------------------------------------------------- #

def _rel(p: Path) -> str:
    """Repository-relative POSIX path; the file name alone for a location outside the repository (tests)."""
    try:
        return paths.rel(p)
    except ValueError:
        return Path(p).name


def design_stem(design: str, variant: str, scheme: str) -> str:
    return f"{design}__{variant}__{scheme}"


def fold_record(f: Fold, fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The index record of one fold; ``fields`` (e.g. the section 13 ``support_tau`` thresholds) are written beside
    the fold hash.  They are not part of the hash, which covers the row assignment only."""
    rec = {"fold_id": f.fold_id, "half": f.half, "seed": f.seed, "batch_id": f.batch_id, "unit_type": f.unit_type,
           "units": list(f.units), "n_hidden": len(f.hidden_row_ids), "n_scored": len(f.scored_row_ids),
           "fold_hash": f.fold_hash, "meta": dict(f.meta)}
    for k, v in dict(fields or {}).items():
        if k in rec:
            raise ValueError(f"fold field {k!r} would overwrite the fold record")
        rec[k] = v
    return rec


def summarise(folds: Sequence[Fold]) -> dict[str, Any]:
    """Counts of one design file: folds, scored units / rows (overall and per half), hashes."""
    a = assignment_frame(folds)
    sc = a[a["role"] == "hidden_scored"]
    out: dict[str, Any] = {
        "n_folds": len(folds), "n_folds_with_scored_rows": int(sum(1 for f in folds if f.scored_row_ids)),
        "n_hidden_row_entries": int(len(a)), "n_scored_row_entries": int(len(sc)),
        "n_distinct_scored_rows": int(sc["row_id"].nunique()), "n_scored_units": int(sc["unit"].nunique()),
        "seeds": sorted({f.seed for f in folds if f.seed is not None}),
        "design_hash": design_hash(folds), "assignment_sha256": assignment_sha256(folds),
        "by_half": {}}
    for h in HALF_VALUES:
        s = sc[sc["half"] == h]
        if len(s) or any(f.half == h for f in folds):
            out["by_half"][h] = {"n_folds": int(sum(1 for f in folds if f.half == h)),
                                 "n_scored_units": int(s["unit"].nunique()), "n_scored_row_entries": int(len(s)),
                                 "n_distinct_scored_rows": int(s["row_id"].nunique())}
    return out


def write_design(folds: Sequence[Fold], out_dir: Path, extra: Mapping[str, Any] | None = None,
                 fold_fields: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[Path, Path, dict]:
    """Write ``<stem>.parquet`` (long assignment) and ``<stem>.json`` (index); returns paths and the summary.
    ``fold_fields`` maps a fold id to extra fields of its index record (:func:`fold_record`)."""
    if not folds:
        raise ValueError("write_design: no folds")
    keys = {(f.design, f.variant, f.scheme) for f in folds}
    if len(keys) != 1:
        raise ValueError(f"write_design: one design / variant / scheme per file, got {sorted(keys)}")
    design, variant, scheme = next(iter(keys))
    stem = design_stem(design, variant, scheme)
    paths.ensure_dir(Path(out_dir))
    a = assignment_frame(folds)
    pq, js = Path(out_dir) / f"{stem}.parquet", Path(out_dir) / f"{stem}.json"
    a.to_parquet(pq, index=False, compression="zstd")
    summary = summarise(folds)
    body = {"schema": SCHEMA, "design": design, "variant": variant, "scheme": scheme,
            "parquet": _rel(pq), "roles": list(ROLES), "summary": summary, **dict(extra or {}),
            "folds": [fold_record(f, (fold_fields or {}).get(f.fold_id)) for f in folds]}
    from gen19ct.manifest import write_json
    write_json(js, body)
    return pq, js, summary


def read_design(stem_or_json: str | Path, folds_dir: Path | None = None, verify: bool = True) -> list[Fold]:
    """Folds of one design file (``<stem>`` or the JSON path); every fold hash is re-derived when ``verify``."""
    p = Path(stem_or_json)
    js = p if p.suffix == ".json" else (Path(folds_dir or paths.FOLDS_DIR) / f"{p.name}.json")
    body = json.loads(js.read_text(encoding="utf-8"))
    a = pd.read_parquet(js.with_suffix(".parquet"))
    by = {k: g for k, g in a.groupby("fold_id", sort=False)}
    out = []
    for rec in body["folds"]:
        g = by.get(rec["fold_id"], a.iloc[0:0])
        hidden = g["row_id"].tolist()
        scored = g.loc[g["role"] == "hidden_scored", "row_id"].tolist()
        f = make_fold(design=body["design"], variant=body["variant"], scheme=body["scheme"], fold_id=rec["fold_id"],
                      half=rec["half"], seed=rec["seed"], hidden=hidden, scored=scored, unit_type=rec["unit_type"],
                      units=rec["units"], batch_id=rec["batch_id"], row_unit=dict(zip(g["row_id"], g["unit"])),
                      row_half=dict(zip(g["row_id"], g["half"])), meta=rec.get("meta") or {})
        if verify and f.fold_hash != rec["fold_hash"]:
            raise AssertionError(f"{js.name}: fold {f.fold_id} hash {f.fold_hash} != index {rec['fold_hash']}")
        out.append(f)
    if verify and design_hash(out) != body["summary"]["design_hash"]:
        raise AssertionError(f"{js.name}: design hash does not reproduce")
    return out


def read_fold_fields(stem_or_json: str | Path, field: str, folds_dir: Path | None = None) -> dict[str, Any]:
    """``{fold_id: record[field]}`` of one design index (``None`` where a record lacks the field)."""
    p = Path(stem_or_json)
    js = p if p.suffix == ".json" else (Path(folds_dir or paths.FOLDS_DIR) / f"{p.name}.json")
    body = json.loads(js.read_text(encoding="utf-8"))
    return {rec["fold_id"]: rec.get(field) for rec in body["folds"]}


def copy_crossings(folds: Sequence[Fold], pairs: pd.DataFrame) -> pd.DataFrame:
    """Scored rows of each fold with a ``leakage.wildcard_copy_pairs`` partner that stays in the fold's training set
    (verification finding VL-04): ``fold_id, half, row_id, partner_id, kind, strict_copy, same_publication``."""
    cols = ["fold_id", "half", "row_id", "partner_id", "kind", "strict_copy", "same_publication"]
    if pairs is None or not len(pairs):
        return pd.DataFrame(columns=cols)
    partners: dict[str, list[tuple[str, str, bool, bool]]] = {}
    for a, b, k, st, sp in zip(pairs["id_a"].astype(str), pairs["id_b"].astype(str), pairs["kind"],
                               pairs["strict_copy"].astype(bool), pairs["same_publication"].astype(bool)):
        partners.setdefault(a, []).append((b, k, st, sp))
        partners.setdefault(b, []).append((a, k, st, sp))
    recs = []
    for f in folds:
        hid = None
        for r in f.scored_row_ids:
            if r not in partners:
                continue
            if hid is None:
                hid = set(f.hidden_row_ids)
            for q, k, st, sp in partners[r]:
                if q not in hid:
                    recs.append((f.fold_id, str(f.row_half.get(r, f.half)), r, q, k, st, sp))
    return pd.DataFrame(recs, columns=cols)


def publication_group_frame(frame: pd.DataFrame, group_col: str = GROUP_COL) -> pd.DataFrame:
    """``frame`` with ``g19_publication_id`` replaced by the registered publication group (``group_col``), so that the
    ``V1`` level of ``leakage.fold_isolation_check`` ("no publication on both sides") reads the section 2 publication
    unit instead of the raw id.  The raw id is nested in the group, so the group-level check is strictly stronger: it
    also sees a row of the test rows' copy group whose ``g19_publication_id`` differs from every test row's (task X,
    finding VR-02 of the leakage lens).  Raises when the column is absent or incomplete."""
    if group_col not in frame.columns:
        raise KeyError(f"publication-group guard: column {group_col!r} absent from the frame")
    if frame[group_col].isna().any():
        raise ValueError(f"publication-group guard: {int(frame[group_col].isna().sum())} row(s) without {group_col}")
    return frame.assign(**{L.PUB_COL: frame[group_col].astype(str)})


def guard(fold: Fold, frame: pd.DataFrame, universe_index: pd.Index, test_ids: Iterable[str], level: str,
          publication_col: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """``leakage.fold_isolation_check`` on (universe minus hidden, ``test_ids``); raises with the fold id.

    ``publication_col`` (e.g. :data:`GROUP_COL`): the publication basis of the ``V1`` level is that column
    (:func:`publication_group_frame`) instead of ``g19_publication_id``; the report records it under
    ``publication_basis``."""
    hidden_idx = index_of(frame, fold.hidden_row_ids)
    if len(hidden_idx) != len(fold.hidden_row_ids):
        raise AssertionError(f"{fold.fold_id}: hidden ids missing from the frame")
    train_idx = universe_index.difference(hidden_idx)
    if len(train_idx) + len(hidden_idx) != len(universe_index) or not hidden_idx.isin(universe_index).all():
        raise AssertionError(f"{fold.fold_id}: training + hidden is not a partition of the universe")
    te = index_of(frame, test_ids)
    kwargs.setdefault("near_dup_sig", NEAR_DUP_SIG)
    chk = frame if publication_col is None else publication_group_frame(frame, publication_col)
    rep = L.fold_isolation_check(train_idx, te, chk, level, raise_on_violation=False, **kwargs)
    rep["publication_basis"] = L.PUB_COL if publication_col is None else publication_col
    if not rep["ok"]:
        bad = {k: v for k, v in rep["violations"].items() if v}
        raise AssertionError(f"fold isolation violated in {fold.design}/{fold.variant}/{fold.scheme}/{fold.fold_id}: "
                             f"{bad}; examples {({k: rep['examples'][k] for k in bad})}")
    return rep
