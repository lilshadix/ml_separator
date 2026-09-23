"""``models/interface.py`` -- the contract every Phase C+ arm implements, the row table the closed-form
baselines share, and split-conformal intervals with inner, publication-grouped calibration.

Registered text: ``preregistration_draft.md`` section 5 (last paragraph: every baseline carries split-conformal
intervals), section 7 (inner designs), section 12 (calibration only in inner folds, publication-grouped), section
13 (support columns of a prediction record), section 2 (what is never scored).  Nothing here reads ``log_D`` except
the conformal residuals, which are computed on inner calibration rows of the OUTER TRAINING set only.

Arm protocol (:class:`Arm`)
---------------------------
``clone() -> Arm`` (an unfitted copy), ``fit(train_rows, context) -> self``, ``predict(query_rows) -> DataFrame``.
Frames are in the :func:`prepare_frame` layout (``support_graph.prepare_support_frame`` + ``log_D`` + ``pub_group``).
``predict`` returns one row per query row, in query order, with :data:`PREDICTION_COLUMNS`:

``row_id``                        the query row's index label
``mean_logD`` / ``std_logD``      point prediction; ``std_logD`` is NaN for an arm without a variance model
``lower_50`` ... ``upper_95``     NaN from a bare arm; filled by :class:`ConformalWrapper`
``fallback_level``                the level whose value was returned: ``B0``, ``B1:state``,
                                  ``B1:element_unknown_state``, ``B2:system``, ``B2:family``, ``B2:expert``, ``B3``,
                                  ``B3x``, ``B3i``, ``B3l``, ``B4``, ``B4x``, ``B4l``, ``B7``
``fallback_reason``               why a chain moved past its first level (empty when it did not)
diagnostics                       nearest row used (``nn_*``), the lookup system / metal, the nearest-radius metal with
                                  its pool / basis / distance, the radius bracket, the nearest ligand system with its
                                  Tanimoto and ``d_desc``, the excluded publication group (B4), the B7 unit, and the
                                  conformal quantiles.  Columns an arm does not use are NaN / None.

An arm may also implement the fast path ``fit_table(table, train_mask, context)`` /
``predict_positions(positions)`` on a :class:`RowTable` built once per process; :class:`ConformalWrapper` uses it
(thousands of inner fits).  Both paths give identical predictions (``tests/test_baselines.py``).

Hidden rows
-----------
``FitContext.hidden_index`` names the rows a fold removed.  An arm must raise ``AssertionError`` when one of them is
among its training rows (at fit) or would enter a candidate pool (at predict), and when a query row is a training row.

Conformal intervals (:class:`ConformalWrapper`)
-----------------------------------------------
``fit``: the outer training rows are split by a registered inner design (:class:`GroupKFoldCalibration` = V1,
:class:`InnerCellCalibration` = V5 / V5-PAIR / V6, :class:`InnerMetalCalibration` = V2; section 7) whose inner units
come from the fold builders' functions (``folds.source_holdout.inner_group_assignment``,
``folds.cell_holdout.inner_cell_majority`` / ``inner_cell_assignment``, ``folds.metal_holdout.inner_state_pick``), so
calibration and the fold builder share one implementation per design (``tests/test_inner_designs.py``).  Every inner
split passes ``context.isolation_check`` (a caller-supplied ``fold_isolation_check`` at the design level) before the
arm is fitted on its inner-training rows; the calibration rows pass ``registered.assert_not_scored``.  The absolute
residuals of all inner calibration rows are pooled; the arm is refitted on the full outer training rows and every
query gets ``mean +- q_level`` with ``q_level`` the ``ceil((n + 1) level)``-th smallest residual (``inf`` when that
rank exceeds ``n``).  Choices where the registration is silent are listed in :data:`CONFORMAL_CHOICES`.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.folds import metal_holdout as MH
from gen19ct.folds import registered as FR
from gen19ct.folds import source_holdout as SH

TARGET_COL = "log_D"
ID_COL = "canonical_measurement_id"
PUB_GROUP_COL = "pub_group"
#: the registered publication unit (prereg section 2), mapped from ``g19_publication_id``
PUB_GROUP_SOURCE = "group_cross_publication_copy"
PUB_COMPONENTS_CSV = paths.DATA_AUDIT_DIR / "leakage_publication_components.csv"
SYSTEMS_CSV = paths.DESCRIPTORS_DIR / "extractant_systems.csv"
COMPONENTS_CSV = paths.DESCRIPTORS_DIR / "extractant_components.csv"
NA_ANION = "NA"
#: the archive marks this oxidation state implausible; never scored (prereg section 2)
SR_III = "Sr(III)"
#: registered interval levels (prereg section 4)
LEVELS: tuple[float, ...] = (0.50, 0.80, 0.95)
#: ``d_desc`` descriptors of the primary extractant component (prereg section 3.5)
D_DESC_COLUMNS: tuple[str, ...] = ("mw", "mol_logp", "rotatable_bonds")
#: mechanism label -> brief expert (the ``g19_feasibility.BRIEF_EXPERT`` grouping, ``feasibility_mechanisms.csv``)
BRIEF_EXPERT: dict[str, str] = {
    "ACIDIC_CATION_EXCHANGE": "E1_acidic_cation_exchange", "MIXED_ACIDIC": "E1_acidic_cation_exchange",
    "NEUTRAL_SOLVATING": "E2_neutral_solvating", "SOFT_N_DONOR": "E2_neutral_solvating",
    "MIXED_NEUTRAL": "E2_neutral_solvating", "ION_PAIR_BASIC": "E3_ion_pair_basic", "CHELATING": "E4_chelating",
    "SYNERGISTIC": "E5_synergistic", "UNKNOWN": "none_unknown",
}
#: the pseudo-expert that is not a chemistry group: B2 skips it (straight to B0)
NO_EXPERT = "none_unknown"

TABLE_COLUMNS: tuple[str, ...] = (SG.METAL_COL, SG.ELEMENT_COL, SG.SYSTEM_COL, SG.ACID_ANION_COL, SG.LOG_ACID_COL,
                                  SG.LOG_EXT_COL, ID_COL, TARGET_COL)
QUERY_COLUMNS: tuple[str, ...] = (SG.METAL_COL, SG.ELEMENT_COL, SG.SYSTEM_COL, SG.ACID_ANION_COL, SG.LOG_ACID_COL,
                                  SG.LOG_EXT_COL)

CORE_COLUMNS: tuple[str, ...] = ("row_id", "mean_logD", "std_logD", "lower_50", "upper_50", "lower_80", "upper_80",
                                 "lower_95", "upper_95", "fallback_level")
DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "fallback_reason", "anion_dropped", "n_candidates", "nn_row_id", "nn_canonical_measurement_id", "nn_distance",
    "lookup_system", "lookup_metal",
    "nearest_radius_metal", "nearest_radius_distance_A", "nearest_radius_basis", "nearest_radius_pool",
    "radius_pool_anion_dropped", "bracket_lower_metal", "bracket_upper_metal", "bracket_lower_logD",
    "bracket_upper_logD", "interp_fraction",
    "nearest_ligand_system", "nearest_ligand_tanimoto", "nearest_ligand_d_desc", "n_ligand_ties",
    "excluded_pub_group",
    "b7_anion", "b7_intercept_metal", "b7_n", "b7_p_eff", "b7_n_status", "b7_p_eff_status", "b7_pub_effect",
    "conformal_n_calibration", "conformal_q50", "conformal_q80", "conformal_q95",
)
PREDICTION_COLUMNS: tuple[str, ...] = CORE_COLUMNS + DIAGNOSTIC_COLUMNS
_FLOAT_COLUMNS = frozenset({"mean_logD", "std_logD", "lower_50", "upper_50", "lower_80", "upper_80", "lower_95",
                            "upper_95", "nn_distance", "nearest_radius_distance_A", "bracket_lower_logD",
                            "bracket_upper_logD", "interp_fraction", "nearest_ligand_tanimoto",
                            "nearest_ligand_d_desc", "b7_n", "b7_p_eff", "b7_pub_effect", "conformal_q50",
                            "conformal_q80", "conformal_q95"})

#: registration gaps the conformal code had to fill (reported with every run that uses it)
CONFORMAL_CHOICES: dict[str, str] = {
    "residual_pooling": "absolute residuals of every inner calibration row of every inner fold are pooled, unweighted",
    "centre": "intervals are centred on the arm refitted on the full outer training rows",
    "quantile": "q = ceil((n + 1) level)-th smallest absolute residual; inf when that rank exceeds n",
    "not_scored_in_calibration": "V6_TARGET_ROWS, Sr(III) rows, context.exclude_from_scoring rows and unknown-state "
                                 "(X(?)) rows, in every inner design: the calibration population is the scored "
                                 "population (folds.io.scorable_mask), X(?) rows being scored in no design",
    "inner_units": "V1 / V0: folds.source_holdout.inner_group_assignment; V5: folds.cell_holdout.inner_cell_majority "
                   "+ inner_cell_assignment; V2: folds.metal_holdout.inner_state_pick -- the fold builder's functions "
                   "and random streams (SeedSequence([seed, tag]), folds.io.STREAM_TAGS), one implementation per design",
    "seeds": "section 15 resolution (2026-09-15): the deterministic arms' inner calibration folds are drawn with each "
             "seed of ConformalWrapper(seeds=[...]) (the 5 discovery seeds; seed 104729 alone where a learned arm's "
             "comparison is on it; each withheld seed at confirmation) and coverage / width are averaged over the "
             "seeds; seeds=None keeps the single context.seed draw of the disclosed pre-seal run",
    "learned_arm_inner_design": "POST-HOC addendum 1 (2026-09-15) item 1: every learned arm and B6 tunes and calibrates "
                                "on models.inner_design.SimultaneousInnerCells -- one inner fit per inner fold with "
                                "all of the fold's inner cells hidden together (the same cells as InnerCellCalibration); "
                                "InnerCellCalibration (one split per cell) stays the registered calibration design of "
                                "the deterministic comparators B0-B4 and B7 only",
}


def _missing(v: Any) -> bool:
    return SG._missing(v)


def _float_array(values: Any) -> np.ndarray:
    out = np.array(pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float), dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def _labels(values: Any) -> np.ndarray:
    """Object array of ``str`` labels, ``None`` for a missing value."""
    return np.array([None if _missing(v) else str(v) for v in pd.Series(values).to_numpy(dtype=object)], dtype=object)


def _codes(values: Any) -> tuple[np.ndarray, list[str]]:
    lab = _labels(values)
    present = sorted({v for v in lab if v is not None})
    lookup = {v: i for i, v in enumerate(present)}
    return np.array([lookup[v] if v is not None else -1 for v in lab], dtype=np.int64), present


def _group(keys: Sequence[np.ndarray], valid: np.ndarray) -> dict[Any, np.ndarray]:
    """``{key: sorted positions}`` over rows where ``valid``; a one-column key is the bare int."""
    idx = np.flatnonzero(valid)
    if not len(idx):
        return {}
    k = np.stack([np.asarray(a)[idx] for a in keys], axis=1)
    order = np.lexsort(k.T[::-1])
    ks, pos = k[order], idx[order]
    change = np.any(ks[1:] != ks[:-1], axis=1) if len(ks) > 1 else np.zeros(0, dtype=bool)
    starts = np.r_[0, np.flatnonzero(change) + 1]
    ends = np.r_[starts[1:], len(ks)]
    out: dict[Any, np.ndarray] = {}
    for s, e in zip(starts, ends):
        key = tuple(int(x) for x in ks[s])
        out[key[0] if len(key) == 1 else key] = np.sort(pos[s:e])
    return out


# --------------------------------------------------------------------------------------------- #
# frames and descriptor tables
# --------------------------------------------------------------------------------------------- #

def load_descriptor_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(extractant_systems.csv indexed by extractant_system_key, extractant_components.csv)``."""
    systems = pd.read_csv(SYSTEMS_CSV)
    return systems.set_index("extractant_system_key", drop=False), pd.read_csv(COMPONENTS_CSV)


def prepare_frame(model_rows: pd.DataFrame, systems: pd.DataFrame | None = None,
                  components_csv: Any = PUB_COMPONENTS_CSV) -> pd.DataFrame:
    """MODEL rows -> the arm frame: ``support_graph.prepare_support_frame`` plus ``pub_group`` (the registered
    ``group_cross_publication_copy`` of ``g19_publication_id``).  Raises when a row has no group."""
    fr = SG.prepare_support_frame(model_rows, systems=systems)
    comp = pd.read_csv(components_csv).set_index("g19_publication_id")[PUB_GROUP_SOURCE]
    fr[PUB_GROUP_COL] = fr[SG.PUB_COL].map(comp)
    if fr[PUB_GROUP_COL].isna().any():
        raise RuntimeError(f"{int(fr[PUB_GROUP_COL].isna().sum())} rows have no {PUB_GROUP_SOURCE}")
    return fr


# --------------------------------------------------------------------------------------------- #
# the row table
# --------------------------------------------------------------------------------------------- #

class RowTable:
    """Integer-coded row arrays and group positions of a frame, built once; a fold's training set is a boolean mask.

    Required columns: :data:`TABLE_COLUMNS`.  Optional: ``pub_group`` (B4, B7, conformal inner designs),
    ``g19_publication_id`` (the ``p`` count of V5 eligibility; defaults to the group), ``system_family``,
    ``mechanism``, ``primary_extractant_smiles`` (else taken from ``systems``).  ``log_D`` may be NaN for rows that are
    only ever queried; a fit refuses non-finite training targets.  ``canonical_measurement_id`` must be unique (it is
    the last tie-break of B3).
    """

    def __init__(self, frame: pd.DataFrame, *, systems: pd.DataFrame | None = None,
                 components: pd.DataFrame | None = None, pub_group_col: str = PUB_GROUP_COL,
                 target_col: str = TARGET_COL):
        missing = [c for c in TABLE_COLUMNS[:-1] + (target_col,) if c not in frame.columns]
        if missing:
            raise KeyError(f"RowTable: columns missing {missing}")
        if not frame.index.is_unique:
            raise ValueError("RowTable: the frame index must be unique")
        n = self.n = len(frame)
        self.index = frame.index
        self.pub_group_col = pub_group_col
        self.systems = systems
        self.components = components
        self.state, self.state_labels = _codes(frame[SG.METAL_COL])
        self.elem, self.elem_labels = _codes(frame[SG.ELEMENT_COL])
        self.sys, self.sys_labels = _codes(frame[SG.SYSTEM_COL])
        anion = [NA_ANION if _missing(v) else str(v) for v in frame[SG.ACID_ANION_COL].to_numpy(dtype=object)]
        self.anion, self.anion_labels = _codes(anion)
        if pub_group_col in frame.columns:
            self.pub, self.pub_labels = _codes(frame[pub_group_col])
        else:
            self.pub, self.pub_labels = np.full(n, -1, dtype=np.int64), []
        if SG.PUB_COL in frame.columns:
            self.pubid, _ = _codes(frame[SG.PUB_COL])
        else:
            self.pubid = self.pub
        self.acid = _float_array(frame[SG.LOG_ACID_COL])
        self.ext = _float_array(frame[SG.LOG_EXT_COL])
        self.y = _float_array(frame[target_col])
        ids = _labels(frame[ID_COL])
        if any(v is None for v in ids) or len(set(ids)) != n:
            raise ValueError(f"RowTable: {ID_COL} must be present and unique")
        self.ids = ids
        order = np.argsort(ids.astype(str), kind="stable")
        self.id_rank = np.empty(n, dtype=np.int64)
        self.id_rank[order] = np.arange(n)
        self.state_code = {v: i for i, v in enumerate(self.state_labels)}
        self.elem_code = {v: i for i, v in enumerate(self.elem_labels)}
        self.sys_code = {v: i for i, v in enumerate(self.sys_labels)}
        self.anion_code = {v: i for i, v in enumerate(self.anion_labels)}
        self.pub_code = {v: i for i, v in enumerate(self.pub_labels)}

        # static per-system chemistry labels: the rows' own columns first, then the descriptor table
        nsys = len(self.sys_labels)
        self.sys_family = np.array([None] * nsys, dtype=object)
        self.sys_mechanism = np.array([None] * nsys, dtype=object)
        self.sys_smiles = np.array([None] * nsys, dtype=object)
        for arr, col in ((self.sys_family, SG.FAMILY_COL), (self.sys_mechanism, SG.MECH_COL),
                         (self.sys_smiles, SG.SMILES_COL)):
            if col in frame.columns:
                vals = _labels(frame[col])
                for i in np.argsort(self.sys, kind="stable")[::-1]:     # first row of each system wins
                    if self.sys[i] >= 0 and vals[i] is not None:
                        arr[self.sys[i]] = vals[i]
            for c, lab in enumerate(self.sys_labels):
                if arr[c] is None:
                    arr[c] = self._from_systems_table(lab, col)
        self.sys_expert = np.array([BRIEF_EXPERT.get(m, NO_EXPERT) if m is not None else None
                                    for m in self.sys_mechanism], dtype=object)
        self.sys_desc = np.vstack([self.descriptor_vector(s) for s in self.sys_smiles]) if nsys else np.zeros((0, 3))
        self.sys_fp = [SG.fingerprint(s) if s is not None else None for s in self.sys_smiles]
        fam_labels = sorted({f for f in self.sys_family if f is not None})
        self.family_code = {f: i for i, f in enumerate(fam_labels)}
        exp_labels = sorted({e for e in self.sys_expert if e is not None and e != NO_EXPERT})
        self.expert_code = {e: i for i, e in enumerate(exp_labels)}
        row_fam = np.array([self.family_code.get(self.sys_family[s], -1) if s >= 0 else -1 for s in self.sys])
        row_exp = np.array([self.expert_code.get(self.sys_expert[s], -1) if s >= 0 else -1 for s in self.sys])

        known = self.state >= 0
        has_sys = self.sys >= 0
        self.by_sma = _group((self.sys, self.state, self.anion), has_sys & known)
        self.by_sm = _group((self.sys, self.state), has_sys & known)
        self.by_s = _group((self.sys,), has_sys)
        self.by_sa_known = _group((self.sys, self.anion), has_sys & known)
        self.by_state = _group((self.state,), known)
        self.by_unknown_elem = _group((self.elem,), ~known & (self.elem >= 0))
        self.by_family = _group((row_fam,), row_fam >= 0)
        self.by_expert = _group((row_exp,), row_exp >= 0)
        self.states_by_sa: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
        for (s, m, a), p in self.by_sma.items():
            self.states_by_sa[(s, a)].append((m, p))
        self.states_by_s: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        self.systems_by_state: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for (s, m), p in self.by_sm.items():
            self.states_by_s[s].append((m, p))
            self.systems_by_state[m].append((s, p))
        self._sharing: dict[tuple[str, int], tuple[Any, np.ndarray]] = {}

    # ----------------------------------------------------------------------------------------- #
    def _from_systems_table(self, system: str, col: str) -> str | None:
        tab = self.systems
        if tab is None:
            return None
        if "extractant_system_key" in tab.columns and tab.index.name != "extractant_system_key":
            tab = tab.set_index("extractant_system_key", drop=False)
        src = {SG.FAMILY_COL: "system_family", SG.MECH_COL: "mechanism", SG.SMILES_COL: "primary_extractant_smiles"}[col]
        if system in tab.index and src in tab.columns and not _missing(tab.loc[system, src]):
            return str(tab.loc[system, src])
        return None

    def descriptor_vector(self, smiles: str | None) -> np.ndarray:
        """``(mw, mol_logp, rotatable_bonds)`` of a component structure (NaN when unknown)."""
        comp = self.components
        if comp is None or smiles is None:
            return np.full(len(D_DESC_COLUMNS), np.nan)
        if not hasattr(self, "_desc_lookup"):
            st = comp[comp["record_type"] == "STRUCTURE"] if "record_type" in comp.columns else comp
            self._desc_lookup = {str(s): np.array([float(v) for v in row], dtype=float)
                                 for s, row in zip(st["smiles_canonical"], st[list(D_DESC_COLUMNS)].to_numpy())}
        return self._desc_lookup.get(str(smiles), np.full(len(D_DESC_COLUMNS), np.nan)).copy()

    def static(self, system: str | None) -> dict[str, Any]:
        """Family, mechanism, brief expert, primary SMILES and fingerprint of a system, in the table or not."""
        if system is None:
            return {"family": None, "mechanism": None, "expert": None, "smiles": None, "fp": None,
                    "desc": np.full(len(D_DESC_COLUMNS), np.nan)}
        c = self.sys_code.get(system)
        if c is not None:
            return {"family": self.sys_family[c], "mechanism": self.sys_mechanism[c], "expert": self.sys_expert[c],
                    "smiles": self.sys_smiles[c], "fp": self.sys_fp[c], "desc": self.sys_desc[c]}
        fam = self._from_systems_table(system, SG.FAMILY_COL)
        mech = self._from_systems_table(system, SG.MECH_COL)
        smi = self._from_systems_table(system, SG.SMILES_COL)
        return {"family": fam, "mechanism": mech, "expert": BRIEF_EXPERT.get(mech, NO_EXPERT) if mech else None,
                "smiles": smi, "fp": SG.fingerprint(smi) if smi else None, "desc": self.descriptor_vector(smi)}

    def positions(self, labels: Iterable[Any]) -> np.ndarray:
        pos = self.index.get_indexer(pd.Index(list(labels)))
        if (pos < 0).any():
            raise KeyError("labels not in the RowTable")
        return pos

    def mask_of(self, labels: Iterable[Any]) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        m[self.positions(labels)] = True
        return m

    def aligned_bool(self, series: pd.Series | None, what: str) -> np.ndarray:
        """A boolean Series on (a superset of) the table index as a positional array."""
        if series is None:
            return np.zeros(self.n, dtype=bool)
        s = series.reindex(self.index)
        if s.isna().any():
            raise KeyError(f"{what} does not cover every RowTable row")
        return s.to_numpy(dtype=bool)

    def sharing_codes(self, system: str, component_map: Mapping[str, str] | None = None) -> np.ndarray:
        """Codes of the table systems equal to ``system`` or sharing a component structure with it."""
        key = (system, id(component_map) if component_map is not None else 0)
        hit = self._sharing.get(key)
        if hit is None or hit[0] is not component_map:           # identity check: ids of dead maps can be reused
            share = SG.systems_sharing_component(self.sys_labels, system, component_map) | {system}
            hit = (component_map, np.array(sorted(self.sys_code[k] for k in share if k in self.sys_code),
                                           dtype=np.int64))
            self._sharing[key] = hit
        return hit[1]


def hide_cell_mask(table: RowTable, mask: np.ndarray, metal_state: str, system: str, *, component_aware: bool = True,
                   component_map: Mapping[str, str] | None = None,
                   include_unknown_state_alias: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """``support_graph.hide_cell`` as a mask operation: ``(mask without the hidden rows, the rows it removed)``.
    Same state-level rule (the cell's state, plus the element's X(?) rows, in the system and -- component-aware --
    every system sharing a component); equality with ``hide_cell`` is tested."""
    codes = table.sharing_codes(system, component_map) if component_aware else \
        np.array([table.sys_code[system]] if system in table.sys_code else [], dtype=np.int64)
    scope = np.isin(table.sys, codes)
    drop = np.zeros(table.n, dtype=bool)
    mc = table.state_code.get(metal_state)
    if mc is not None:
        drop |= scope & (table.state == mc)
    if include_unknown_state_alias:
        ec = table.elem_code.get(SG.metal_properties(metal_state)["symbol"])
        if ec is not None:
            drop |= scope & (table.state < 0) & (table.elem == ec)
    drop &= mask
    return mask & ~drop, drop


# --------------------------------------------------------------------------------------------- #
# context and protocol
# --------------------------------------------------------------------------------------------- #

@dataclass
class FitContext:
    """What a fit may see besides its training rows.

    ``systems`` / ``components``  descriptor tables (static chemistry, never targets)
    ``pub_group_col``             the publication-group column used by B4, B7 and the inner designs
    ``seed``                      the run seed (inner designs and conformal subsamples draw from it)
    ``table``                     a :class:`RowTable` over a superset of every frame of this process (fast path)
    ``hidden_index``              rows removed by the fold; never training rows, never candidates
    ``v6_mask``                   ``V6_TARGET_ROWS`` over the table index (required by :class:`ConformalWrapper`)
    ``exclude_from_scoring``      further rows never scored (e.g. acidic co-extractant rows)
    ``isolation_check``           ``(train_labels, test_labels) -> report`` -- ``fold_isolation_check`` at the design
                                  level; required for inner splits
    ``guard_cache``               verified inner splits, shared by every arm of one outer fold
    ``train_rows``                the outer training frame (for :attr:`support_index`)
    """

    systems: pd.DataFrame | None = None
    components: pd.DataFrame | None = None
    pub_group_col: str = PUB_GROUP_COL
    seed: int | None = None
    table: RowTable | None = None
    hidden_index: pd.Index | None = None
    v6_mask: pd.Series | None = None
    exclude_from_scoring: pd.Series | None = None
    isolation_check: Callable[[pd.Index, pd.Index], Mapping[str, Any]] | None = None
    guard_cache: dict = field(default_factory=dict)
    train_rows: pd.DataFrame | None = None
    _support_index: Any = field(default=None, repr=False)

    def for_training(self, train_rows: pd.DataFrame | None = None,
                     hidden_index: Iterable[Any] | None = None) -> "FitContext":
        """A copy bound to ``train_rows`` (the SupportIndex is rebuilt lazily on them) with ``hidden_index`` added."""
        hid = self.hidden_index
        if hidden_index is not None:
            extra = pd.Index(list(hidden_index))
            hid = extra if hid is None else hid.union(extra)
        return replace(self, train_rows=train_rows, hidden_index=hid, _support_index=None)

    @property
    def support_index(self) -> SG.SupportIndex:
        """``SupportIndex`` fitted on ``train_rows`` only (built on first use)."""
        if self._support_index is None:
            if self.train_rows is None:
                raise ValueError("FitContext.support_index needs train_rows (use for_training)")
            self._support_index = SG.SupportIndex(self.train_rows, systems=self.systems)
        return self._support_index


@runtime_checkable
class Arm(Protocol):
    name: str

    def clone(self) -> "Arm": ...

    def fit(self, train_rows: pd.DataFrame, context: FitContext) -> "Arm": ...

    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame: ...


def table_and_mask(train_rows: pd.DataFrame, context: FitContext) -> tuple[RowTable, np.ndarray]:
    """``context.table`` and the mask of ``train_rows`` when the table holds every training row with the same target
    and system; otherwise a fresh table built on ``train_rows``."""
    t = context.table
    if t is not None:
        pos = t.index.get_indexer(train_rows.index)
        if len(pos) and (pos >= 0).all():
            y = _float_array(train_rows[TARGET_COL])
            same_y = np.array_equal(t.y[pos], y, equal_nan=True)
            same_sys = np.array_equal(np.array([t.sys_labels[c] if c >= 0 else None for c in t.sys[pos]], dtype=object),
                                      _labels(train_rows[SG.SYSTEM_COL]))
            if not (same_y and same_sys):
                raise ValueError("context.table disagrees with train_rows (target or system); rebuild the table")
            m = np.zeros(t.n, dtype=bool)
            m[pos] = True
            return t, m
    t = RowTable(train_rows, systems=context.systems, components=context.components,
                 pub_group_col=context.pub_group_col)
    return t, np.ones(t.n, dtype=bool)


def forbidden_mask(table: RowTable, context: FitContext) -> np.ndarray:
    """Positions of ``context.hidden_index`` rows present in the table."""
    out = np.zeros(table.n, dtype=bool)
    if context.hidden_index is not None and len(context.hidden_index):
        pos = table.index.get_indexer(pd.Index(context.hidden_index))
        out[pos[pos >= 0]] = True
    return out


@dataclass
class Queries:
    """Query attributes as arrays (from a frame or from table positions)."""

    labels: np.ndarray
    state: np.ndarray
    elem: np.ndarray
    system: np.ndarray
    anion: np.ndarray
    acid: np.ndarray
    ext: np.ndarray
    pub_group: np.ndarray
    table_pos: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, table: RowTable) -> "Queries":
        missing = [c for c in QUERY_COLUMNS if c not in frame.columns]
        if missing:
            raise KeyError(f"query rows: columns missing {missing}")
        if not frame.index.is_unique:
            raise ValueError("query rows: the index must be unique")
        n = len(frame)
        anion = np.array([NA_ANION if _missing(v) else str(v) for v in frame[SG.ACID_ANION_COL].to_numpy(dtype=object)],
                         dtype=object)
        pub = _labels(frame[table.pub_group_col]) if table.pub_group_col in frame.columns else \
            np.array([None] * n, dtype=object)
        return cls(labels=frame.index.to_numpy(dtype=object), state=_labels(frame[SG.METAL_COL]),
                   elem=_labels(frame[SG.ELEMENT_COL]), system=_labels(frame[SG.SYSTEM_COL]), anion=anion,
                   acid=_float_array(frame[SG.LOG_ACID_COL]), ext=_float_array(frame[SG.LOG_EXT_COL]), pub_group=pub,
                   table_pos=table.index.get_indexer(frame.index))

    @classmethod
    def from_positions(cls, table: RowTable, positions: np.ndarray) -> "Queries":
        p = np.asarray(positions, dtype=np.int64)

        def lab(codes: np.ndarray, labels: list[str]) -> np.ndarray:
            return np.array([labels[c] if c >= 0 else None for c in codes[p]], dtype=object)
        return cls(labels=table.index.to_numpy(dtype=object)[p], state=lab(table.state, table.state_labels),
                   elem=lab(table.elem, table.elem_labels), system=lab(table.sys, table.sys_labels),
                   anion=lab(table.anion, table.anion_labels), acid=table.acid[p].copy(), ext=table.ext[p].copy(),
                   pub_group=lab(table.pub, table.pub_labels), table_pos=p.copy())


def empty_prediction_record() -> dict[str, Any]:
    return {c: (np.nan if c in _FLOAT_COLUMNS else None) for c in PREDICTION_COLUMNS}


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame.from_records(records, columns=list(PREDICTION_COLUMNS))
    for c in _FLOAT_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def fallback_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rows per ``(fallback_level, fallback_reason)`` -- the per-fold fallback count the registration reports."""
    p = predictions.assign(fallback_reason=predictions["fallback_reason"].fillna(""))
    return p.groupby(["fallback_level", "fallback_reason"], dropna=False).size().rename("n_rows").reset_index()


# --------------------------------------------------------------------------------------------- #
# inner designs (prereg section 7) and split-conformal intervals (section 12)
# --------------------------------------------------------------------------------------------- #

@dataclass
class InnerSplit:
    """One inner split of the outer training rows.  ``certificate`` (optional) is a superset split
    ``(train_mask, test_positions)`` that contains this split on both sides.  ``row_units`` (optional) holds the
    section 4 / section 7 averaging unit of every calibration row, aligned with ``cal_positions`` (V5: the hidden cell,
    V1: the publication group or ``REMAINDER`` below 20 outer-training rows, V2: the metal state): a tuned arm's inner
    macro MAE averages over these units within an inner fold (section 7 "the design's own averaging"; addendum 1 item 2:
    the selection score is then the mean over the inner folds).  Conformal residuals never depend on it."""

    unit: Any
    fold: int
    train_mask: np.ndarray
    cal_positions: np.ndarray
    hidden_positions: np.ndarray
    certificate: tuple[np.ndarray, np.ndarray] | None = None
    row_units: np.ndarray | None = None
    #: optional design bookkeeping (e.g. the inner cells of a simultaneous split and the cells dropped from its score);
    #: never read by a fit or a calibration
    meta: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.row_units is not None:
            self.row_units = np.asarray(self.row_units, dtype=object)
            if len(self.row_units) != len(self.cal_positions):
                raise ValueError(f"inner split {self.unit}: {len(self.row_units)} row units for "
                                 f"{len(self.cal_positions)} calibration rows")


def cell_unit_label(state: str, system: str) -> str:
    """The V5 averaging-unit label of a cell (``folds.cell_holdout.cell_label``)."""
    return CH.cell_label((str(state), str(system)))


def _require_seed(context: FitContext) -> int:
    if context.seed is None:
        raise ValueError("inner calibration designs draw from context.seed; it must be set")
    return int(context.seed)


def scorable_mask(table: RowTable, context: FitContext, *, known_state_only: bool) -> np.ndarray:
    """Rows that may enter a calibration (scoring) set: never ``V6_TARGET_ROWS``, never Sr(III), never
    ``context.exclude_from_scoring``; optionally only known metal states."""
    if context.v6_mask is None:
        raise ValueError("context.v6_mask (V6_TARGET_ROWS) is required before any calibration row is scored")
    ok = ~table.aligned_bool(context.v6_mask, "v6_mask")
    if context.exclude_from_scoring is not None:
        ok &= ~table.aligned_bool(context.exclude_from_scoring, "exclude_from_scoring")
    sr = table.state_code.get(SR_III)
    if sr is not None:
        ok &= table.state != sr
    if known_state_only:
        ok &= table.state >= 0
    return ok


def _require_groups(table: RowTable, positions: np.ndarray) -> None:
    if len(positions) and (table.pub[positions] < 0).any():
        raise ValueError(f"training rows without {table.pub_group_col}; the inner designs group by publication")


def unit_frame(table: RowTable, mask: np.ndarray) -> pd.DataFrame:
    """The rows of ``table`` under ``mask`` as the minimal frame the fold builders' inner-unit functions read
    (``canonical_measurement_id``, metal state, element, system, ``g19_publication_id`` codes and the publication
    group), indexed by the table labels.  The calibration splitters below take their inner units from those
    functions on this frame, so the fold builder and the conformal wrapper share one implementation per design."""
    pos = np.flatnonzero(mask)

    def lab(codes: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        arr = np.array([None] + list(labels), dtype=object)
        return arr[codes[pos] + 1]
    pubid = np.array([None if c < 0 else f"pubid_{int(c)}" for c in table.pubid[pos]], dtype=object)
    return pd.DataFrame({FI.ROW_ID: table.ids[pos], SG.METAL_COL: lab(table.state, table.state_labels),
                         SG.ELEMENT_COL: lab(table.elem, table.elem_labels), SG.SYSTEM_COL: lab(table.sys, table.sys_labels),
                         SG.PUB_COL: pubid, FI.GROUP_COL: lab(table.pub, table.pub_labels)},
                        index=table.index[pos])


class GroupKFoldCalibration:
    """V1 inner design (also the V0 calibration design): grouped ``n_folds``-fold over the publication groups of the
    outer training rows, assigned by ``folds.source_holdout.inner_group_assignment`` (the fold builder's rule and
    random stream).  Calibration rows: the held-out groups' rows that a fold scores -- known metal state, not Sr(III),
    not ``V6_TARGET_ROWS``, not ``context.exclude_from_scoring`` (the calibration population equals the scored
    population; X(?) rows are never scored)."""

    name = "V1_group_kfold"
    UNITS = ("v1_unit", "publication_group")

    def __init__(self, n_folds: int = 3, unit: str = "v1_unit"):
        if unit not in self.UNITS:
            raise ValueError(f"unit must be one of {self.UNITS}")
        self.n_folds = int(n_folds)
        self.unit = unit

    def unit_assignment(self, table: RowTable, mask: np.ndarray, context: FitContext) -> dict[str, int]:
        """``{publication group: inner fold}`` of the outer training rows."""
        pos = np.flatnonzero(mask)
        _require_groups(table, pos)
        codes, counts = np.unique(table.pub[pos], return_counts=True)
        return SH.inner_group_assignment({table.pub_labels[c]: int(n) for c, n in zip(codes, counts)},
                                         _require_seed(context), self.n_folds)

    def row_unit_labels(self, table: RowTable, mask: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """The averaging unit of rows ``positions``: ``unit="v1_unit"`` the section 3.2 V1 unit applied to the outer
        training rows (the publication group, ``REMAINDER`` for a group below 20 of them; ``source_holdout.unit_of_rows``,
        the boosted and neural inner designs' unit), ``unit="publication_group"`` the group itself (V0)."""
        p = np.asarray(positions, dtype=np.int64)
        labels = np.array(table.pub_labels, dtype=object)[table.pub[p]]
        if self.unit == "publication_group":
            return labels
        codes, counts = np.unique(table.pub[np.flatnonzero(mask)], return_counts=True)
        big = {int(c) for c, n in zip(codes, counts) if n >= SH.REGISTERED_MIN_ROWS}
        return np.array([lab if int(c) in big else SH.REMAINDER for c, lab in zip(table.pub[p], labels)], dtype=object)

    def splits(self, table: RowTable, mask: np.ndarray, context: FitContext) -> list[InnerSplit]:
        assign = self.unit_assignment(table, mask, context)
        ok = scorable_mask(table, context, known_state_only=True)
        out = []
        for f in range(self.n_folds):
            groups = sorted(g for g, k in assign.items() if k == f)
            if not groups:
                continue
            hid = mask & np.isin(table.pub, [table.pub_code[g] for g in groups])
            cal = np.flatnonzero(hid & ok)
            if len(cal):
                out.append(InnerSplit(unit=tuple(groups), fold=f, train_mask=mask & ~hid, cal_positions=cal,
                                      hidden_positions=np.flatnonzero(hid),
                                      row_units=self.row_unit_labels(table, mask, cal)))
        return out


#: the inner V5 units of the last few (table, mask, seed, setting) calls -- every arm of one outer fold asks again
_INNER_V5_UNIT_CACHE: dict[tuple, tuple[Any, list[list[tuple[str, str]]]]] = {}
_INNER_V5_UNIT_CACHE_MAX = 4


class InnerCellCalibration:
    """V5 inner design: the candidate cells of ``folds.cell_holdout.inner_cell_majority`` (eligible under the outer
    thresholds recomputed on the outer training rows, no ``V6_TARGET_ROWS`` row, at least one scorable row) dealt into
    ``n_folds`` inner folds of at most ``max_cells_per_fold`` cells by ``folds.cell_holdout.inner_cell_assignment``
    (the fold builder's rule and random streams).  Each cell is hidden alone under the registered component-aware
    rule (exact leave-one-cell-out: the outer rule for the deterministic arms)."""

    name = "V5_inner_cells"

    def __init__(self, k: int = 10, p: int = 1, m: int = 3, n_folds: int = 3, max_cells_per_fold: int = 30,
                 component_aware: bool = True, component_map: Mapping[str, str] | None = None):
        self.k, self.p, self.m = int(k), int(p), int(m)
        self.n_folds, self.max_cells = int(n_folds), int(max_cells_per_fold)
        self.component_aware, self.component_map = component_aware, component_map

    def _majority(self, table: RowTable, mask: np.ndarray, context: FitContext) -> dict[tuple[str, str], str]:
        if context.v6_mask is None:
            raise ValueError("context.v6_mask is required: V6 cells are never inner validation cells")
        frame = unit_frame(table, mask)
        _require_groups(table, np.flatnonzero(mask))
        v6 = pd.Series(table.aligned_bool(context.v6_mask, "v6_mask")[np.flatnonzero(mask)], index=frame.index)
        return CH.inner_cell_majority(frame, CH.Thresholds(self.k, self.p, self.m), v6)

    def eligible_cells(self, table: RowTable, mask: np.ndarray, context: FitContext) -> list[tuple[str, str, np.ndarray]]:
        """``[(metal_state, system, training positions of the cell)]`` of the candidate cells, sorted by labels."""
        out = []
        for st, sy in sorted(self._majority(table, mask, context)):
            p = table.by_sm[(table.sys_code[sy], table.state_code[st])]
            out.append((st, sy, p[mask[p]]))
        return out

    def unit_assignment(self, table: RowTable, mask: np.ndarray, context: FitContext) -> list[list[tuple[str, str]]]:
        """The cells of each inner fold (``folds.cell_holdout.inner_cell_assignment``)."""
        seed = _require_seed(context)
        key = (id(table), hashlib.sha1(np.packbits(mask).tobytes()).hexdigest(), int(mask.sum()), seed, self.k, self.p,
               self.m, self.n_folds, self.max_cells, id(context.v6_mask))
        hit = _INNER_V5_UNIT_CACHE.get(key)
        if hit is not None and hit[0] is table:
            return hit[1]
        per_fold = CH.inner_cell_assignment(self._majority(table, mask, context), seed, self.n_folds, self.max_cells)
        if len(_INNER_V5_UNIT_CACHE) >= _INNER_V5_UNIT_CACHE_MAX:
            _INNER_V5_UNIT_CACHE.pop(next(iter(_INNER_V5_UNIT_CACHE)))
        _INNER_V5_UNIT_CACHE[key] = (table, per_fold)
        return per_fold

    def splits(self, table: RowTable, mask: np.ndarray, context: FitContext) -> list[InnerSplit]:
        per_fold = self.unit_assignment(table, mask, context)
        ok = scorable_mask(table, context, known_state_only=True)
        universe = np.ones(table.n, dtype=bool)
        out = []
        for f, members in enumerate(per_fold):
            for st, sy in members:
                p = table.by_sm[(table.sys_code[sy], table.state_code[st])]
                q = p[mask[p]]
                new_mask, dropped = hide_cell_mask(table, mask, st, sy, component_aware=self.component_aware,
                                                   component_map=self.component_map)
                cal = q[ok[q]]
                if not len(cal):              # only context.exclude_from_scoring rows (never on the MODEL rows)
                    continue
                cert_train, _ = hide_cell_mask(table, universe, st, sy, component_aware=self.component_aware,
                                               component_map=self.component_map)
                cert_test = p
                out.append(InnerSplit(unit=(st, sy), fold=f, train_mask=new_mask, cal_positions=cal,
                                      hidden_positions=np.flatnonzero(dropped), certificate=(cert_train, cert_test),
                                      row_units=np.full(len(cal), cell_unit_label(st, sy), dtype=object)))
        return out


class InnerMetalCalibration:
    """V2 inner design: leave-one-metal-out over the states of ``folds.metal_holdout.inner_state_pick`` (V2-eligible on
    the outer training rows, drawn by the fold builder's random stream); element-level hiding; only the state's own
    rows that a fold scores (outside ``V6_TARGET_ROWS`` and ``context.exclude_from_scoring``) are calibration rows."""

    name = "V2_inner_metals"

    def __init__(self, n_metals: int = 3, min_rows: int = 100, min_systems: int = 5):
        self.n_metals, self.min_rows, self.min_systems = int(n_metals), int(min_rows), int(min_systems)

    def unit_assignment(self, table: RowTable, mask: np.ndarray, context: FitContext) -> list[str]:
        return MH.inner_state_pick(unit_frame(table, mask), _require_seed(context), self.n_metals, self.min_rows,
                                   self.min_systems)

    def splits(self, table: RowTable, mask: np.ndarray, context: FitContext) -> list[InnerSplit]:
        take = self.unit_assignment(table, mask, context)
        if not take:
            return []
        ok = scorable_mask(table, context, known_state_only=True)
        out = []
        for st in take:
            mc = table.state_code[st]
            ec = table.elem_code[SG.metal_properties(st)["symbol"]]
            hid = mask & (table.elem == ec)
            cal = np.flatnonzero(hid & (table.state == mc) & ok)
            if len(cal):
                out.append(InnerSplit(unit=st, fold=len(out), train_mask=mask & ~hid, cal_positions=cal,
                                      hidden_positions=np.flatnonzero(hid),
                                      row_units=np.full(len(cal), str(st), dtype=object)))
        return out


def conformal_quantile(abs_residuals: np.ndarray, level: float) -> float:
    """Split-conformal quantile: the ``ceil((n + 1) level)``-th smallest absolute residual (``inf`` past ``n``)."""
    r = np.sort(np.asarray(abs_residuals, dtype=float))
    n = len(r)
    if n == 0:
        return float("inf")
    rank = int(math.ceil((n + 1) * float(level) - 1e-12))
    return float("inf") if rank > n else float(r[rank - 1])


def calibration_detail_of(sp: "InnerSplit", table: "RowTable", mean: np.ndarray) -> dict[str, Any]:
    """What one inner split's calibration produced, for a calibration of PAIR residuals: the split's unit and inner
    fold, its calibration rows' TABLE INDEX LABELS (``labels``, i.e. ``table.index`` -- the frame's index, which is not
    the ``canonical_measurement_id``: a caller keyed by row id maps them) and positions, and the SIGNED residual
    ``prediction - observed`` of each row.

    POST-HOC addendum 7 item 1 registers the S2(c) logSF interval as split conformal fitted on the pair residuals of the
    fold's inner calibration comparable pairs, and a pair residual is exactly ``signed_a - signed_b`` -- so the pair
    calibration needs the sign and the row, which the pooled absolute residuals of :class:`ConformalWrapper` drop.
    ``abs(signed)`` is the row residual the registered row calibration already uses, so nothing about the row intervals
    changes."""
    pos = np.asarray(sp.cal_positions)
    return {"unit": sp.unit, "fold": int(sp.fold), "positions": pos.copy(),
            "labels": list(table.index[pos]),
            "signed": np.asarray(mean, dtype=float) - table.y[pos]}


class ConformalWrapper:
    """Split-conformal intervals for any arm with the table fast path (module docstring).

    ``guard="every_split"`` (default) passes every inner split through ``context.isolation_check``;
    ``guard="nested_certificate"`` passes, once per inner unit, a superset split that contains the inner split on both
    sides (asserted), which certifies it because every ``fold_isolation_check`` violation needs a training row and a
    test row -- removing rows from either side cannot create one.  Verified splits are cached in
    ``context.guard_cache`` under a digest of the exact (train, test) positions, so one dict may be shared by every
    arm and every outer fold of a design: with ``nested_certificate`` a V5 sweep then checks each inner cell once
    (about 0.33 s per ``fold_isolation_check`` call on the MODEL rows; ``every_split`` makes about 90 calls per outer
    fold at the primary setting).

    **Seeds** (prereg section 15, resolved by the orchestrator 2026-09-15).  ``seeds=None`` (default): one calibration
    drawn with ``context.seed`` -- the behaviour of the disclosed single-seed pre-seal run, unchanged.  ``seeds=[...]``:
    the inner design is drawn once per seed (``context.seed`` replaced; point predictions unaffected), giving
    ``residuals_by_seed`` / ``quantiles_by_seed`` / ``calibration_units_by_seed``; the arm is refitted ONCE on the full
    outer training rows (the interval centre), so multi-seed mode is for arms whose point prediction does not depend on
    the seed -- the deterministic arms B0-B4, B7.  :meth:`predict_positions_by_seed` returns one interval frame per
    seed, :meth:`predict_positions` their long concatenation with a ``conformal_seed`` column, and
    :meth:`seed_interval_metrics` the per-seed and seed-mean coverage / width
    (``evaluation.calibration.seed_mean_interval_metrics``)."""

    def __init__(self, arm: Any, calibration: str = "inner", splitter: Any = None, levels: Sequence[float] = LEVELS,
                 guard: str = "every_split", seeds: Sequence[int] | None = None):
        if calibration != "inner":
            raise ValueError("calibration is fitted in inner folds only (prereg section 12)")
        if splitter is None:
            raise ValueError("ConformalWrapper needs an inner design (GroupKFoldCalibration, InnerCellCalibration, "
                             "InnerMetalCalibration)")
        if guard not in ("every_split", "nested_certificate"):
            raise ValueError(f"unknown guard mode {guard!r}")
        if tuple(levels) != LEVELS:
            raise ValueError(f"the registered levels are {LEVELS}")
        if seeds is not None:
            if isinstance(seeds, (str, bytes, int, np.integer)):
                raise TypeError("seeds must be a sequence of integers")
            seeds = tuple(int(s) for s in seeds)
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("seeds must be a non-empty sequence of distinct integers")
        self.arm, self.splitter, self.guard, self.seeds = arm, splitter, guard, seeds
        self.name = arm.name
        self.residuals: np.ndarray | None = None
        self.quantiles: dict[float, float] = {}
        self.fitted_arm: Any = None
        self.calibration_units: list[Any] = []
        self.fit_seed: int | None = None
        self.residuals_by_seed: dict[int, np.ndarray] = {}
        self.quantiles_by_seed: dict[int, dict[float, float]] = {}
        self.calibration_units_by_seed: dict[int, list[Any]] = {}
        #: per inner split, what that split's calibration produced: ``{"unit", "fold", "positions", "signed"}`` with
        #: ``signed = prediction - observed`` of every calibration row, in ``cal_positions`` order.  The absolute values
        #: of ``signed`` ARE :attr:`residuals` (asserted by ``tests/test_confirmation.py``), so this adds no quantity to
        #: the registered row calibration; it exposes the SIGN and the row, which a calibration of PAIR residuals needs
        #: (POST-HOC addendum 7 item 1: the S2(c) logSF interval is split conformal on the pair residuals of the fold's
        #: inner calibration comparable pairs, and a pair residual is ``signed_a - signed_b``).  Filled by
        #: :meth:`_calibrate`; per seed in :attr:`calibration_detail_by_seed` in multi-seed mode.
        self.calibration_detail: list[dict[str, Any]] = []
        self.calibration_detail_by_seed: dict[int, list[dict[str, Any]]] = {}
        self._last_detail: list[dict[str, Any]] = []

    @property
    def multi_seed(self) -> bool:
        return self.seeds is not None

    def clone(self) -> "ConformalWrapper":
        return ConformalWrapper(self.arm.clone(), splitter=self.splitter, guard=self.guard, seeds=self.seeds)

    def fit(self, train_rows: pd.DataFrame, context: FitContext) -> "ConformalWrapper":
        table, mask = table_and_mask(train_rows, context)
        return self.fit_table(table, mask, context.for_training(train_rows))

    def _verify(self, table: RowTable, sp: InnerSplit, context: FitContext) -> None:
        if context.isolation_check is None:
            raise ValueError("inner splits must pass fold_isolation_check: context.isolation_check is required")
        tr, te = sp.train_mask, np.sort(sp.cal_positions)
        if self.guard == "nested_certificate" and sp.certificate is not None:
            ctr, cte = sp.certificate
            if (tr & ~ctr).any() or not np.isin(te, cte).all():
                raise AssertionError(f"inner split {sp.unit} is not nested in its certificate split")
            tr, te = ctr, np.sort(np.asarray(cte))
        key = hashlib.sha256(np.flatnonzero(tr).astype(np.int64).tobytes() + b"|" +
                             te.astype(np.int64).tobytes()).hexdigest()
        if key in context.guard_cache:
            return
        rep = context.isolation_check(table.index[tr], table.index[te])
        if not (isinstance(rep, Mapping) and rep.get("ok", False)):
            raise AssertionError(f"inner split {sp.unit} failed the isolation check")
        context.guard_cache[key] = True

    def _calibrate(self, table: RowTable, mask: np.ndarray, context: FitContext) -> tuple[np.ndarray, list[Any]]:
        """Pooled absolute residuals and inner units of one draw of the inner design (``context.seed``)."""
        splits = self.splitter.splits(table, mask, context)
        if not splits:
            raise ValueError("the inner design produced no calibration split")
        res, units, detail = [], [], []
        for sp in splits:
            if (sp.train_mask & ~mask).any() or not mask[sp.cal_positions].all() or sp.train_mask[sp.cal_positions].any():
                raise AssertionError(f"inner split {sp.unit} is not inside the outer training rows")
            self._verify(table, sp, context)
            FR.assert_not_scored(table.index[sp.cal_positions], context.v6_mask, "conformal calibration set")
            ctx = context.for_training(None, hidden_index=table.index[sp.hidden_positions])
            arm = self.arm.clone().fit_table(table, sp.train_mask, ctx)
            pred = arm.predict_positions(sp.cal_positions)
            mean = pred["mean_logD"].to_numpy(dtype=float)
            if not np.isfinite(mean).all():
                raise AssertionError(f"{self.name}: non-finite calibration prediction in {sp.unit}")
            res.append(np.abs(table.y[sp.cal_positions] - mean))
            units.append(sp.unit)
            detail.append(calibration_detail_of(sp, table, mean))
        self._last_detail = detail
        return np.concatenate(res), units

    def fit_table(self, table: RowTable, mask: np.ndarray, context: FitContext) -> "ConformalWrapper":
        if self.seeds is None:
            self.fit_seed = context.seed
            self.residuals, self.calibration_units = self._calibrate(table, mask, context)
            self.calibration_detail = list(self._last_detail)
            self.quantiles = {lv: conformal_quantile(self.residuals, lv) for lv in LEVELS}
        else:
            self.residuals, self.quantiles, self.calibration_units, self.fit_seed = None, {}, [], None
            self.residuals_by_seed, self.quantiles_by_seed, self.calibration_units_by_seed = {}, {}, {}
            self.calibration_detail, self.calibration_detail_by_seed = [], {}
            for s in self.seeds:
                res, units = self._calibrate(table, mask, replace(context, seed=int(s)))
                self.residuals_by_seed[s] = res
                self.calibration_units_by_seed[s] = units
                self.calibration_detail_by_seed[s] = list(self._last_detail)
                self.quantiles_by_seed[s] = {lv: conformal_quantile(res, lv) for lv in LEVELS}
        self.fitted_arm = self.arm.clone().fit_table(table, mask, context)
        return self

    def _attach(self, pred: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        if seed is None:
            quantiles, n_cal = self.quantiles, len(self.residuals)
        else:
            quantiles, n_cal = self.quantiles_by_seed[seed], len(self.residuals_by_seed[seed])
        out = pred.copy()
        mean = out["mean_logD"].to_numpy(dtype=float)
        for lv in LEVELS:
            pct = int(round(lv * 100))
            q = quantiles[lv]
            out[f"lower_{pct}"] = mean - q
            out[f"upper_{pct}"] = mean + q
            out[f"conformal_q{pct}"] = q
        out["conformal_n_calibration"] = int(n_cal)
        return out

    def _require_fit(self) -> None:
        if self.fitted_arm is None:
            raise RuntimeError("fit first")

    def _by_seed(self, base: pd.DataFrame) -> dict[int, pd.DataFrame]:
        if self.seeds is None:
            return {self.fit_seed: self._attach(base)}
        return {s: self._attach(base, s) for s in self.seeds}

    @staticmethod
    def _long(frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
        return pd.concat([f.assign(conformal_seed=s) for s, f in frames.items()], ignore_index=True)

    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        """Single-seed mode: one row per query (:data:`PREDICTION_COLUMNS`).  Multi-seed mode: the per-seed frames
        concatenated in seed order with a ``conformal_seed`` column."""
        self._require_fit()
        base = self.fitted_arm.predict(query_rows)
        return self._attach(base) if self.seeds is None else self._long(self._by_seed(base))

    def predict_positions(self, positions: np.ndarray) -> pd.DataFrame:
        """As :meth:`predict`, on table positions."""
        self._require_fit()
        base = self.fitted_arm.predict_positions(positions)
        return self._attach(base) if self.seeds is None else self._long(self._by_seed(base))

    def predict_by_seed(self, query_rows: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """``{seed: prediction frame}`` (single-seed mode: ``{context.seed: frame}``); every frame has the same point
        predictions and its seed's intervals."""
        self._require_fit()
        return self._by_seed(self.fitted_arm.predict(query_rows))

    def predict_positions_by_seed(self, positions: np.ndarray) -> dict[int, pd.DataFrame]:
        """As :meth:`predict_by_seed`, on table positions."""
        self._require_fit()
        return self._by_seed(self.fitted_arm.predict_positions(positions))

    def seed_interval_metrics(self, positions: np.ndarray, y: Any, *, units: Any = None) -> tuple[dict[int, pd.DataFrame],
                                                                                                pd.DataFrame]:
        """``(per-seed interval frames, coverage / width per seed and their mean over seeds)`` on scored positions.
        ``y`` (and optional ``units``, the section 4 averaging unit) hold one value per position."""
        frames = self.predict_positions_by_seed(positions)
        idx = pd.Index(np.asarray(positions, dtype=np.int64), name="position")
        yv = pd.Series(np.asarray(y, dtype=float), index=idx)
        uv = None if units is None else pd.Series(np.asarray(units, dtype=object), index=idx)
        from gen19ct.evaluation import calibration as EC

        aligned = {s: f.set_index(idx) for s, f in frames.items()}
        return frames, EC.seed_mean_interval_metrics(yv, aligned, units=uv)
