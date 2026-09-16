"""``models/baselines.py`` -- the closed-form lookup baselines B0-B4 of ``preregistration_draft.md`` section 5.

Every arm is fitted on the outer-training rows of its fold only and records, per prediction, the level it used
(``fallback_level``) and why an earlier level was skipped (``fallback_reason``).  No arm has a seed: identical inputs
give identical outputs whatever the row order (ties are resolved by labels and ids, never by position).

Levels (the registered definitions; interpretation choices are listed in :data:`REGISTRATION_CHOICES`)
-------------------------------------------------------------------------------------------------------
``B0``   row-weighted mean of training ``log_D``.
``B1``   mean over training rows of the query's ``g19_metal_state``; else the element's unknown-state rows; else B0.
``B2``   mean over training rows of the query's ``extractant_system_key``; else rows of systems of the same
         ``system_family``; else rows of systems of the same brief expert (mechanism group); else B0.
``B3``   gen18 ``NearestConditionD`` as used by its R1 evaluation (raw 1-NN, no depletion, no tracer lift).  Candidates:
         training rows with the query's system, metal state and ``acid_anion`` -- else the same system and state with
         the anion dropped (``anion_dropped``).  Distance: unstandardised Euclidean in (``log10_acid_M``,
         ``log10_extractant_M``).  Ties: smaller ``|delta log10 acid|``, then smaller ``canonical_measurement_id``
         (string order).  When the query misses a coordinate, or every candidate misses one, the prediction is the
         mean of the candidates; otherwise candidates missing a coordinate are not used.  Undefined without training
         rows of the (system, state) pair.
``B3x``  cross-metal lookup: within the system and the query's anion, the metal is replaced by the nearest-radius
         training metal state -- ``SupportIndex._nearest_radius`` applied to the metal states with training rows under
         (system, anion) (pool: same formal and aqueous-species charge, else same formal charge, else all; CN8 when the
         query and a pool member have CN8 radii, else CN6; ties by label) -- and B3 is applied to (system, that metal).
         When no other metal state has a training row with the query's anion, the pool is taken over every anion of the
         system (``radius_pool_anion_dropped``).
``B3i``  radius interpolation: B3 on the bracketing same-charge metals (``radius_bracket_lower_metal`` /
         ``radius_bracket_upper_metal`` of the same call), linear in the radius of the B3x basis; equals B3x
         (``fallback_reason = not_bracketed``) where the query is not bracketed.
``B3l``  cross-ligand lookup: within the query's metal state, the system is replaced by the training system with the
         highest Morgan-r2 Tanimoto on the primary-extractant SMILES; ties (exact equality) by smaller ``d_desc``
         (Euclidean in ``mw``, ``mol_logp``, ``rotatable_bonds`` of the primary component, standardised over the
         training systems), then same ``system_family``, then more training rows of that metal, then key order; then B3.
``B4``, ``B4x``, ``B4l``  B3 / B3x / B3l with every training row of the query row's publication group removed; the
         nearest-radius metal and the nearest system are re-chosen after the removal.

Relation to gen18 (``tests/test_baselines.py`` cross-check on gen18 system ``sys_5cb78e5000d40860``, TODGA/nitrate)
    On gen18's replicate-aggregated points, B3 picks exactly the neighbour of ``gen18proc.evalproto._nearest_index``
    for every leave-one-publication-out partition.  On raw archive rows three differences remain, all by design:
    (1) gen18 averages each replicate group before the lookup, B3 takes one row of it by the registered id tie
    rule; (2) equidistant neighbours are ordered by ``canonical_measurement_id`` here, by gen18 ``record_id`` there;
    (3) gen18 keys metals by element, so it also uses rows without a recorded oxidation state, which gen19 never
    uses as a state.  The 1-NN distance itself always agrees.

Arms and their chains (a level that is undefined passes to the next)::

    B0: B0          B1: B1 (-> B0)        B2: B2 (-> B0)
    B3: B3 -> B3x -> B3l -> B2            B4: the B3 chain on the reduced training set
    B3x: B3 -> B3x -> B2                  B4x: the B3x chain on the reduced training set
    B3i: B3 -> B3i -> B2
    B3l: B3 -> B3l -> B2                  B4l: the B3l chain on the reduced training set
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs

from gen19ct.chemistry import support_graph as SG
from gen19ct.models import interface as I

ARM_NAMES: tuple[str, ...] = ("B0", "B1", "B2", "B3", "B3x", "B3i", "B3l", "B4", "B4x", "B4l")
CHAINS: dict[str, tuple[str, ...]] = {
    "B0": ("B0",), "B1": ("B1",), "B2": ("B2",),
    "B3": ("B3", "B3x", "B3l", "B2"), "B3x": ("B3", "B3x", "B2"), "B3i": ("B3", "B3i", "B2"),
    "B3l": ("B3", "B3l", "B2"),
    "B4": ("B3", "B3x", "B3l", "B2"), "B4x": ("B3", "B3x", "B2"), "B4l": ("B3", "B3l", "B2"),
}
SAME_SOURCE_EXCLUDED = frozenset({"B4", "B4x", "B4l"})
_B4_LEVEL = {"B3": "B4", "B3x": "B4x", "B3l": "B4l"}
_RADIUS_NS = SimpleNamespace(radius_tol=SG.RADIUS_NEIGHBOUR_TOL_A)
_EMPTY = np.zeros(0, dtype=np.int64)
_BASE = ("base",)

#: where the registration left a choice open, the reading implemented here
REGISTRATION_CHOICES: dict[str, str] = {
    "B2_expert": "the pseudo-expert none_unknown (mechanism UNKNOWN) is not a group: B2 goes from family to B0",
    "B3_missing_coordinate": "query missing a coordinate or every candidate missing one -> mean of the candidates; "
                             "otherwise candidates missing a coordinate are dropped (no MODEL row misses one)",
    "B3_anion_NA": "a missing acid_anion is the category 'NA'",
    "B3x_pool_anion": "the nearest-radius pool is the metal states with training rows under (system, query anion); "
                      "all anions of the system only when none has one (flag radius_pool_anion_dropped)",
    "B3l_anion": "B3l chooses the system over every anion; the anion rule is B3's, inside the chosen system",
    "chains": "B3 -> B3x -> B3l -> B2; each variant arm is B3 -> variant -> B2 (the exact pair is used when it exists)",
    "B4_scope": "the publication-group removal applies to the whole chain, B2 and B0 fallbacks included",
    "B1_B2_B0_rows": "B0, B2 and the B1 element fallback use unknown-state rows; B1 and every B3 pool use known states",
}


class LookupEngine:
    """Masked lookups over a :class:`~gen19ct.models.interface.RowTable` (one fit)."""

    def __init__(self, table: I.RowTable, mask: np.ndarray, forbidden: np.ndarray):
        self.t = table
        self.mask = mask
        self.forbidden = forbidden
        self._masks: dict[tuple, np.ndarray] = {_BASE: mask}
        self._cache: dict[tuple, Any] = {}

    # ----------------------------------------------------------------------------------------- #
    def mask_for(self, mk: tuple) -> np.ndarray:
        if mk not in self._masks:
            gc = self.t.pub_code.get(mk[1])
            self._masks[mk] = self.mask if gc is None else self.mask & (self.t.pub != gc)
        return self._masks[mk]

    def pool(self, positions: np.ndarray | None, mk: tuple) -> np.ndarray:
        """Training positions among ``positions``; asserts no hidden row enters a candidate pool."""
        if positions is None or not len(positions):
            return _EMPTY
        p = positions[self.mask_for(mk)[positions]]
        if len(p) and self.forbidden[p].any():
            raise AssertionError("a hidden row entered a candidate pool")
        return p

    def _mean(self, p: np.ndarray) -> float:
        """Mean summed in ``canonical_measurement_id`` order, so it does not depend on the row order of the frame."""
        return float(np.mean(self.t.y[p[np.argsort(self.t.id_rank[p], kind="stable")]]))

    # ----------------------------------------------------------------------------------------- #
    def level_B0(self, q: I.Queries, i: int, mk: tuple) -> dict[str, Any]:
        key = ("B0", mk)
        if key not in self._cache:
            p = np.flatnonzero(self.mask_for(mk))
            if not len(p):
                raise ValueError("no training rows")
            if self.forbidden[p].any():
                raise AssertionError("a hidden row entered a candidate pool")
            self._cache[key] = (self._mean(p), len(p))
        v, n = self._cache[key]
        return {"value": v, "level": "B0", "n_candidates": n}

    def level_B1(self, q: I.Queries, i: int, mk: tuple) -> dict[str, Any]:
        mc = self.t.state_code.get(q.state[i]) if q.state[i] is not None else None
        if mc is not None:
            p = self.pool(self.t.by_state.get(mc), mk)
            if len(p):
                return {"value": self._mean(p), "level": "B1:state", "n_candidates": len(p)}
        ec = self.t.elem_code.get(q.elem[i]) if q.elem[i] is not None else None
        if ec is not None:
            p = self.pool(self.t.by_unknown_elem.get(ec), mk)
            if len(p):
                return {"value": self._mean(p), "level": "B1:element_unknown_state", "n_candidates": len(p),
                        "fallback_reason": "no_state_rows"}
        r = self.level_B0(q, i, mk)
        return {**r, "fallback_reason": "no_state_or_element_rows"}

    def level_B2(self, q: I.Queries, i: int, mk: tuple) -> dict[str, Any]:
        t = self.t
        sc = t.sys_code.get(q.system[i]) if q.system[i] is not None else None
        if sc is not None:
            p = self.pool(t.by_s.get(sc), mk)
            if len(p):
                return {"value": self._mean(p), "level": "B2:system", "n_candidates": len(p)}
        st = t.static(q.system[i])
        fc = t.family_code.get(st["family"]) if st["family"] is not None else None
        if fc is not None:
            p = self.pool(t.by_family.get(fc), mk)
            if len(p):
                return {"value": self._mean(p), "level": "B2:family", "n_candidates": len(p),
                        "fallback_reason": "system_absent"}
        xc = t.expert_code.get(st["expert"]) if st["expert"] not in (None, I.NO_EXPERT) else None
        if xc is not None:
            p = self.pool(t.by_expert.get(xc), mk)
            if len(p):
                return {"value": self._mean(p), "level": "B2:expert", "n_candidates": len(p),
                        "fallback_reason": "family_absent"}
        return {**self.level_B0(q, i, mk), "fallback_reason": "expert_absent"}

    # ----------------------------------------------------------------------------------------- #
    def _nearest(self, cand: np.ndarray, qa: float, qe: float) -> dict[str, Any]:
        t = self.t
        A, E = t.acid[cand], t.ext[cand]
        fa, fe = np.isfinite(A), np.isfinite(E)
        keep = fa & fe
        if not (np.isfinite(qa) and np.isfinite(qe)) or not fa.any() or not fe.any() or not keep.any():
            return {"value": self._mean(cand), "n_candidates": len(cand), "b3_mode": "mean_missing_coordinate"}
        c = cand[keep]
        dA = t.acid[c] - qa
        dE = t.ext[c] - qe
        dist = np.sqrt(dA ** 2 + dE ** 2)
        j = int(np.lexsort((t.id_rank[c], np.abs(dA), dist))[0])
        pos = int(c[j])
        return {"value": float(t.y[pos]), "n_candidates": len(c), "nn_pos": pos, "nn_distance": float(dist[j])}

    def b3_rule(self, q: I.Queries, i: int, mk: tuple, state: str | None, system: str | None) -> dict[str, Any] | None:
        """The B3 rule within ``(system, state)`` for query ``i`` (None when the pair has no training row)."""
        t = self.t
        if state is None or system is None:
            return None
        sc, mc = t.sys_code.get(system), t.state_code.get(state)
        if sc is None or mc is None:
            return None
        ac = t.anion_code.get(q.anion[i])
        key = ("pool", mk, sc, mc, ac)
        if key not in self._cache:
            cand = self.pool(t.by_sma.get((sc, mc, ac)), mk) if ac is not None else _EMPTY
            dropped = False
            if not len(cand):
                cand, dropped = self.pool(t.by_sm.get((sc, mc)), mk), True
            self._cache[key] = (cand, dropped)
        cand, dropped = self._cache[key]
        if not len(cand):
            return None
        r = self._nearest(cand, q.acid[i], q.ext[i])
        r.update(anion_dropped=dropped, lookup_system=system, lookup_metal=state)
        return r

    def level_B3(self, q: I.Queries, i: int, mk: tuple) -> dict[str, Any] | None:
        r = self.b3_rule(q, i, mk, q.state[i], q.system[i])
        if r is not None:
            r["level"] = "B3"
        return r

    def nearest_metal(self, q: I.Queries, i: int, mk: tuple) -> dict[str, Any] | None:
        """``SupportIndex._nearest_radius`` over the metal states with training rows under (system, anion)."""
        t = self.t
        st, sy = q.state[i], q.system[i]
        if st is None or sy is None or sy not in t.sys_code:
            return None
        sc, ac = t.sys_code[sy], t.anion_code.get(q.anion[i])
        key = ("metal", mk, st, sc, ac)
        if key in self._cache:
            return self._cache[key]

        def pool_states(entries) -> list[str]:
            return sorted(t.state_labels[m] for m, p in entries if t.state_labels[m] != st and len(self.pool(p, mk)))
        states = pool_states(t.states_by_sa.get((sc, ac), [])) if ac is not None else []
        pool_dropped = False
        if not states:
            states, pool_dropped = pool_states(t.states_by_s.get(sc, [])), True
        res = None
        if states:
            props = SG.metal_properties(st)
            out = SG.SupportIndex._nearest_radius(_RADIUS_NS, props, {k: SG.metal_properties(k) for k in states})
            if out["nearest_radius_metal"] is not None:
                pool = ("same_species_charge" if out["nearest_radius_same_species_charge"] else
                        "same_charge" if out["nearest_radius_same_charge"] else "any")
                res = {**out, "pool": pool, "pool_anion_dropped": pool_dropped, "query_props": props}
        self._cache[key] = res
        return res

    @staticmethod
    def _radius_diag(ch: dict[str, Any]) -> dict[str, Any]:
        return {"nearest_radius_metal": ch["nearest_radius_metal"],
                "nearest_radius_distance_A": ch["nearest_radius_distance_A"],
                "nearest_radius_basis": ch["nearest_radius_basis"], "nearest_radius_pool": ch["pool"],
                "radius_pool_anion_dropped": ch["pool_anion_dropped"],
                "bracket_lower_metal": ch["radius_bracket_lower_metal"],
                "bracket_upper_metal": ch["radius_bracket_upper_metal"]}

    def level_B3x(self, q: I.Queries, i: int, mk: tuple) -> dict[str, Any] | None:
        ch = self.nearest_metal(q, i, mk)
        if ch is None:
            return None
        r = self.b3_rule(q, i, mk, ch["nearest_radius_metal"], q.system[i])
        if r is None:
            raise AssertionError("B3x: the nearest-radius metal has no training row under the system")
        r.update(self._radius_diag(ch), level="B3x")
        return r

    def level_B3i(self, q: I.Queries, i: int, mk: tuple) -> dict[str, Any] | None:
        ch = self.nearest_metal(q, i, mk)
        if ch is None:
            return None
        if not ch["radius_bracketed_same_charge"]:
            r = self.level_B3x(q, i, mk)
            r["fallback_reason"] = "not_bracketed"
            return r
        lo, hi = ch["radius_bracket_lower_metal"], ch["radius_bracket_upper_metal"]
        key = "r_cn8" if ch["nearest_radius_basis"] == "CN8" else "r_cn6"
        rq = float(ch["query_props"][key])
        rlo, rhi = float(SG.metal_properties(lo)[key]), float(SG.metal_properties(hi)[key])
        a, b = self.b3_rule(q, i, mk, lo, q.system[i]), self.b3_rule(q, i, mk, hi, q.system[i])
        if a is None or b is None or not (rlo < rq < rhi):
            raise AssertionError("B3i: a bracket metal has no training row or the bracket does not contain the query")
        frac = (rq - rlo) / (rhi - rlo)
        value = a["value"] + frac * (b["value"] - a["value"])
        return {"value": float(value), "level": "B3i", "n_candidates": a["n_candidates"] + b["n_candidates"],
                "anion_dropped": bool(a["anion_dropped"] or b["anion_dropped"]), "lookup_system": q.system[i],
                "lookup_metal": f"{lo}|{hi}", **self._radius_diag(ch), "bracket_lower_logD": a["value"],
                "bracket_upper_logD": b["value"], "interp_fraction": float(frac)}

    def _d_desc_scale(self, mk: tuple) -> tuple[np.ndarray, np.ndarray]:
        key = ("ddesc", mk)
        if key not in self._cache:
            present = np.unique(self.t.sys[self.mask_for(mk) & (self.t.sys >= 0)])
            D = self.t.sys_desc[present] if len(present) else np.zeros((0, len(I.D_DESC_COLUMNS)))
            with np.errstate(all="ignore"):
                mu = np.nanmean(D, axis=0) if len(D) else np.full(D.shape[1], np.nan)
                sd = np.nanstd(D, axis=0) if len(D) else np.full(D.shape[1], np.nan)
            sd = np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
            self._cache[key] = (mu, sd)
        return self._cache[key]

    def nearest_system(self, q: I.Queries, i: int, mk: tuple) -> dict[str, Any] | None:
        t = self.t
        st, sy = q.state[i], q.system[i]
        mc = t.state_code.get(st) if st is not None else None
        if mc is None:
            return None
        key = ("system", mk, mc, sy)
        if key in self._cache:
            return self._cache[key]
        qs = t.static(sy)
        res = None
        cands = []
        for s, p in t.systems_by_state.get(mc, []):
            if t.sys_labels[s] == sy:
                continue
            n = len(self.pool(p, mk))
            if n and t.sys_fp[s] is not None:
                cands.append((t.sys_labels[s], s, n))
        if qs["fp"] is not None and cands:
            cands.sort(key=lambda c: c[0])
            sims = DataStructs.BulkTanimotoSimilarity(qs["fp"], [t.sys_fp[s] for _, s, _ in cands])
            best = max(sims)
            ties = [(c, sim) for c, sim in zip(cands, sims) if sim == best]
            d_by = {}
            if len(ties) > 1:
                if t.components is None:
                    raise ValueError("B3l: a Tanimoto tie needs d_desc; RowTable.components is missing")
                mu, sd = self._d_desc_scale(mk)
                zq = (qs["desc"] - mu) / sd
                for (lab, s, _), _sim in ties:
                    d = float(np.sqrt(np.sum(((t.sys_desc[s] - mu) / sd - zq) ** 2)))
                    d_by[lab] = d if np.isfinite(d) else float("inf")

            def order(c: tuple) -> tuple:
                lab, s, n = c[0]
                same_family = qs["family"] is not None and t.sys_family[s] == qs["family"]
                return (d_by.get(lab, 0.0), 0 if same_family else 1, -n, lab)
            (lab, s, n), sim = min(ties, key=order)
            res = {"system": lab, "tanimoto": float(sim), "n_ties": len(ties),
                   "d_desc": d_by.get(lab, np.nan) if len(ties) > 1 else np.nan}
        self._cache[key] = res
        return res

    def level_B3l(self, q: I.Queries, i: int, mk: tuple) -> dict[str, Any] | None:
        ch = self.nearest_system(q, i, mk)
        if ch is None:
            return None
        r = self.b3_rule(q, i, mk, q.state[i], ch["system"])
        if r is None:
            raise AssertionError("B3l: the nearest system has no training row of the metal state")
        d = ch["d_desc"]
        r.update(level="B3l", nearest_ligand_system=ch["system"], nearest_ligand_tanimoto=ch["tanimoto"],
                 nearest_ligand_d_desc=(d if np.isfinite(d) else np.nan), n_ligand_ties=ch["n_ties"])
        return r

    # ----------------------------------------------------------------------------------------- #
    def run_chain(self, q: I.Queries, i: int, chain: tuple[str, ...], mk: tuple) -> dict[str, Any]:
        skipped = []
        for level in chain:
            r = getattr(self, f"level_{level}")(q, i, mk)
            if r is not None:
                if skipped and not r.get("fallback_reason"):
                    r["fallback_reason"] = f"{skipped[-1]}_undefined"
                elif skipped:
                    r["fallback_reason"] = f"{skipped[-1]}_undefined;{r['fallback_reason']}"
                return r
            skipped.append(level)
        raise AssertionError(f"chain {chain} ended without a value")   # B0 / B2 are always defined

    def record(self, q: I.Queries, i: int, r: dict[str, Any], same_source_excluded: bool, mk: tuple) -> dict[str, Any]:
        rec = I.empty_prediction_record()
        rec["row_id"] = q.labels[i]
        rec["mean_logD"] = r["value"]
        level = r["level"]
        rec["fallback_level"] = _B4_LEVEL.get(level, level) if same_source_excluded else level
        for k in ("fallback_reason", "anion_dropped", "n_candidates", "nn_distance", "lookup_system", "lookup_metal",
                  "nearest_radius_metal", "nearest_radius_distance_A", "nearest_radius_basis", "nearest_radius_pool",
                  "radius_pool_anion_dropped", "bracket_lower_metal", "bracket_upper_metal", "bracket_lower_logD",
                  "bracket_upper_logD", "interp_fraction", "nearest_ligand_system", "nearest_ligand_tanimoto",
                  "nearest_ligand_d_desc", "n_ligand_ties"):
            if k in r:
                rec[k] = r[k]
        if r.get("b3_mode"):
            rec["fallback_reason"] = ";".join(x for x in (rec["fallback_reason"], r["b3_mode"]) if x)
        if "nn_pos" in r:
            rec["nn_row_id"] = self.t.index[r["nn_pos"]]
            rec["nn_canonical_measurement_id"] = self.t.ids[r["nn_pos"]]
        if same_source_excluded:
            rec["excluded_pub_group"] = mk[1]
        return rec


class BaselineArm:
    """One of :data:`ARM_NAMES` (the module docstring)."""

    def __init__(self, name: str):
        if name not in CHAINS:
            raise ValueError(f"unknown baseline {name!r}; expected one of {ARM_NAMES}")
        self.name = name
        self.chain = CHAINS[name]
        self.same_source_excluded = name in SAME_SOURCE_EXCLUDED
        self.engine: LookupEngine | None = None
        self.context: I.FitContext | None = None

    def clone(self) -> "BaselineArm":
        return BaselineArm(self.name)

    def fit(self, train_rows: pd.DataFrame, context: I.FitContext) -> "BaselineArm":
        table, mask = I.table_and_mask(train_rows, context)
        return self.fit_table(table, mask, context)

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "BaselineArm":
        self.engine = make_engine(table, mask, context)
        self.context = context
        return self

    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        eng = self._engine()
        return self._predict(I.Queries.from_frame(query_rows, eng.t))

    def predict_positions(self, positions: np.ndarray) -> pd.DataFrame:
        eng = self._engine()
        return self._predict(I.Queries.from_positions(eng.t, positions))

    def _engine(self) -> LookupEngine:
        if self.engine is None:
            raise RuntimeError(f"{self.name}: fit first")
        return self.engine

    def _predict(self, q: I.Queries) -> pd.DataFrame:
        eng = self._engine()
        check_queries(eng, q)
        recs = []
        for i in range(len(q)):
            mk = query_mask_key(eng, q, i, self.same_source_excluded)
            r = eng.run_chain(q, i, self.chain, mk)
            recs.append(eng.record(q, i, r, self.same_source_excluded, mk))
        return I.records_to_frame(recs)


def make_engine(table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> LookupEngine:
    """Validate a fit's training mask (finite targets, no hidden row) and build its engine."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (table.n,):
        raise ValueError("training mask does not match the RowTable")
    if not mask.any():
        raise ValueError("no training rows")
    if not np.isfinite(table.y[mask]).all():
        raise ValueError("training rows with a non-finite log_D")
    forbidden = I.forbidden_mask(table, context)
    if (mask & forbidden).any():
        raise AssertionError(f"{int((mask & forbidden).sum())} hidden row(s) among the training rows")
    return LookupEngine(table, mask.copy(), forbidden)


def check_queries(engine: LookupEngine, q: I.Queries) -> None:
    pos = q.table_pos[q.table_pos >= 0]
    if len(pos) and engine.mask[pos].any():
        raise AssertionError("a query row is one of the training rows")


def query_mask_key(engine: LookupEngine, q: I.Queries, i: int, same_source_excluded: bool) -> tuple:
    if not same_source_excluded:
        return _BASE
    g = q.pub_group[i]
    if g is None:
        raise ValueError(f"B4: query row {q.labels[i]!r} has no publication group")
    return ("excluding", g)


def make_arm(name: str) -> BaselineArm:
    return BaselineArm(name)


def all_arms() -> list[BaselineArm]:
    return [BaselineArm(n) for n in ARM_NAMES]
