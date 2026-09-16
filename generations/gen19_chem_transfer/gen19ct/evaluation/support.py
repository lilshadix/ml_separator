"""``evaluation/support.py`` -- pre-registration section 13: the per-fold condition thresholds, the ``support_score v1``
components and the domain-status labels of a prediction.

Moved here from ``scripts/g19_run_preseal.py`` (verification finding VL-05) so that every run -- the pre-seal
baselines, the fold builder (which writes the per-fold thresholds beside the fold hash) and any later arm -- uses one
tested implementation.  Everything here is target-free: it reads training-row conditions and the
``SupportIndex.features`` counts, never ``log_D``.

:func:`tau_thresholds`
    ``tau_in`` / ``tau_ext`` / ``tau_max`` = the 50th / 95th / 99.5th percentiles (``numpy.percentile``, linear
    interpolation) of the leave-one-row-out ``condition_distance_pair`` over the training rows whose (metal state,
    system) pair has >= 2 training rows, on the fold ``SupportIndex``'s standardised scale.  A training value missing
    in a row is imputed as in ``SupportIndex`` (the training mean); a dimension missing in the query row is skipped.
:func:`support_components` / :func:`support_score`
    s1-s8 of the section 13 table and their mean over the available (finite) components.
:func:`s4_registered` / :func:`s4_component` / :func:`s4_candidate_counts`
    s4 = max over training systems measured with the metal of Tanimoto x exp(-d_desc / 2).  Reading resolved by the
    orchestrator on 2026-09-15 (section 13, not score-driven; :data:`S4_READING`): the query's own system counts when
    it has training rows of the query's metal state (so under V5, where the hidden cell has none, it never counts), and
    a system with an undefined ``d_desc`` -- or an undefined fingerprint on either side -- is skipped (undefined
    similarity is no support).  ``support_score`` is therefore defined on every design, V1 and V0 included.
:func:`domain_status`
    the eight section 13 labels, first match wins (:data:`DOMAIN_STATUS_ORDER`).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gen19ct.chemistry import support_graph as SG

#: section 13 label order (first match wins)
DOMAIN_STATUS_ORDER: tuple[str, ...] = ("UNSUPPORTED", "FAMILY_EXTRAPOLATION", "CROSS_METAL_LIGAND_TRANSFER",
                                        "CROSS_LIGAND_TRANSFER", "CROSS_METAL_TRANSFER", "CONDITION_EXTRAPOLATION",
                                        "IN_DOMAIN", "INTERPOLATION")
#: section 13 threshold percentiles
TAU_PERCENTILES: tuple[float, float, float] = (50.0, 95.0, 99.5)
#: the pseudo-expert of mechanism UNKNOWN (``models.interface.NO_EXPERT``); it is not a mechanism group
NO_EXPERT = "none_unknown"
_CHUNK = 400


def tau_thresholds(si: SG.SupportIndex, train: pd.DataFrame) -> tuple[float, float, float, int]:
    """``(tau_in, tau_ext, tau_max, n_rows)`` of one fold (module docstring).  ``train`` must be the frame ``si`` was
    fitted on (same rows, same order).  ``(nan, nan, nan, 0)`` when no row qualifies."""
    if len(train) != si.n_train_rows:
        raise ValueError("tau_thresholds: train is not the SupportIndex's training frame")
    st = train[SG.METAL_COL].to_numpy(dtype=object)
    sy = train[SG.SYSTEM_COL].to_numpy(dtype=object)
    raw = np.column_stack([pd.to_numeric(train[c], errors="coerce").to_numpy(dtype=float) for c in SG.CONDITION_DIMS]) \
        if len(train) else np.zeros((0, len(SG.CONDITION_DIMS)))
    raw[~np.isfinite(raw)] = np.nan
    zq = (raw - si.cond_mean) / si.cond_std
    known = np.array([not SG._missing(a) and not SG._missing(b) for a, b in zip(st, sy)], dtype=bool)
    pos = np.flatnonzero(known)
    if not len(pos):
        return float("nan"), float("nan"), float("nan"), 0
    groups = pd.DataFrame({"m": st[pos], "s": sy[pos]}).groupby(["m", "s"], sort=True).indices
    out = []
    for _, gi in groups.items():
        if len(gi) < 2:
            continue
        idx = pos[gi]
        Q, T = zq[idx], si._Z[idx]
        use = np.isfinite(Q)
        for start in range(0, len(idx), _CHUNK):
            q, u = Q[start:start + _CHUNK], use[start:start + _CHUNK]
            diff = np.where(u[:, None, :], np.nan_to_num(q)[:, None, :] - T[None, :, :], 0.0)
            d2 = (diff ** 2).sum(axis=2)
            d2[np.arange(len(q)), start + np.arange(len(q))] = np.inf       # leave the row itself out
            d = np.sqrt(d2.min(axis=1))
            d[~u.any(axis=1)] = np.nan
            out.append(d)
    dist = np.concatenate(out) if out else np.zeros(0)
    dist = dist[np.isfinite(dist)]
    if not len(dist):
        return float("nan"), float("nan"), float("nan"), 0
    p50, p95, p995 = np.percentile(dist, list(TAU_PERCENTILES))
    return float(p50), float(p95), float(p995), int(len(dist))


def fold_tau(train_support_frame: pd.DataFrame, systems: pd.DataFrame | None = None) -> dict[str, Any]:
    """The section 13 thresholds of one outer fold from its training rows (``SupportIndex`` fitted on them), as the
    record the fold builder writes beside the fold hash."""
    si = SG.SupportIndex(train_support_frame, systems=systems)
    t_in, t_ext, t_max, n = tau_thresholds(si, train_support_frame)
    return {"tau_in": t_in, "tau_ext": t_ext, "tau_max": t_max, "n_tau_rows": n}


#: section 13 s4 reading, resolved by the orchestrator on 2026-09-15 (not score-driven)
S4_READING: dict[str, str] = {
    "query_system": "the query's own system counts when it has training rows of the query's metal state (legitimate "
                    "in-domain support; under V5 the hidden cell has none, so it never counts there)",
    "missing_descriptor": "a system with an undefined d_desc (or an undefined fingerprint) is skipped: undefined "
                          "similarity is no support",
    "candidates": "training systems with >= 1 training row of the query's metal state",
}


def s4_candidate_counts(train_states: Iterable[Any], train_systems: Iterable[Any], metal_state: str) -> dict[str, int]:
    """``{system: number of training rows of metal_state}`` over the training rows (unknown states and missing systems
    never count) -- the systems "measured with the metal" of s4, the query's own system among them exactly when it has
    such a row."""
    out: dict[str, int] = {}
    for st, sy in zip(train_states, train_systems):
        if SG._missing(st) or SG._missing(sy) or str(st) != metal_state:
            continue
        out[str(sy)] = out.get(str(sy), 0) + 1
    return dict(sorted(out.items()))


def s4_registered(query_system: str, query_fp: Any, query_desc_z: np.ndarray,
                  candidates: Iterable[tuple[str, Any, np.ndarray, int]]) -> float:
    """The registered s4 (:data:`S4_READING`).  ``candidates``: ``(system key, Morgan fingerprint, standardised d_desc
    vector, training rows of the query's metal state in that system)`` for the training systems; a system with no such
    row is not a candidate, so the query's own system counts exactly when it has one; undefined ``d_desc`` or
    fingerprints are skipped; 0 when nothing is left."""
    use = []
    for key, fp, desc_z, n_state_rows in candidates:
        if int(n_state_rows) < 0:
            raise ValueError("training-row counts must be >= 0")
        if int(n_state_rows) > 0:
            use.append((key, fp, desc_z))
    return s4_component(query_system, query_fp, query_desc_z, use, include_query_system=True, missing_descriptor="skip")


def s4_component(query_system: str, query_fp: Any, query_desc_z: np.ndarray,
                 candidates: Iterable[tuple[str, Any, np.ndarray]], *, include_query_system: bool = True,
                 missing_descriptor: str = "skip") -> float:
    """s4 = max over ``candidates`` -- ``(system key, Morgan fingerprint, standardised d_desc vector)`` of the training
    systems measured with the metal -- of Tanimoto x exp(-d_desc / 2); 0 when there is none.

    The defaults are the registered reading (:data:`S4_READING`, resolved 2026-09-15) provided ``candidates`` holds
    only systems with training rows of the query's metal state (:func:`s4_registered` enforces that).  The switches
    remain only to reproduce the pre-seal ambiguity report: ``include_query_system=False`` drops the query's own system,
    ``missing_descriptor="zero"`` treats an undefined ``d_desc`` as 0 -- both unregistered readings."""
    from rdkit import DataStructs

    if missing_descriptor not in ("skip", "zero"):
        raise ValueError(f"missing_descriptor {missing_descriptor!r}")
    vals = []
    for key, fp, desc_z in candidates:
        if not include_query_system and key == query_system:
            continue
        if fp is None or query_fp is None:
            continue
        sim = float(DataStructs.TanimotoSimilarity(query_fp, fp))
        d = float(np.sqrt(np.sum((np.asarray(desc_z, dtype=float) - np.asarray(query_desc_z, dtype=float)) ** 2)))
        if not np.isfinite(d):
            if missing_descriptor == "skip":
                continue
            d = 0.0
        vals.append(sim * math.exp(-d / 2.0))
    return max(vals) if vals else 0.0


def support_components(f: Mapping[str, Any], *, s4: float, tau_ext: float, metal_series: str | None) -> dict[str, float]:
    """s1-s8 of section 13 from ``SupportIndex.features`` output ``f`` (s8 is NaN outside the Ln and An series)."""
    exact = int(f["exact_pair_rows"])
    cd = f["condition_distance_pair"] if exact > 0 else f["condition_distance_system"]
    radius_ok = f["nearest_radius_metal"] is not None and np.isfinite(f["nearest_radius_distance_A"])
    return {
        "s1": min(1.0, math.log1p(exact) / math.log1p(20)),
        "s2": min(1.0, f["n_neighbour_metals_same_charge"] / 5.0),
        "s3": math.exp(-f["nearest_radius_distance_A"] / 0.05) if radius_ok else 0.0,
        "s4": float(s4),
        "s5": min(1.0, math.log1p(f["n_same_family_rows_for_metal"]) / math.log1p(100)),
        "s6": min(1.0, f["n_publications_system"] / 5.0),
        "s7": 0.5 if not np.isfinite(cd) else math.exp(-cd / tau_ext),
        "s8": ((1.0 if f["series_bracketed"] else 0.5 if f["n_series_neighbours_pm1"] >= 1 else 0.0)
               if metal_series in ("Ln", "An") else float("nan")),
    }


def support_score(components: Mapping[str, float]) -> float:
    """``support_score v1``: the mean of the finite components (NaN when none is finite)."""
    vals = [float(v) for v in components.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def domain_status(*, system_present: bool, metal_present: bool, family_rows: int, mechanism_rows: int,
                  expert: str | None, f: Mapping[str, Any], fam_cd: float, tau: Sequence[float],
                  anion_unseen: bool) -> tuple[str, bool, bool, bool]:
    """Section 13 labels, first match wins: ``(status, condition_extrapolated, both_nodes_new, conditions_unknown)``.

    ``fam_cd`` is the condition distance to the nearest training row of the query's family (used when the system is
    absent); ``tau`` = ``(tau_in, tau_ext, tau_max)``; ``anion_unseen`` is the caller's reading of "the query's acid
    anion is never seen with the system or family in training"."""
    tau_in, tau_ext, tau_max = tau
    exact = int(f["exact_pair_rows"])
    cd_near = f["condition_distance_system"] if system_present else fam_cd
    cdp = f["condition_distance_pair"]
    pm1 = int(f["n_series_neighbours_pm1"])
    same_charge = bool(f["nearest_radius_same_charge"])
    cext = bool(np.isfinite(f["condition_distance_system"]) and f["condition_distance_system"] > tau_ext)
    unsupported = (mechanism_rows == 0 or expert in (None, NO_EXPERT)
                   or (not metal_present and not same_charge and pm1 == 0)
                   or (not system_present and family_rows == 0 and not metal_present)
                   or (np.isfinite(cd_near) and cd_near > tau_max)
                   or anion_unseen)
    if unsupported:
        return "UNSUPPORTED", False, False, False
    if not system_present and family_rows == 0:
        return "FAMILY_EXTRAPOLATION", False, False, False
    if exact == 0 and system_present and metal_present:
        return "CROSS_METAL_LIGAND_TRANSFER", cext, False, False
    if not system_present and not metal_present and family_rows > 0:
        return "CROSS_METAL_LIGAND_TRANSFER", cext, True, False
    if not system_present and family_rows > 0 and metal_present:
        return "CROSS_LIGAND_TRANSFER", cext, False, False
    if system_present and not metal_present and (same_charge or pm1 >= 1):
        return "CROSS_METAL_TRANSFER", cext, False, False
    if exact >= 1 and np.isfinite(cdp) and cdp > tau_ext:
        return "CONDITION_EXTRAPOLATION", False, False, False
    if exact >= 5 and np.isfinite(cdp) and cdp <= tau_in:
        return "IN_DOMAIN", False, False, False
    if exact >= 1:
        return "INTERPOLATION", False, False, not np.isfinite(cdp)
    raise AssertionError("domain status: no rule matched")
