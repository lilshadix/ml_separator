"""``optimize.py`` -- the regime design space, LHS + Pareto search, epsilon-constraint knees and
the gated GP-BO (DESIGN.md section 9; addendum WB4b).

* ``DesignSpace`` (9.1): ``bounds`` per variable, the ``integer`` variables (stage counts and the
  two stage offsets, drawn as ``floor(u (hi - lo + 1)) + lo`` clipped to ``[lo, hi]``), ``fixed``
  values, and -- an addition recorded in ``addenda/WB4b.md`` -- ``levels`` (a variable restricted
  to a finite set of values, drawn as an index; the case study's saponification degree
  ``{0, 0.3, 0.5}``).  ``DesignSpace.from_config`` builds the default space from
  ``config/design_spaces.json`` (stage counts, ratios, family overrides) and from the parameter
  set's ``ApplicabilityDomain`` (the ``from_domain`` concentration axes); ``widen`` and ``fixed``
  are the user's overrides.  Every candidate outside the recorded intervals carries its OOD flags
  from the cascade; the space itself makes no chemical claim.
* ``candidate_spec`` maps decoded variables onto a ``CascadeSpec`` (units: stage counts
  dimensionless; ``oa_ext``, ``s_over_a``, ``w_over_a`` volumetric ratios; acids and anions
  mol/L; ``scrub_target_mM`` mM; ``ligand_total_M`` the formal monomer concentration, halved for
  a dimeric ligand through the model's ``ligand_scale``; ``feed_dilution`` volumes of water per
  volume of feed; ``f_bleed`` and ``saponification_degree`` fractions).
* ``lhs_pareto`` (9.2): ``scipy.stats.qmc.LatinHypercube(d, scramble=True, seed)``, one cascade
  per row (``solve_cascade`` then ``compute_metrics``); ``invalid_spec`` (``ValueError`` from the
  cascade) and ``failed`` rows are kept and counted (``df.attrs["n_failed"]``,
  ``df.attrs["n_invalid_spec"]``); non-dominated rank ``front`` on (purity_mol up,
  recovery_from_feed up, consumption_index down, n_stages_total down) -- ``cost_proxy`` replaces
  the consumption index under ``objective="cost"`` -- computed over ``all`` rows and again over the
  in-domain rows (``front_in_domain``); rows sorted by (front, variable tuple) so the CSV is
  byte-identical across runs.  ``consumption_index = acid + base + complexant`` mol per kg of
  target oxide, NaN components excluded and flagged (``consumption_index_incomplete``).
* ``pareto_fronts`` returns the first front twice (``in_domain_only`` and ``all``);
  ``epsilon_knees`` the minimum-consumption regime at every (purity_min, recovery_min) cell of a
  spec grid (both subsets).
* ``gp_bo`` (9.3): ParEGO -- random Chebyshev scalarisation per iteration, a Matern(nu = 2.5) ARD
  + WhiteKernel ``GaussianProcessRegressor`` on the unit hypercube, expected improvement over 2000
  LHS candidates, kriging believer inside a batch.  Gated: it runs only when the D source's
  reliability flags pass (PRE_REGISTRATION.md section 6: both slopes ``interpretable``); otherwise
  it returns ``BOResult(status="gated")`` without evaluating a cascade.  Never a headline number.

Sources: DESIGN.md sections 9, 13.2; addenda WB2 (``ligand_scale``), WB3 (``compute_metrics``
keywords, ``ValueError`` on malformed specs, failure as a status).
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import qmc

from gen18proc.cascade import solve_cascade
from gen18proc.dmodel import SystemModel
from gen18proc.metrics import CONSUMPTION_ITEMS, ITEM_UNITS, Prices, compute_metrics
from gen18proc.types import (
    OOD_FLAGS,
    ApplicabilityDomain,
    AqStream,
    CascadeSpec,
    Flag,
    OrgStream,
    ProcessMetrics,
)

__all__ = [
    "DesignSpace", "VARIABLE_NAMES", "IN_DOMAIN_STATUSES", "OBJECTIVES",
    "candidate_spec", "evaluate_candidate", "lhs_pareto", "pareto_rank", "pareto_fronts",
    "epsilon_knees", "consumption_index", "gp_bo", "BOResult", "reliability_passes",
    "load_design_space_config",
]

VARIABLE_NAMES: tuple[str, ...] = (
    "n_ext", "n_scr", "n_str", "feed_stage_offset", "scrub_return_offset", "oa_ext", "s_over_a",
    "w_over_a", "scrub_acid_M", "scrub_target_mM", "scrub_complexant_M", "strip_acid_M",
    "strip_anion_M", "ligand_total_M", "saponification_degree", "feed_dilution", "f_bleed",
)
"""The design variables of DESIGN.md section 9.1, in the design's order."""

IN_DOMAIN_STATUSES: frozenset[str] = frozenset({"IN_DOMAIN", "IN_DOMAIN_WITH_CAVEATS"})
"""Regime statuses that count as in-domain for the ``in_domain_only`` front (section 9.2)."""

OBJECTIVES: dict[str, tuple[tuple[str, int], ...]] = {
    "consumption": (("purity_mol", -1), ("recovery_from_feed", -1), ("consumption_index", 1),
                    ("n_stages_total", 1)),
    "cost": (("purity_mol", -1), ("recovery_from_feed", -1), ("cost_proxy_per_kg_oxide", 1),
             ("n_stages_total", 1)),
}
"""Objective columns and their sense (``+1`` minimise, ``-1`` maximise) per ``objective``."""

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "design_spaces.json"
_METRIC_SCALARS: tuple[str, ...] = (
    "purity_mol", "purity_mass", "purity_oxide", "recovery_from_feed", "recovery_total",
    "scrub_target_return", "net_product_mol_h", "n_stages_total", "n_ext", "n_scr", "n_str",
    "oa_ext", "s_over_a", "w_over_a", "throughput_mol_T_per_h_per_L_org",
    "throughput_kg_oxide_per_h", "cost_proxy_per_kg_oxide", "on_spec", "regime_status",
)
_STATUS_ORDER = ("converged_newton", "converged_ss", "failed", "invalid_spec")


# =============================================================================================
# 9.1  Design space
# =============================================================================================

def load_design_space_config(path: str | Path | None = None) -> dict[str, Any]:
    """The parsed ``config/design_spaces.json`` (schema ``gen18.design_spaces.1``)."""
    p = Path(path) if path is not None else _CONFIG_PATH
    with p.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if obj.get("schema") != "gen18.design_spaces.1":
        raise ValueError(f"unexpected design-space schema {obj.get('schema')!r} in {p}")
    return obj


@dataclass(frozen=True)
class DesignSpace:
    """The regime design space (DESIGN.md section 9.1).

    ``bounds[name] = (lo, hi)`` for every variable; ``integer`` names the integer variables;
    ``fixed[name] = value`` removes a variable from the sampled dimensions; ``levels[name]`` (an
    addition, addendum WB4b) restricts a variable to a finite value set drawn by index.  Only
    names in ``VARIABLE_NAMES`` are accepted.  The sampled dimensions are ``variables`` (the
    bounded, unfixed names in the order of ``VARIABLE_NAMES``).
    """

    bounds: dict[str, tuple[float, float]]
    integer: tuple[str, ...] = ()
    fixed: dict[str, float] = field(default_factory=dict)
    levels: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in list(self.bounds) + list(self.fixed) + list(self.levels) + list(self.integer):
            if name not in VARIABLE_NAMES:
                raise ValueError(f"unknown design variable {name!r}; known: {VARIABLE_NAMES}")
        for name, (lo, hi) in self.bounds.items():
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError(f"bounds of {name!r} must be finite with lo <= hi, got {lo}, {hi}")
        for name, vals in self.levels.items():
            if len(vals) == 0:
                raise ValueError(f"levels of {name!r} must not be empty")
        object.__setattr__(self, "bounds", {k: (float(v[0]), float(v[1]))
                                            for k, v in self.bounds.items()})
        object.__setattr__(self, "fixed", {k: float(v) for k, v in self.fixed.items()})
        object.__setattr__(self, "levels", {k: tuple(float(x) for x in v)
                                            for k, v in self.levels.items()})
        object.__setattr__(self, "integer", tuple(self.integer))

    @property
    def variables(self) -> tuple[str, ...]:
        """Sampled variables in the design's order (bounded or levelled, not fixed)."""
        return tuple(v for v in VARIABLE_NAMES
                     if (v in self.bounds or v in self.levels) and v not in self.fixed)

    @property
    def dimension(self) -> int:
        return len(self.variables)

    def decode(self, u: Sequence[float] | np.ndarray) -> dict[str, float]:
        """Unit-hypercube coordinates (one per ``variables`` entry) -> variable values, with the
        fixed values appended.  Integers: ``floor(u (hi - lo + 1)) + lo`` clipped to ``[lo, hi]``;
        levels: index ``floor(u * n_levels)`` clipped; continuous: ``lo + u (hi - lo)``."""
        u = np.asarray(u, dtype=float)
        names = self.variables
        if u.shape != (len(names),):
            raise ValueError(f"decode expects {len(names)} coordinates, got shape {u.shape}")
        out: dict[str, float] = {}
        for name, ui in zip(names, u):
            ui = min(max(float(ui), 0.0), 1.0)
            if name in self.levels:
                vals = self.levels[name]
                idx = min(int(math.floor(ui * len(vals))), len(vals) - 1)
                out[name] = float(vals[idx])
            elif name in self.integer:
                lo, hi = self.bounds[name]
                lo_i, hi_i = int(round(lo)), int(round(hi))
                val = int(math.floor(ui * (hi_i - lo_i + 1))) + lo_i
                out[name] = float(min(max(val, lo_i), hi_i))
            else:
                lo, hi = self.bounds[name]
                out[name] = lo + ui * (hi - lo)
        out.update(self.fixed)
        return out

    def with_fixed(self, **fixed: float) -> "DesignSpace":
        """A copy with more fixed variables."""
        return replace(self, fixed={**self.fixed, **{k: float(v) for k, v in fixed.items()}})

    def with_bounds(self, **bounds: tuple[float, float]) -> "DesignSpace":
        """A copy with widened / replaced bounds (the user's widening of section 9.1)."""
        return replace(self, bounds={**self.bounds,
                                     **{k: (float(v[0]), float(v[1])) for k, v in bounds.items()}})

    def to_json(self) -> dict[str, Any]:
        return {"bounds": {k: list(v) for k, v in self.bounds.items()},
                "integer": list(self.integer), "fixed": dict(self.fixed),
                "levels": {k: list(v) for k, v in self.levels.items()},
                "variables": list(self.variables)}

    @classmethod
    def from_config(cls, *, family: str | None = None, domain: ApplicabilityDomain | None = None,
                    ligand: str | None = None, config_path: str | Path | None = None,
                    widen: Mapping[str, tuple[float, float]] | None = None,
                    fixed: Mapping[str, float] | None = None,
                    levels: Mapping[str, Sequence[float]] | None = None) -> "DesignSpace":
        """Default space (section 9.1): stage counts, offsets, ratios and the fallback
        concentration windows from ``config/design_spaces.json`` (``families.<family>.bounds``
        override the defaults); the ``from_domain`` concentration axes take the parameter set's
        ``ApplicabilityDomain`` intervals when ``domain`` is given (``10 ** log_acid`` for the
        scrub and strip acid, ``10 ** log_ligand[ligand]`` for the ligand total,
        ``10 ** log_complexant`` when recorded); then ``widen`` replaces bounds, ``fixed`` pins
        variables and ``levels`` restricts them to value sets."""
        cfg = load_design_space_config(config_path)
        default = cfg["default"]
        bounds = {k: (float(v[0]), float(v[1])) for k, v in default["bounds"].items()}
        integer = tuple(default.get("integer", ()))
        fixed_all = {k: float(v) for k, v in (default.get("fixed") or {}).items()}
        if family and family in (cfg.get("families") or {}):
            for k, v in (cfg["families"][family].get("bounds") or {}).items():
                bounds[k] = (float(v[0]), float(v[1]))
        if domain is not None:
            acid = (10.0 ** domain.log_acid[0], 10.0 ** domain.log_acid[1])
            for k in ("scrub_acid_M", "strip_acid_M"):
                if k in default.get("from_domain", ()):
                    bounds[k] = acid
            lig_name = ligand or (next(iter(domain.log_ligand)) if domain.log_ligand else None)
            if lig_name is not None and lig_name in domain.log_ligand \
                    and "ligand_total_M" in default.get("from_domain", ()):
                lo, hi = domain.log_ligand[lig_name]
                bounds["ligand_total_M"] = (10.0 ** lo, 10.0 ** hi)
            if domain.log_complexant is not None \
                    and "scrub_complexant_M" in default.get("from_domain", ()):
                lo, hi = domain.log_complexant
                bounds["scrub_complexant_M"] = (10.0 ** lo, 10.0 ** hi)
        if widen:
            for k, v in widen.items():
                bounds[k] = (float(v[0]), float(v[1]))
        if fixed:
            fixed_all.update({k: float(v) for k, v in fixed.items()})
        lv = {k: tuple(float(x) for x in v) for k, v in (levels or {}).items()}
        return cls(bounds=bounds, integer=integer, fixed=fixed_all, levels=lv)


# =============================================================================================
# candidate -> CascadeSpec
# =============================================================================================

def _ligand_scale_of(system: SystemModel, ligand: str) -> float:
    md = system.dmodels.get(ligand)
    return float(getattr(md, "ligand_scale", 1.0) or 1.0)


def _lean_organic(system: SystemModel, flow: float, ligand_total: Mapping[str, float],
                  metals: Iterable[str]) -> OrgStream:
    names = tuple(metals)
    zeros = {m: 0.0 for m in names}
    return OrgStream(flow_L_h=flow, metals=dict(zeros),
                     metals_by_ligand={k: dict(zeros) for k in ligand_total},
                     ligand_total=dict(ligand_total), ligand_free=dict(ligand_total),
                     acid_in_org={k: 0.0 for k in ligand_total}, alkali_reserve=0.0)


def candidate_spec(values: Mapping[str, float], base_spec: CascadeSpec, system: SystemModel,
                   target: str) -> tuple[CascadeSpec, dict[str, float]]:
    """The ``CascadeSpec`` of one candidate (variables of ``VARIABLE_NAMES``; a variable absent
    from ``values`` keeps the corresponding field of ``base_spec``).

    Feed: ``feed_dilution`` d dilutes the base feed (flow ``A (1 + d)``, every concentration
    ``/ (1 + d)``); ``O = oa_ext A'``, ``S = s_over_a A'``, ``W = w_over_a A'``.  Scrub liquor:
    ``h = anion = scrub_acid_M`` plus ``3 x`` the displacement target (``scrub_target_mM``, as the
    metal's trivalent salt) and ``scrub_complexant_M``.  Strip liquor: ``h = strip_acid_M``,
    ``anion = strip_acid_M + strip_anion_M`` (the salting anion beyond the acid).  The ligand
    total applies to the system's first ligand (formal monomer concentration / ``ligand_scale``);
    other ligands keep the base values.  ``feed_stage = n_ext - 1 - feed_stage_offset`` and the
    scrub return likewise, clipped to ``[0, n_ext - 1]``.  ``f_bleed > 0`` adds a lean fresh
    organic with the same ligand totals.  Returns the spec and the extras ``compute_metrics``
    needs (``feed_dilution_L_h``) plus ``saponification_degree``.
    """
    v = dict(values)
    feed = base_spec.feed
    d = float(v.get("feed_dilution", 0.0))
    if d < 0:
        raise ValueError("feed_dilution must be >= 0")
    if d > 0:
        f = 1.0 / (1.0 + d)
        feed = AqStream(flow_L_h=feed.flow_L_h * (1.0 + d),
                        metals={m: c * f for m, c in feed.metals.items()}, h=feed.h * f,
                        anion=feed.anion * f, complexant_total=feed.complexant_total * f,
                        sodium=feed.sodium * f)
    A = float(feed.flow_L_h)
    n_ext = int(round(v.get("n_ext", base_spec.n_ext)))
    n_scr = int(round(v.get("n_scr", base_spec.n_scr)))
    n_str = int(round(v.get("n_str", base_spec.n_str)))
    O = float(v["oa_ext"]) * A if "oa_ext" in v else float(base_spec.organic_flow_L_h)
    S = float(v["s_over_a"]) * A if "s_over_a" in v else float(base_spec.scrub.flow_L_h)
    W = float(v["w_over_a"]) * A if "w_over_a" in v else float(base_spec.strip.flow_L_h)
    # scrub liquor
    scrub = base_spec.scrub
    scrub_metals = dict(scrub.metals)
    for m in feed.metals:
        scrub_metals.setdefault(m, 0.0)
    if "scrub_target_mM" in v:
        scrub_metals[target] = float(v["scrub_target_mM"]) * 1e-3
    if "scrub_acid_M" in v:
        s_h = float(v["scrub_acid_M"])
        s_anion = s_h + 3.0 * scrub_metals.get(target, 0.0)
    else:
        s_h, s_anion = float(scrub.h), float(scrub.anion)
        if "scrub_target_mM" in v:
            s_anion = s_h + 3.0 * scrub_metals.get(target, 0.0)
    s_c = float(v["scrub_complexant_M"]) if "scrub_complexant_M" in v else float(
        scrub.complexant_total)
    scrub = AqStream(flow_L_h=S, metals=scrub_metals, h=s_h, anion=s_anion, complexant_total=s_c,
                     sodium=float(scrub.sodium))
    # strip liquor
    strip = base_spec.strip
    strip_metals = dict(strip.metals)
    for m in feed.metals:
        strip_metals.setdefault(m, 0.0)
    w_h = float(v["strip_acid_M"]) if "strip_acid_M" in v else float(strip.h)
    if "strip_anion_M" in v:
        w_anion = w_h + float(v["strip_anion_M"])
    elif "strip_acid_M" in v:
        w_anion = w_h + max(0.0, float(strip.anion) - float(strip.h))
    else:
        w_anion = float(strip.anion)
    strip = AqStream(flow_L_h=W, metals=strip_metals, h=w_h, anion=w_anion,
                     complexant_total=float(strip.complexant_total), sodium=float(strip.sodium))
    # ligand totals (code basis)
    ligand_total = dict(base_spec.ligand_total)
    ligands = tuple(system.dmodels)
    if "ligand_total_M" in v and ligands:
        primary = ligands[0]
        ligand_total[primary] = float(v["ligand_total_M"]) / _ligand_scale_of(system, primary)
    sap = float(v.get("saponification_degree", base_spec.saponification_degree))
    f_bleed = float(v.get("f_bleed", base_spec.f_bleed))
    feed_stage = base_spec.feed_stage
    ret = base_spec.scrub_return_stage
    if "feed_stage_offset" in v or "scrub_return_offset" in v:
        # DESIGN 7.2 / addendum WB3: every extraction stage must be reached by an aqueous stream,
        # so at least one of the two entries is the top extraction stage.  Both offsets are
        # shifted down by their minimum (the relative position of feed and scrub return is kept);
        # with no scrub section the feed itself enters at the top.  The effective offsets are
        # written back into ``values`` so the row records what was run (addendum WB4b).
        fo = int(round(v.get("feed_stage_offset", 0.0)))
        ro = int(round(v.get("scrub_return_offset", 0.0)))
        fo, ro = max(fo, 0), max(ro, 0)
        shift = min(fo, ro)
        fo, ro = fo - shift, ro - shift
        if n_scr == 0:
            fo = 0
        fo = min(fo, n_ext - 1)
        ro = min(ro, n_ext - 1)
        if "feed_stage_offset" in v:
            v["feed_stage_offset"] = float(fo)
            feed_stage = n_ext - 1 - fo
        if "scrub_return_offset" in v:
            v["scrub_return_offset"] = float(ro)
            ret = n_ext - 1 - ro
        if isinstance(values, dict):
            values.update({k: v[k] for k in ("feed_stage_offset", "scrub_return_offset")
                           if k in v})
    fresh = base_spec.fresh_organic
    if f_bleed > 0.0 and fresh is None:
        fresh = _lean_organic(system, O, ligand_total, feed.metals)
    elif fresh is not None:
        fresh = replace(fresh, flow_L_h=O, ligand_total=dict(ligand_total),
                        ligand_free=dict(ligand_total))
    spec = CascadeSpec(n_ext=n_ext, n_scr=n_scr, n_str=n_str, feed=feed, scrub=scrub,
                       strip=strip, organic_flow_L_h=O, ligand_total=ligand_total,
                       saponification_degree=sap, feed_stage=feed_stage,
                       scrub_return_stage=ret, f_bleed=f_bleed, fresh_organic=fresh,
                       target=target or base_spec.target)
    extras = {"feed_dilution_L_h": d * float(base_spec.feed.flow_L_h),
              "saponification_degree": sap}
    return spec, extras


# =============================================================================================
# one evaluation
# =============================================================================================

def consumption_index(consumption: Mapping[str, float]) -> tuple[float, bool]:
    """``acid + base + complexant`` mol per kg of target oxide; NaN components are excluded and
    reported through the ``incomplete`` flag (NaN when every component is NaN)."""
    parts = [consumption.get(f"{item}_mol_per_kg_oxide", math.nan)
             for item in ("acid", "base", "complexant")]
    finite = [p for p in parts if p is not None and math.isfinite(p)]
    if not finite:
        return math.nan, True
    return float(sum(finite)), len(finite) < len(parts)


def _saponification_flag(system: SystemModel, sap: float) -> set[Flag]:
    """``SAPONIFICATION_RANGE_UNKNOWN`` when a non-zero degree is outside (or without) the
    parameter set's studied range (DESIGN.md section 13.2)."""
    if sap <= 0.0:
        return set()
    for md in system.dmodels.values():
        params = getattr(md, "params", None)
        studied = getattr(params, "saponification_degree_studied", None)
        if studied is not None and studied[0] <= sap <= studied[1]:
            continue
        return {Flag.SAPONIFICATION_RANGE_UNKNOWN}
    return set()


def _flag_string(flags: Iterable[Flag | str]) -> str:
    return "|".join(sorted(f.value if isinstance(f, Flag) else str(f) for f in flags))


def _regime_status(flags: set[Flag]) -> str:
    from gen18proc.cascade import regime_status_of

    return regime_status_of(frozenset(flags))


def _metrics_row(metrics: ProcessMetrics, impurities: Sequence[str], extra_flags: set[Flag],
                 ) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name in _METRIC_SCALARS:
        val = getattr(metrics, name)
        row[name] = (math.nan if val is None else val) if name not in ("on_spec", "regime_status") \
            else val
    for imp in impurities:
        row[f"enrichment_factor_{imp}"] = float(metrics.enrichment_factor.get(imp, math.nan))
        ext = metrics.sf_tracer.get("extraction", {})
        row[f"sf_tracer_extraction_{imp}"] = float(ext.get(imp, math.nan))
        scr = metrics.sf_tracer.get("scrub", {})
        row[f"sf_tracer_scrub_{imp}"] = float(scr.get(imp, math.nan))
    for lig, lam in metrics.max_loading_fraction.items():
        row[f"max_loading_{lig}"] = float(lam)
    for item in CONSUMPTION_ITEMS:
        unit = ITEM_UNITS[item]
        for suffix in (f"{unit}_h", f"{unit}_per_kg_oxide"):
            key = f"{item}_{suffix}"
            row[key] = float(metrics.consumption.get(key, math.nan))
    ci, incomplete = consumption_index(metrics.consumption)
    row["consumption_index"] = ci
    row["consumption_index_incomplete"] = bool(incomplete)
    flags = set(metrics.flags) | set(extra_flags)
    row["regime_status"] = _regime_status(flags) if extra_flags else metrics.regime_status
    row["flags"] = _flag_string(flags)
    row["n_flags"] = len(flags)
    row["ood_distance_max"] = float(max(metrics.ood_distance.values())) \
        if metrics.ood_distance else 0.0
    row["in_domain"] = row["regime_status"] in IN_DOMAIN_STATUSES
    return row


def _nan_row(impurities: Sequence[str], ligands: Iterable[str], status: str, reason: str,
             ) -> dict[str, Any]:
    row: dict[str, Any] = {name: math.nan for name in _METRIC_SCALARS}
    row["on_spec"] = False
    row["regime_status"] = "INADMISSIBLE"
    for imp in impurities:
        row[f"enrichment_factor_{imp}"] = math.nan
        row[f"sf_tracer_extraction_{imp}"] = math.nan
        row[f"sf_tracer_scrub_{imp}"] = math.nan
    for lig in ligands:
        row[f"max_loading_{lig}"] = math.nan
    for item in CONSUMPTION_ITEMS:
        unit = ITEM_UNITS[item]
        row[f"{item}_{unit}_h"] = math.nan
        row[f"{item}_{unit}_per_kg_oxide"] = math.nan
    row["consumption_index"] = math.nan
    row["consumption_index_incomplete"] = True
    row["flags"] = Flag.NOT_CONVERGED.value if status == "failed" else ""
    row["n_flags"] = 1 if status == "failed" else 0
    row["ood_distance_max"] = math.nan
    row["in_domain"] = False
    row["status"] = status
    row["reason"] = reason
    return row


def evaluate_candidate(values: Mapping[str, float], base_spec: CascadeSpec, system: SystemModel,
                       target: str, impurities: Sequence[str],
                       spec_limits: Mapping[str, float] | None, prices: Prices | None, *,
                       solver_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One row of the LHS table: the variables, ``status`` (``converged_newton``,
    ``converged_ss``, ``failed``, ``invalid_spec``), every scalar metric, the consumption
    columns, ``consumption_index``, flags, ``regime_status`` and ``in_domain``.  A ``ValueError``
    from the cascade (malformed specification) is recorded as ``invalid_spec`` with its message in
    ``reason``; a failed solve carries NaN metrics; nothing raises."""
    impurities = tuple(impurities)
    vals: dict[str, float] = {k: float(v) for k, v in values.items()}
    row: dict[str, Any] = {}
    try:
        spec, extras = candidate_spec(vals, base_spec, system, target)
        with np.errstate(all="ignore"):
            result = solve_cascade(spec, system, **(solver_kwargs or {}))
    except ValueError as exc:
        row.update(_nan_row(impurities, system.ligands, "invalid_spec", str(exc)))
        row.update(vals)          # the variables (effective offsets) survive the NaN metrics
        row["n_stages_total"] = float(int(round(vals.get("n_ext", base_spec.n_ext)))
                                      + int(round(vals.get("n_scr", base_spec.n_scr)))
                                      + int(round(vals.get("n_str", base_spec.n_str))))
        row["iterations"] = 0
        row["balance_rel_max"] = math.nan
        return row
    with np.errstate(all="ignore"):
        metrics = compute_metrics(result, spec, system, target, impurities, prices,
                                  spec_limits=spec_limits,
                                  feed_dilution_L_h=extras["feed_dilution_L_h"])
    extra_flags = _saponification_flag(system, extras["saponification_degree"])
    row.update(_metrics_row(metrics, impurities, extra_flags))
    row.update(vals)              # variable columns hold the decoded (effective) values
    row["status"] = result.status
    row["reason"] = ""
    row["iterations"] = int(result.iterations)
    row["balance_rel_max"] = float(result.balance_rel_max)
    if result.status == "failed":
        row["in_domain"] = False
    return row


# =============================================================================================
# 9.2  LHS + Pareto
# =============================================================================================

def pareto_rank(objectives: np.ndarray) -> np.ndarray:
    """Non-dominated fronts of ``objectives`` (rows = candidates, columns = objectives to be
    **minimised**): front 1 is non-dominated, 2 the next, ... ; rows with a non-finite objective
    get 0 (they are ranked after every finite row by the caller)."""
    F = np.asarray(objectives, dtype=float)
    n = F.shape[0]
    front = np.zeros(n, dtype=int)
    if n == 0:
        return front
    finite = np.all(np.isfinite(F), axis=1)
    remaining = np.flatnonzero(finite)
    k = 1
    while remaining.size:
        sub = F[remaining]
        # dominated[i] = exists j: sub[j] <= sub[i] (all) and sub[j] < sub[i] (any)
        le = np.all(sub[None, :, :] <= sub[:, None, :], axis=2)
        lt = np.any(sub[None, :, :] < sub[:, None, :], axis=2)
        dominated = np.any(le & lt, axis=1)
        nd = remaining[~dominated]
        front[nd] = k
        remaining = remaining[dominated]
        k += 1
    return front


def _objective_matrix(df: pd.DataFrame, objective: str) -> np.ndarray:
    cols = OBJECTIVES[objective]
    mat = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) * s
                           for c, s in cols]) if len(df) else np.zeros((0, len(cols)))
    return mat


def _assign_fronts(df: pd.DataFrame, objective: str) -> pd.DataFrame:
    """``front`` over all rows and ``front_in_domain`` over the in-domain rows; rows without a
    finite objective vector (failed, invalid) are put after every ranked row."""
    df = df.copy()
    if len(df) == 0:
        df["front"] = pd.Series(dtype=int)
        df["front_in_domain"] = pd.Series(dtype=int)
        return df
    ranks = pareto_rank(_objective_matrix(df, objective))
    worst = int(ranks.max()) + 1 if ranks.size else 1
    ranks = np.where(ranks == 0, worst, ranks)
    df["front"] = ranks.astype(int)
    ind = df["in_domain"].to_numpy(dtype=bool)
    fid = np.zeros(len(df), dtype=int)
    if ind.any():
        sub = pareto_rank(_objective_matrix(df.loc[ind], objective))
        w = int(sub.max()) + 1 if sub.size else 1
        fid[ind] = np.where(sub == 0, w, sub)
    df["front_in_domain"] = fid
    return df


def lhs_pareto(space: DesignSpace, base_spec: CascadeSpec, system: SystemModel, target: str,
               impurities: Sequence[str], spec_limits: Mapping[str, float] | None,
               prices: Prices | None, *, n: int = 2000, seed: int = 18,
               objective: str = "consumption", solver_kwargs: Mapping[str, Any] | None = None,
               progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """LHS regime search with non-dominated ranking (DESIGN.md section 9.2).

    ``qmc.LatinHypercube(d, scramble=True, seed=seed).random(n)`` decoded through ``space``; one
    cascade per row; every row keeps its variables, ``status``, metrics, flags and
    ``regime_status``; ``front`` (all rows) and ``front_in_domain`` (rows with ``regime_status``
    in ``IN_DOMAIN_STATUSES``); rows sorted by (front, variable tuple) so the CSV is byte-identical
    across runs.  ``df.attrs`` carries ``n``, ``seed``, ``n_failed``, ``n_invalid_spec``,
    ``n_converged``, ``objective``, ``variables`` and the space.  ``objective`` in
    ``OBJECTIVES``; ``progress(i, n)`` is called after every evaluation when given.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {tuple(OBJECTIVES)}, got {objective!r}")
    if n < 1:
        raise ValueError("n must be >= 1")
    names = space.variables
    d = len(names)
    if d == 0:
        U = np.zeros((n, 0))
    else:
        U = qmc.LatinHypercube(d, scramble=True, seed=seed).random(n)
    rows = []
    for i in range(n):
        values = space.decode(U[i])
        row = evaluate_candidate(values, base_spec, system, target, impurities, spec_limits,
                                 prices, solver_kwargs=solver_kwargs)
        row["candidate"] = i
        rows.append(row)
        if progress is not None:
            progress(i + 1, n)
    df = pd.DataFrame(rows)
    df = _assign_fronts(df, objective)
    var_cols = [v for v in VARIABLE_NAMES if v in df.columns]
    df = df.sort_values(["front", *var_cols, "candidate"], kind="mergesort").reset_index(drop=True)
    ordered = ["front", "front_in_domain", "status", "regime_status", "in_domain", "on_spec",
               *var_cols]
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]
    status = df["status"].astype(str)
    df.attrs.update({
        "n": int(n), "seed": int(seed), "objective": objective, "variables": list(names),
        "n_failed": int((status == "failed").sum()),
        "n_invalid_spec": int((status == "invalid_spec").sum()),
        "n_converged": int(status.str.startswith("converged").sum()),
        "space": space.to_json(),
    })
    return df


def pareto_fronts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The first non-dominated front twice: ``in_domain_only`` (rows with ``front_in_domain ==
    1``) and ``all`` (rows with ``front == 1``), each sorted as the input."""
    out = {"in_domain_only": df.loc[df["front_in_domain"] == 1].reset_index(drop=True),
           "all": df.loc[df["front"] == 1].reset_index(drop=True)}
    return out


def epsilon_knees(df: pd.DataFrame, purity_grid: Sequence[float], recovery_grid: Sequence[float],
                  *, objective_col: str = "consumption_index",
                  purity_col: str = "purity_mol", recovery_col: str = "recovery_from_feed",
                  ) -> pd.DataFrame:
    """Epsilon-constraint knees (section 9.2): for every (purity_min, recovery_min) cell and each
    subset (``all``, ``in_domain_only``) the converged row of minimum ``objective_col`` among the
    rows meeting both thresholds.  Columns: ``subset, purity_min, recovery_min, reachable,
    n_feasible`` followed by the knee row's columns (NaN when unreachable)."""
    rows = []
    converged = df["status"].astype(str).str.startswith("converged") if len(df) else pd.Series(
        dtype=bool)
    for subset in ("all", "in_domain_only"):
        base = df.loc[converged] if len(df) else df
        if subset == "in_domain_only" and len(base):
            base = base.loc[base["in_domain"].astype(bool)]
        for pmin in purity_grid:
            for rmin in recovery_grid:
                rec = {"subset": subset, "purity_min": float(pmin), "recovery_min": float(rmin)}
                if len(base):
                    ok = (pd.to_numeric(base[purity_col], errors="coerce") >= pmin) & (
                        pd.to_numeric(base[recovery_col], errors="coerce") >= rmin)
                    feas = base.loc[ok]
                    obj = pd.to_numeric(feas[objective_col], errors="coerce") if len(feas) \
                        else pd.Series(dtype=float)
                    feas = feas.loc[obj.notna()] if len(feas) else feas
                else:
                    feas = base
                rec["n_feasible"] = int(len(feas))
                rec["reachable"] = bool(len(feas) > 0)
                if len(feas):
                    obj = pd.to_numeric(feas[objective_col], errors="coerce")
                    best = feas.loc[obj.idxmin()]
                    rec.update({k: best[k] for k in df.columns})
                rows.append(rec)
    cols = ["subset", "purity_min", "recovery_min", "reachable", "n_feasible", *list(df.columns)]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


# =============================================================================================
# 9.3  GP-BO (ParEGO), gated on the reliability flags
# =============================================================================================

@dataclass(frozen=True)
class BOResult:
    """``status`` in ``{"done", "gated", "no_candidates"}``; ``reason``; ``table`` the evaluated
    rows in the ``lhs_pareto`` layout plus ``source`` (``init`` / ``bo``) and ``bo_iteration``."""

    status: str
    reason: str
    table: pd.DataFrame


def reliability_passes(system: SystemModel, reliability: Mapping[str, Any] | None = None,
                       dmodels_dir: str | Path | None = None) -> tuple[bool, str]:
    """Whether the D source of ``system`` may drive the GP-BO (PRE_REGISTRATION.md section 6).

    ``reliability`` is ``{"interpretable": {"n": bool, "p_eff": bool}}`` (the shape of
    ``results/dmodels/<system_id>.json``); when None it is read from ``dmodels_dir /
    <system_id>.json`` (``paths.RESULTS_DMODELS_DIR`` by default).  A system without a fitted,
    interpretable slope pair (literature placeholders, the nearest-condition fallback) does not
    pass: there is no reliability flag to gate on."""
    if reliability is None:
        sid = system.system_id
        if sid is None:
            return False, "the system model has no entry / system_id to look up"
        if dmodels_dir is None:
            from gen18proc import paths

            dmodels_dir = paths.RESULTS_DMODELS_DIR
        path = Path(dmodels_dir) / f"{sid}.json"
        if not path.is_file():
            return False, f"no fit reliability file {path.name} (D source not corpus-fitted)"
        with path.open("r", encoding="utf-8") as fh:
            reliability = json.load(fh)
    interp = (reliability or {}).get("interpretable") or {}
    ok = bool(interp.get("n", False)) and bool(interp.get("p_eff", False))
    if not ok:
        return False, f"slopes not interpretable: {dict(interp)}"
    if "fitted" not in str(getattr(system, "params_source", "")):
        return False, (f"the system model's D source is {system.params_source!r}, not the "
                       "fitted set the reliability refers to")
    return True, "both slopes interpretable and the fitted set is in use"


def _scalarise(F01: np.ndarray, w: np.ndarray, rho: float = 0.05) -> np.ndarray:
    """ParEGO augmented Chebyshev scalarisation of normalised objectives (minimise)."""
    wf = F01 * w[None, :]
    return wf.max(axis=1) + rho * wf.sum(axis=1)


def _expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    from scipy.stats import norm

    sd = np.maximum(sd, 1e-12)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def gp_bo(space: DesignSpace, base_spec: CascadeSpec, system: SystemModel, target: str,
          impurities: Sequence[str], spec_limits: Mapping[str, float] | None,
          prices: Prices | None, *, n_init: int = 64, n_iter: int = 30, batch: int = 4,
          seed: int = 18, objective: str = "consumption",
          reliability: Mapping[str, Any] | None = None, force: bool = False,
          n_candidates: int = 2000, solver_kwargs: Mapping[str, Any] | None = None,
          progress: Callable[[int, int], None] | None = None) -> BOResult:
    """ParEGO GP-BO over ``space`` (DESIGN.md section 9.3), gated on the reliability flags.

    ``n_init`` LHS evaluations, then ``n_iter`` iterations of ``batch`` points each: a random
    weight vector on the simplex (seeded), the augmented Chebyshev scalarisation of the
    min-max-normalised objectives of the converged rows, a ``GaussianProcessRegressor`` with
    ``Matern(nu=2.5)`` ARD + ``WhiteKernel`` on the unit hypercube, expected improvement over
    ``n_candidates`` LHS candidates, kriging believer for the batch.  ``reliability`` /
    ``force`` control the gate (``reliability_passes``); a gated call evaluates nothing and
    returns ``status = "gated"``.  The returned table has the ``lhs_pareto`` columns plus
    ``source`` and ``bo_iteration``; fronts are recomputed over every evaluated row.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    if not force:
        ok, reason = reliability_passes(system, reliability)
        if not ok:
            return BOResult("gated", reason, pd.DataFrame())
    else:
        reason = "gate overridden by the caller (force=True); not a headline number"
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {tuple(OBJECTIVES)}, got {objective!r}")
    rng = np.random.default_rng(seed)
    names = space.variables
    d = len(names)
    if d == 0:
        return BOResult("no_candidates", "the design space has no free variable", pd.DataFrame())
    U = qmc.LatinHypercube(d, scramble=True, seed=seed).random(n_init)
    rows: list[dict[str, Any]] = []
    coords: list[np.ndarray] = []
    total = n_init + n_iter * batch
    done = 0

    def run(u: np.ndarray, source: str, it: int) -> None:
        nonlocal done
        values = space.decode(u)
        row = evaluate_candidate(values, base_spec, system, target, impurities, spec_limits,
                                 prices, solver_kwargs=solver_kwargs)
        row["candidate"] = len(rows)
        row["source"] = source
        row["bo_iteration"] = it
        rows.append(row)
        coords.append(np.asarray(u, dtype=float))
        done += 1
        if progress is not None:
            progress(done, total)

    for i in range(n_init):
        run(U[i], "init", 0)
    obj_cols = OBJECTIVES[objective]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d),
                                                       length_scale_bounds=(1e-2, 1e2), nu=2.5) \
        + WhiteKernel(1e-4, (1e-8, 1e-1))
    for it in range(1, n_iter + 1):
        df = pd.DataFrame(rows)
        F = _objective_matrix(df, objective)
        finite = np.all(np.isfinite(F), axis=1)
        if finite.sum() < 3:
            # nothing to model yet: keep exploring by LHS
            Ui = qmc.LatinHypercube(d, scramble=True,
                                    seed=int(rng.integers(2**31 - 1))).random(batch)
            for b in range(batch):
                run(Ui[b], "bo", it)
            continue
        X = np.vstack(coords)[finite]
        lo, hi = F[finite].min(axis=0), F[finite].max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        F01 = (F[finite] - lo) / span
        w = rng.dirichlet(np.ones(len(obj_cols)))
        y = _scalarise(F01, w)
        y_mean, y_sd = float(y.mean()), float(y.std() or 1.0)
        yn = (y - y_mean) / y_sd
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=0,
                                      random_state=int(rng.integers(2**31 - 1)))
        gp.fit(X, yn)
        cand = qmc.LatinHypercube(d, scramble=True, seed=int(rng.integers(2**31 - 1))).random(
            n_candidates)
        Xb, yb = X.copy(), yn.copy()
        chosen: list[np.ndarray] = []
        fitted_kernel = gp.kernel_
        for b in range(batch):
            mu, sd = gp.predict(cand, return_std=True)
            ei = _expected_improvement(mu, sd, float(yb.min()))
            j = int(np.argmax(ei))
            chosen.append(cand[j].copy())
            # kriging believer: the predicted mean stands in for the unknown value
            Xb = np.vstack([Xb, cand[j][None, :]])
            yb = np.append(yb, mu[j])
            cand = np.delete(cand, j, axis=0)
            gp = GaussianProcessRegressor(kernel=fitted_kernel, optimizer=None,
                                          normalize_y=False)
            gp.fit(Xb, yb)
        for u in chosen:
            run(u, "bo", it)
    df = pd.DataFrame(rows)
    df = _assign_fronts(df, objective)
    var_cols = [v for v in VARIABLE_NAMES if v in df.columns]
    df = df.sort_values(["front", *var_cols, "candidate"], kind="mergesort").reset_index(drop=True)
    ordered = ["front", "front_in_domain", "status", "regime_status", "in_domain", "on_spec",
               "source", "bo_iteration", *var_cols]
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]
    status = df["status"].astype(str)
    df.attrs.update({"n_init": int(n_init), "n_iter": int(n_iter), "batch": int(batch),
                     "seed": int(seed), "objective": objective, "variables": list(names),
                     "n_failed": int((status == "failed").sum()),
                     "n_invalid_spec": int((status == "invalid_spec").sum()),
                     "space": space.to_json(), "gate": reason})
    return BOResult("done", reason, df)
