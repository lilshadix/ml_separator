"""``metrics.py`` — purity, recovery, enrichment, selectivity, throughput, reagent consumption
and the cost proxy of a solved cascade (DESIGN.md section 8).

``compute_metrics(result, spec, system, target, impurities, prices)`` returns a
``ProcessMetrics``; every numeric field is None / NaN when ``result.status == "failed"``.

* Purities are fractions of the target T among the metals of the product (the strip liquor, or
  the loaded organic when ``n_str = 0``) on the mol basis, the mass basis (IUPAC conventional
  atomic weights, g/mol) and the sesquioxide basis (``M2O3``: ``(2 M + 48.0) / 2`` g per mol of
  metal, i.e. ``M + 24.0``).
* ``recovery_from_feed`` (headline), ``recovery_total``, ``scrub_target_return`` and
  ``net_product_mol_h`` are the origin-labelled quantities of section 7.8 (``result.origin``
  when the cascade was solved with this target, else recomputed by the same linear pass at the
  converged D).
* ``enrichment_factor[I] = (n_T / n_I)_product / (n_T / n_I)_feed`` (named so; not a separation
  factor); ``sf_tracer[section][I] = D_T / D_I`` at each section's tracer conditions;
  ``sf_by_stage[j][I]`` from the converged stage D.
* Consumption (mol/h or L/h, then per mol of T in the product and per kg of T oxide):
  ``acid = S h_S + W h_W + max(0, feed acidification)``; ``base = O * saponification_degree *
  [HA]_T`` (the re-created alkali reserve, monomer basis of the acid-releasing ligands);
  ``salting_anion = S max(0, anion_S - h_S) + W max(0, anion_W - h_W)`` (anion beyond the acid);
  ``complexant = (A cT_F + S cT_S + W cT_W) (1 - regeneration_fraction)`` (NaN +
  ``COST_INCOMPLETE`` when the fraction is unknown and complexant is used);
  ``extractant_makeup = f_bleed O [L]_T,formal + ligand_loss (A + S + W)`` (NaN +
  ``LIGAND_LOSS_NOT_MEASURED`` when the loss is unmeasured and ``f_bleed = 0``; the flag alone
  when the loss is unmeasured but the bleed term exists); ``diluent = f_bleed O`` L/h (NaN when
  there is no bleed: entrainment losses are not measured); ``water = S + W + dilution``.
* ``cost_proxy_per_kg_oxide = sum_i price_i * consumption_i`` from ``config/prices.json`` (every
  entry a labelled placeholder with a range); NaN + ``COST_INCOMPLETE`` when any price or any
  consumption is NaN.  Consumption is the primary economic number.
* ``on_spec = purity_mol >= purity_min and recovery_from_feed >= recovery_min`` from
  ``spec_limits`` (None when no limits are given); ``regime_status`` recomputed over the
  cascade flags plus the metric flags.

Units: mol/L, L/h, mol/h, kg/h, g/mol; ``log`` means log10.  Sources: DESIGN.md section 8,
addendum WB3 (keyword arguments, key names, the ``strip`` origin label).
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .cascade import (
    _Problem,
    alkali_reserve_inlet,
    origin_pass,
    regime_status_of,
    section_tracer_d,
)
from .dmodel import SystemModel
from .types import (
    AqStream,
    CascadeResult,
    CascadeSpec,
    Flag,
    OrgStream,
    PhaseBehaviour,
    ProcessMetrics,
    Sourced,
)

__all__ = [
    "ATOMIC_MASS_G_MOL", "OXIDE_OXYGEN_PER_METAL_G_MOL", "CONSUMPTION_ITEMS", "ITEM_UNITS",
    "Prices", "compute_metrics", "product_moles", "oxide_mass_per_mol_metal",
    "regime_status_of", "effective_f_bleed",
]

#: IUPAC conventional atomic weights (g/mol) of the lanthanides (same table as WB1's ingest).
ATOMIC_MASS_G_MOL: dict[str, float] = {
    "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.36, "Eu": 151.964,
    "Gd": 157.25, "Tb": 158.925, "Dy": 162.500, "Ho": 164.930, "Er": 167.259, "Tm": 168.934,
    "Yb": 173.045, "Lu": 174.967,
}

OXIDE_OXYGEN_PER_METAL_G_MOL = 24.0
"""Oxygen mass per mol of metal in a sesquioxide M2O3: ``48.0 / 2`` (DESIGN.md section 8)."""

CONSUMPTION_ITEMS: tuple[str, ...] = ("acid", "base", "salting_anion", "complexant",
                                      "extractant_makeup", "diluent", "water")
ITEM_UNITS: dict[str, str] = {"acid": "mol", "base": "mol", "salting_anion": "mol",
                              "complexant": "mol", "extractant_makeup": "mol", "diluent": "L",
                              "water": "L"}

_DEFAULT_PRICES = Path(__file__).resolve().parents[1] / "config" / "prices.json"


def oxide_mass_per_mol_metal(metal: str) -> float:
    """Grams of ``M2O3`` per mol of metal atoms: ``(2 M + 48.0) / 2`` (NaN for an unknown
    metal)."""
    mass = ATOMIC_MASS_G_MOL.get(metal)
    return math.nan if mass is None else mass + OXIDE_OXYGEN_PER_METAL_G_MOL


def effective_f_bleed(spec: CascadeSpec) -> float:
    """``f_bleed`` as the cascade uses it: forced to 1 when there is no strip section (7.2)."""
    return 1.0 if int(spec.n_str) == 0 else float(spec.f_bleed)


# ---------------------------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Prices:
    """The price table of ``config/prices.json``: ``items[name] = {value, unit, currency,
    status, assumed_label, range, note}``; every value is a labelled placeholder until a person
    replaces it (open item U5).  ``price(name)`` returns the value (None when absent or null);
    ``unit(name)`` its unit string (``"<currency>/mol"`` or ``"<currency>/L"``)."""

    items: Mapping[str, Mapping[str, Any]]
    currency: str = "USD"
    path: str | None = None
    schema: str = "gen18.prices.1"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Prices":
        p = Path(path) if path is not None else _DEFAULT_PRICES
        with p.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
        return cls.from_json(obj, path=str(p))

    @classmethod
    def from_json(cls, obj: Mapping[str, Any], *, path: str | None = None) -> "Prices":
        items = {str(k): dict(v) for k, v in (obj.get("items") or {}).items()}
        return cls(items=items, currency=str(obj.get("currency", "USD")), path=path,
                   schema=str(obj.get("schema", "gen18.prices.1")))

    def price(self, name: str) -> float | None:
        item = self.items.get(name)
        if item is None or item.get("value") is None:
            return None
        return float(item["value"])

    def unit(self, name: str) -> str | None:
        item = self.items.get(name)
        return None if item is None else item.get("unit")

    def without(self, *names: str) -> "Prices":
        """A copy lacking ``names`` (used by the tests for the ``COST_INCOMPLETE`` path)."""
        return replace(self, items={k: v for k, v in self.items.items() if k not in names})

    def table(self) -> list[dict[str, Any]]:
        """One row per item with value, unit, currency, status, range and note."""
        rows = []
        for name in CONSUMPTION_ITEMS:
            item = self.items.get(name, {})
            rows.append({"item": name, "value": item.get("value"), "unit": item.get("unit"),
                         "currency": item.get("currency", self.currency),
                         "status": item.get("status"), "assumed_label": item.get("assumed_label"),
                         "range": item.get("range"), "note": item.get("note", "")})
        return rows


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def product_moles(product: AqStream | OrgStream) -> dict[str, float]:
    """mol/h of every metal in the product stream (aqueous strip liquor or loaded organic)."""
    return {m: float(product.flow_L_h * v) for m, v in product.metals.items()}


def _feed_moles(spec: CascadeSpec) -> dict[str, float]:
    return {m: float(spec.feed.flow_L_h * v) for m, v in spec.feed.metals.items()}


def _phase_items(system: SystemModel) -> dict[str, Any]:
    phase: PhaseBehaviour | None = getattr(system, "phase", None)
    if phase is None and getattr(system, "entry", None) is not None:
        phase = system.entry.phase
    if phase is None:
        return {"loc_metal_M": Sourced.unknown("mol/L", note="no phase data in the system model"),
                "disengagement_s": Sourced.unknown("s"),
                "ligand_loss": Sourced.unknown("mol/L_aq"),
                "third_phase_observed": Sourced.unknown("1"), "regenerability_note": ""}
    return {"loc_metal_M": phase.loc_metal_M, "disengagement_s": phase.disengagement_s,
            "ligand_loss": phase.ligand_loss_mol_per_L_aq,
            "third_phase_observed": phase.third_phase_observed,
            "regenerability_note": phase.regenerability_note}


def _origin_for(result: CascadeResult, spec: CascadeSpec, system: SystemModel, target: str,
                ) -> Mapping[str, float] | None:
    if result.origin is not None and spec.target == target:
        return result.origin
    if result.status == "failed":
        return None
    spec_t = replace(spec, target=target)
    prob = _Problem(spec_t, system)
    if target not in prob.metals:
        return None
    D_T = np.array([d.d[target] for d in result.diagnostics])
    if not np.all(np.isfinite(D_T)):
        return None
    x_T = np.array([a.metals.get(target, 0.0) for a in result.stages_aq])
    y_T = np.array([o.metals.get(target, 0.0) for o in result.stages_org])
    return origin_pass(prob, D_T, x_T, y_T)[0]


def _nan_metrics(result: CascadeResult, spec: CascadeSpec, system: SystemModel, target: str,
                 impurities: tuple[str, ...], flags: set[Flag]) -> ProcessMetrics:
    top_flows = _flows(spec)
    A, S, W, O = top_flows
    nan = math.nan
    consumption = {}
    for item in CONSUMPTION_ITEMS:
        unit = ITEM_UNITS[item]
        consumption[f"{item}_{unit}_h"] = nan
        consumption[f"{item}_{unit}_per_mol_T"] = nan
        consumption[f"{item}_{unit}_per_kg_oxide"] = nan
    n = int(spec.n_ext) + int(spec.n_scr) + int(spec.n_str)
    flags = set(flags) | {Flag.NOT_CONVERGED}
    return ProcessMetrics(
        purity_mol=None, purity_mass=None, purity_oxide=None, recovery_from_feed=None,
        recovery_total=None, scrub_target_return=None, net_product_mol_h=None,
        enrichment_factor={i: nan for i in impurities}, sf_tracer={}, sf_by_stage=(),
        n_stages_total=n, n_ext=int(spec.n_ext), n_scr=int(spec.n_scr), n_str=int(spec.n_str),
        oa_ext=O / A, s_over_a=S / A, w_over_a=W / A, throughput_mol_T_per_h_per_L_org=None,
        throughput_kg_oxide_per_h=None,
        max_loading_fraction={k: nan for k in spec.ligand_total}, phase=_phase_items(system),
        consumption=consumption, cost_proxy_per_kg_oxide=None, on_spec=None,
        regime_status=regime_status_of(flags), flags=frozenset(flags),
        ood_distance=dict(result.ood_distance))


def _flows(spec: CascadeSpec) -> tuple[float, float, float, float]:
    A = float(spec.feed.flow_L_h)
    S = float(spec.scrub.flow_L_h) if int(spec.n_scr) > 0 else 0.0
    W = float(spec.strip.flow_L_h) if int(spec.n_str) > 0 else 0.0
    return A, S, W, float(spec.organic_flow_L_h)


# ---------------------------------------------------------------------------------------------
# section 8
# ---------------------------------------------------------------------------------------------

def compute_metrics(result: CascadeResult, spec: CascadeSpec, system: SystemModel, target: str,
                    impurities: Iterable[str], prices: Prices | None = None, *,
                    spec_limits: Mapping[str, float] | None = None,
                    feed_acidification_mol_h: float = 0.0,
                    feed_dilution_L_h: float = 0.0) -> ProcessMetrics:
    """Process metrics of a solved cascade (module docstring; DESIGN.md section 8).

    ``spec_limits`` (``{"purity_min": ..., "recovery_min": ...}``) sets ``on_spec``;
    ``feed_acidification_mol_h`` (acid added to bring the raw feed to ``spec.feed.h``) and
    ``feed_dilution_L_h`` (water added to the raw feed) enter the acid and water consumption.
    Never raises on a failed result (NaN metrics); ``ValueError`` when ``target`` is not a metal
    of the feed or the system.
    """
    impurities = tuple(impurities)
    if target not in system.metals:
        raise ValueError(f"target {target!r} is not parameterised by the system")
    flags: set[Flag] = set(result.flags)
    if result.status == "failed":
        return _nan_metrics(result, spec, system, target, impurities, flags)
    A, S, W, O = _flows(spec)
    n_prod = product_moles(result.product)
    n_feed = _feed_moles(spec)
    n_T = n_prod.get(target, 0.0)
    total_mol = sum(n_prod.values())
    # purities
    if total_mol > 0.0:
        purity_mol = n_T / total_mol
        mass = {m: n * ATOMIC_MASS_G_MOL.get(m, math.nan) for m, n in n_prod.items()}
        oxide = {m: n * oxide_mass_per_mol_metal(m) for m, n in n_prod.items()}
        purity_mass = mass.get(target, 0.0) / sum(mass.values())
        purity_oxide = oxide.get(target, 0.0) / sum(oxide.values())
    else:
        purity_mol = purity_mass = purity_oxide = math.nan
    # origin-labelled recovery (7.8)
    origin = _origin_for(result, spec, system, target)
    if origin is not None:
        recovery_from_feed = float(origin["recovery_from_feed"])
        recovery_total = float(origin["recovery_total"])
        str_ = origin["scrub_target_return"]
        scrub_target_return = (None if (isinstance(str_, float) and math.isnan(str_))
                               else float(str_))
        net_product = float(origin["net_product_mol_h"])
    else:
        feed_T = n_feed.get(target, 0.0)
        recovery_total = n_T / feed_T if feed_T > 0 else math.nan
        recovery_from_feed = math.nan
        scrub_target_return = None
        net_product = n_T - (S * float(spec.scrub.metals.get(target, 0.0)))
    # enrichment per impurity
    enrichment: dict[str, float] = {}
    for imp in impurities:
        n_I = n_prod.get(imp, 0.0)
        f_T, f_I = n_feed.get(target, 0.0), n_feed.get(imp, 0.0)
        if f_T <= 0.0 or f_I <= 0.0 or n_T <= 0.0:
            enrichment[imp] = math.nan
        elif n_I <= 0.0:
            enrichment[imp] = math.inf
        else:
            enrichment[imp] = (n_T / n_I) / (f_T / f_I)
    # selectivity at the tracer conditions and per stage
    tracer = section_tracer_d(spec, system)
    sf_tracer: dict[str, dict[str, float]] = {}
    for section, dvals in tracer.items():
        d_T = dvals.get(target, math.nan)
        sf_tracer[section] = {imp: (d_T / dvals[imp] if dvals.get(imp, 0.0) > 0 else math.inf)
                              for imp in impurities}
    sf_by_stage = []
    for diag in result.diagnostics:
        d_T = diag.d.get(target, math.nan)
        sf_by_stage.append({imp: (d_T / diag.d[imp] if diag.d.get(imp, 0.0) > 0 else math.inf)
                            for imp in impurities})
    # throughput
    kg_oxide_h = n_T * oxide_mass_per_mol_metal(target) / 1000.0
    throughput_mol = n_T / O
    # loading
    max_loading: dict[str, float] = {}
    for diag in result.diagnostics:
        for k, v in diag.loading_fraction.items():
            max_loading[k] = max(max_loading.get(k, -math.inf), v)
    # consumption (mol/h, L/h)
    cons_h: dict[str, float] = {}
    cons_h["acid"] = S * spec.scrub.h + W * spec.strip.h + max(0.0, float(feed_acidification_mol_h))
    cons_h["base"] = O * alkali_reserve_inlet(spec, system)
    cons_h["salting_anion"] = (S * max(0.0, spec.scrub.anion - spec.scrub.h)
                               + W * max(0.0, spec.strip.anion - spec.strip.h))
    c_total = (A * spec.feed.complexant_total + S * spec.scrub.complexant_total
               + W * spec.strip.complexant_total)
    if c_total > 0.0:
        cm = system.complexant_model
        rf = cm.regeneration_fraction if cm is not None else None
        if rf is None:
            cons_h["complexant"] = math.nan
            flags.add(Flag.COST_INCOMPLETE)
        else:
            cons_h["complexant"] = c_total * (1.0 - float(rf))
    else:
        cons_h["complexant"] = 0.0
    f_bleed = effective_f_bleed(spec)
    formal_lt = 0.0
    for name, lt in spec.ligand_total.items():
        md = system.dmodels.get(name)
        formal_lt += float(getattr(md, "ligand_scale", 1.0) or 1.0) * float(lt)
    phase = _phase_items(system)
    loss_obj = phase["ligand_loss"]
    loss = loss_obj.value if isinstance(loss_obj, Sourced) else None
    bleed_term = f_bleed * O * formal_lt
    if loss is None:
        flags.add(Flag.LIGAND_LOSS_NOT_MEASURED)
        cons_h["extractant_makeup"] = math.nan if f_bleed == 0.0 else bleed_term
    else:
        cons_h["extractant_makeup"] = bleed_term + float(loss) * (A + S + W)
    cons_h["diluent"] = f_bleed * O if f_bleed > 0.0 else math.nan
    cons_h["water"] = S + W + max(0.0, float(feed_dilution_L_h))
    consumption: dict[str, float] = {}
    for item in CONSUMPTION_ITEMS:
        unit = ITEM_UNITS[item]
        val = cons_h[item]
        consumption[f"{item}_{unit}_h"] = val
        consumption[f"{item}_{unit}_per_mol_T"] = val / n_T if n_T > 0 else math.nan
        consumption[f"{item}_{unit}_per_kg_oxide"] = (val / kg_oxide_h if kg_oxide_h > 0
                                                      else math.nan)
    # cost proxy
    cost = 0.0
    complete = prices is not None
    if prices is not None:
        for item in CONSUMPTION_ITEMS:
            price = prices.price(item)
            per_kg = consumption[f"{item}_{ITEM_UNITS[item]}_per_kg_oxide"]
            unit = prices.unit(item) or ""
            if price is None or not math.isfinite(per_kg) or not unit.endswith(
                    "/" + ITEM_UNITS[item]):
                complete = False
                break
            cost += price * per_kg
    if not complete:
        cost = math.nan
        flags.add(Flag.COST_INCOMPLETE)
    # on_spec
    on_spec: bool | None = None
    if spec_limits is not None:
        pmin = float(spec_limits.get("purity_min", 0.0))
        rmin = float(spec_limits.get("recovery_min", 0.0))
        on_spec = bool(math.isfinite(purity_mol) and math.isfinite(recovery_from_feed)
                       and purity_mol >= pmin and recovery_from_feed >= rmin)
    n = int(spec.n_ext) + int(spec.n_scr) + int(spec.n_str)
    return ProcessMetrics(
        purity_mol=purity_mol, purity_mass=purity_mass, purity_oxide=purity_oxide,
        recovery_from_feed=recovery_from_feed, recovery_total=recovery_total,
        scrub_target_return=scrub_target_return, net_product_mol_h=net_product,
        enrichment_factor=enrichment, sf_tracer=sf_tracer, sf_by_stage=tuple(sf_by_stage),
        n_stages_total=n, n_ext=int(spec.n_ext), n_scr=int(spec.n_scr), n_str=int(spec.n_str),
        oa_ext=O / A, s_over_a=S / A, w_over_a=W / A,
        throughput_mol_T_per_h_per_L_org=throughput_mol, throughput_kg_oxide_per_h=kg_oxide_h,
        max_loading_fraction=max_loading, phase=phase, consumption=consumption,
        cost_proxy_per_kg_oxide=cost, on_spec=on_spec, regime_status=regime_status_of(flags),
        flags=frozenset(flags), ood_distance=dict(result.ood_distance))
