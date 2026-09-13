"""Tests of ``gen18proc.domain`` (DESIGN.md section 5.5; test row of 12.2: inside the box and
hull -> no OOD flag; outside on one axis -> that flag with the stated distance, abs 1e-12).

Also: ``build_domain`` on tiny synthetic frames in the ``corpus_records.csv`` layout and in the
flattened / mapping layouts, the degenerate hulls (segment, point), ``coerce_domain`` on the
JSON block of section 3.7, and the dimer ``ligand_scale``.  Every number is synthetic.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import testsystems as ts  # noqa: E402
from gen18proc.dmodel import tracer_state  # noqa: E402
from gen18proc.domain import (  # noqa: E402
    AXIS_FLAGS,
    build_domain,
    coerce_domain,
    domain_flags,
    hull_distance,
    hull_of_points,
    interval_distance,
)
from gen18proc.types import ApplicabilityDomain, Flag, StageState  # noqa: E402

ABS = 1e-12
LIG = ts.SOLV_LIGAND


def box_domain(**overrides) -> ApplicabilityDomain:
    """A literature-style box (no hull), section 3.7 shape."""
    base = dict(anion="chloride", diluent_family="aliphatic_hydrocarbon", modifiers=("octanol",),
                log_acid=(-1.42, -1.02), log_ligand={LIG: (-0.699, 0.0)},
                log_metal_total_mM=(1.0, 1.7), loading_fraction=(0.0, 0.2),
                log_complexant=(-2.0, -1.0), oa_ratio=(1.0, 1.0), temperature_C=(25.0, 25.0),
                saponification_degree=None, hull_vertices=None, n_records=0, n_publications=0)
    base.update(overrides)
    return ApplicabilityDomain(**base)


def state(*, h: float = 10 ** -1.2, lt: float = 0.5, lf: float | None = None, x: float = 0.03,
          y: float = 0.0, c: float = 0.03, r: float = 1.0, temperature: float = 25.0,
          ligand: str = LIG) -> StageState:
    """A stage state inside ``box_domain()`` by default (metal 30 mM, loading 0.1, c 0.03)."""
    lf = 0.9 * lt if lf is None else lf
    return StageState(v_aq_L=1.0, v_org_L=r, x_total={"Nd": x}, y={"Nd": y},
                      y_by_ligand={ligand: {"Nd": y}}, h=h, anion=0.31, c_free=c,
                      ligand_total={ligand: lt}, ligand_free={ligand: lf},
                      acid_in_org={ligand: 0.0}, alkali_reserve=0.0, temperature_C=temperature)


def categorical() -> dict:
    return dict(anion="chloride", diluent_family="aliphatic_hydrocarbon", modifiers=("octanol",))


# ---------------------------------------------------------------------------------------------
# 12.2 row: inside -> nothing; outside on one axis -> that flag with the stated distance
# ---------------------------------------------------------------------------------------------

def test_inside_box_no_flags() -> None:
    flags, dist = domain_flags(box_domain(), state(), LIG, **categorical())
    assert flags == frozenset()
    assert set(dist) == {"acid", "ligand", "metal", "loading", "complexant", "oa", "temperature",
                         "anion", "diluent", "modifier"}
    assert all(v == 0.0 for v in dist.values())
    # on the boundary counts as inside
    flags, dist = domain_flags(box_domain(), state(h=10 ** -1.02, lt=1.0), LIG)
    assert flags == frozenset() and dist["acid"] <= ABS and dist["ligand"] <= ABS
    # no domain -> nothing to say
    assert domain_flags(None, state(), LIG) == (frozenset(), {})


@pytest.mark.parametrize("axis, kwargs, expected", [
    ("acid", dict(h=10 ** -0.5), -0.5 - (-1.02)),
    ("acid", dict(h=10 ** -2.0), -1.42 - (-2.0)),
    ("ligand", dict(lt=10 ** 0.5), 0.5),
    ("ligand", dict(lt=10 ** -1.0), -0.699 - (-1.0)),
    ("metal", dict(x=0.5), math.log10(500.0) - 1.7),
    ("metal", dict(x=0.005), 1.0 - math.log10(5.0)),
    ("loading", dict(lf=0.6 * 0.5), 0.4 - 0.2),
    ("complexant", dict(c=1.0), 0.0 - (-1.0)),
    ("complexant", dict(c=1e-3), -2.0 - math.log10(1e-3)),
    ("oa", dict(r=2.0), math.log10(2.0)),
    ("oa", dict(r=0.25), math.log10(4.0)),
    ("temperature", dict(temperature=40.0), 15.0),
])
def test_outside_on_one_axis(axis: str, kwargs: dict, expected: float) -> None:
    flags, dist = domain_flags(box_domain(), state(**kwargs), LIG, **categorical())
    assert flags == frozenset({AXIS_FLAGS[axis]}), (axis, flags)
    assert abs(dist[axis] - expected) < ABS, (axis, dist[axis], expected)
    others = {k: v for k, v in dist.items() if k != axis}
    assert all(v == 0.0 for v in others.values()), others


@pytest.mark.parametrize("axis, kwargs", [
    ("anion", dict(anion="nitrate")),
    ("diluent", dict(diluent_family="aromatic_hydrocarbon")),
    ("modifier", dict(modifiers=())),
    ("modifier", dict(modifiers=("octanol", "tbp"))),
])
def test_categorical_mismatch_distance_one(axis: str, kwargs: dict) -> None:
    cat = categorical()
    cat.update(kwargs)
    flags, dist = domain_flags(box_domain(), state(), LIG, **cat)
    assert flags == frozenset({AXIS_FLAGS[axis]})
    assert dist[axis] == 1.0
    # categorical axes are only checked when given
    flags2, dist2 = domain_flags(box_domain(), state(), LIG)
    assert flags2 == frozenset() and axis not in dist2


def test_metal_axis_counts_both_phases_and_loading_kwarg() -> None:
    # total metal per litre aqueous is x + r y: 0.01 + 2 * 0.02 = 0.05 M = 50 mM -> log 1.699
    flags, dist = domain_flags(box_domain(), state(x=0.01, y=0.02, r=2.0, lf=0.5 * 0.9), LIG)
    assert "metal" in dist and abs(dist["metal"] - 0.0) < ABS
    # the loading axis takes the caller's lambda when given (EffectiveCapacity case)
    flags, dist = domain_flags(box_domain(), state(), LIG, loading=0.35)
    assert flags == frozenset({Flag.OOD_LOADING}) and abs(dist["loading"] - 0.15) < ABS
    # a complexant present against a domain without a complexant interval: distance 1.0
    flags, dist = domain_flags(box_domain(log_complexant=None), state(c=0.01), LIG)
    assert flags == frozenset({Flag.OOD_COMPLEXANT}) and dist["complexant"] == 1.0
    flags, dist = domain_flags(box_domain(log_complexant=None), state(c=0.0), LIG)
    assert "complexant" not in dist and flags == frozenset()
    # axes without an interval are absent from the distances
    flags, dist = domain_flags(box_domain(log_metal_total_mM=None, loading_fraction=None,
                                          oa_ratio=None, temperature_C=None), state(), LIG)
    assert set(dist) == {"acid", "ligand", "complexant"}


def test_dimer_ligand_scale() -> None:
    # the domain records the formal monomer concentration; the state carries the dimer total
    dom = box_domain(log_ligand={ts.CE_LIGAND: (math.log10(0.8), math.log10(0.8))})
    st = state(lt=0.4, ligand=ts.CE_LIGAND)
    flags, dist = domain_flags(dom, st, ts.CE_LIGAND, ligand_scale=2.0)
    assert Flag.OOD_LIGAND not in flags and dist["ligand"] <= ABS
    flags, dist = domain_flags(dom, st, ts.CE_LIGAND, ligand_scale=1.0)
    assert Flag.OOD_LIGAND in flags and abs(dist["ligand"] - math.log10(2.0)) < ABS
    # through the model: the cation-exchange fixture (box [-2, 0.5] on log [HA]_T) at 0.4 M dimer
    system = ts.two_metal_cation_exchange()
    model = system.dmodels[ts.CE_LIGAND]
    ev = model.evaluate("Nd", tracer_state(system, {ts.CE_LIGAND: 0.4}, 0.01, 0.31))
    assert Flag.OOD_LIGAND not in ev.flags and ev.ood_distance["ligand"] == 0.0
    ev = model.evaluate("Nd", tracer_state(system, {ts.CE_LIGAND: 4.0}, 0.01, 0.31))
    assert Flag.OOD_LIGAND in ev.flags
    assert abs(ev.ood_distance["ligand"] - (math.log10(8.0) - 0.5)) < ABS


# ---------------------------------------------------------------------------------------------
# build_domain and the hull
# ---------------------------------------------------------------------------------------------

def frame(points, *, ligand_col: str = "ligand_M", mm=None, pubs=None, eligible=None,
          temperature=25.0) -> pd.DataFrame:
    """Records in the corpus_records.csv layout at (log acid, log ligand) ``points``."""
    n = len(points)
    df = pd.DataFrame({
        "record_id": [f"r{i:02d}" for i in range(n)],
        "metal": ["Nd"] * n,
        "log_d": np.linspace(0.0, 1.0, n),
        "acid_nominal_M": [10.0 ** p[0] for p in points],
        ligand_col: [10.0 ** p[1] for p in points],
        "metal_initial_mM": mm if mm is not None else [math.nan] * n,
        "publication_id": pubs if pubs is not None else ["pubA"] * n,
        "fit_eligible": eligible if eligible is not None else [True] * n,
        "temperature_C": [temperature] * n,
        "ligand_name": [LIG] * n,
        "anion": ["nitrate"] * n,
        "diluent_family": ["aliphatic_hydrocarbon"] * n,
        "modifiers": [""] * n,
    })
    df["d"] = 10.0 ** df["log_d"]
    return df


def test_build_domain_box_and_hull() -> None:
    tri = [(-2.0, -1.0), (0.0, -1.0), (-2.0, 0.0), (-1.5, -0.8)]        # a triangle + interior
    df = frame(tri, mm=[0.5, 5.0, math.nan, 50.0], pubs=["pubA", "pubA", "pubB", "pubC"])
    dom = build_domain(df, LIG, n_prior=3.0)
    assert dom.log_acid == (-2.0, 0.0)
    assert dom.log_ligand == {LIG: (-1.0, 0.0)}
    assert dom.log_metal_total_mM == (math.log10(0.5), math.log10(50.0))
    # loading = n_prior * mM * 1e-3 / ligand_M on the rows with a metal concentration
    lam = [3.0 * 0.5e-3 / 0.1, 3.0 * 5e-3 / 0.1, 3.0 * 50e-3 / 10 ** -0.8]
    assert abs(dom.loading_fraction[0] - min(lam)) < ABS
    assert abs(dom.loading_fraction[1] - max(lam)) < ABS
    assert dom.temperature_C == (25.0, 25.0) and dom.oa_ratio is None
    assert dom.log_complexant is None and dom.saponification_degree is None
    assert dom.anion == "nitrate" and dom.diluent_family == "aliphatic_hydrocarbon"
    assert dom.modifiers == () and dom.n_records == 4 and dom.n_publications == 3
    assert dom.hull_vertices is not None and len(dom.hull_vertices) == 3
    assert set(dom.hull_vertices) == set(tri[:3])
    # inside the hull -> nothing (the default state loads the ligand to 0.1, inside the
    # recorded loading interval, and carries 30 mM metal, inside the metal interval)
    inside = state(h=10 ** -1.5, lt=10 ** -0.8, c=0.0)
    flags, dist = domain_flags(dom, inside, LIG)
    assert flags == frozenset() and dist["hull"] == 0.0
    assert dist["loading"] == 0.0 and dist["metal"] == 0.0
    # inside the box but outside the hull -> OOD_HULL only, Euclidean distance to the hypotenuse
    p = (-0.5, -0.2)
    outside = state(h=10 ** p[0], lt=10 ** p[1], c=0.0)
    flags, dist = domain_flags(dom, outside, LIG)
    assert flags == frozenset({Flag.OOD_HULL})
    assert abs(dist["hull"] - abs(p[0] + 2 * p[1] + 2) / math.sqrt(5.0)) < ABS
    assert dist["acid"] == 0.0 and dist["ligand"] == 0.0
    # outside the box on the acid axis -> OOD_ACID and OOD_HULL, distances as stated
    far = state(h=10 ** 0.5, lt=10 ** -1.0, c=0.0)
    flags, dist = domain_flags(dom, far, LIG)
    assert flags == frozenset({Flag.OOD_ACID, Flag.OOD_HULL})
    assert abs(dist["acid"] - 0.5) < ABS and abs(dist["hull"] - 0.5) < ABS
    # a vertex is inside (round-off tolerance)
    at_vertex = state(h=10 ** -2.0, lt=10 ** -1.0, c=0.0)
    assert domain_flags(dom, at_vertex, LIG)[0] == frozenset()
    # zero loading lies below the recorded loading interval (its minimum is 0.015)
    unloaded = state(h=10 ** -1.5, lt=10 ** -0.8, lf=10 ** -0.8, c=0.0)
    flags, dist = domain_flags(dom, unloaded, LIG)
    assert flags == frozenset({Flag.OOD_LOADING}) and abs(dist["loading"] - min(lam)) < ABS


def test_build_domain_filters_and_degenerate_hulls() -> None:
    pts = [(-1.0, -1.0), (0.0, -1.0), (0.5, -1.0), (1.0, -1.0)]        # collinear
    df = frame(pts, eligible=[True, True, False, True], temperature=25.0)
    dom = build_domain(df, LIG)
    assert dom.n_records == 3 and dom.log_acid == (-1.0, 1.0)
    assert dom.hull_vertices is not None and len(dom.hull_vertices) == 2
    assert set(dom.hull_vertices) == {(-1.0, -1.0), (1.0, -1.0)}
    st = state(h=10 ** 0.0, lt=10 ** -0.5, lf=10 ** -0.5, c=0.0)
    flags, dist = domain_flags(dom, st, LIG)
    assert flags == frozenset({Flag.OOD_LIGAND, Flag.OOD_HULL})
    assert abs(dist["hull"] - 0.5) < ABS and abs(dist["ligand"] - 0.5) < ABS
    # the ineligible point is not in the box
    assert 0.5 not in [v[0] for v in dom.hull_vertices]
    # one point
    one = build_domain(frame([(-1.0, -1.0)]), LIG)
    assert one.hull_vertices == ((-1.0, -1.0),)
    assert abs(domain_flags(one, state(h=10 ** -1.0, lt=10 ** -0.5, lf=10 ** -0.5, c=0.0),
                            LIG)[1]["hull"] - 0.5) < ABS
    # the band filter (NaN temperature counts as 20-30C)
    df2 = frame(pts, temperature=25.0)
    df2.loc[1, "temperature_C"] = 45.0
    df2.loc[2, "temperature_C"] = math.nan
    assert build_domain(df2, LIG, band="20-30C").n_records == 3
    assert build_domain(df2, LIG, band="40-50C").n_records == 1
    # no usable row -> ValueError from the ligand column, or NaN box when the column exists
    with pytest.raises(ValueError):
        build_domain(df.drop(columns=["ligand_M"]), "OTHER")


def test_build_domain_layouts_and_hull_helpers() -> None:
    pts = [(-2.0, -1.0), (0.0, -1.0), (-2.0, 0.0)]
    a = build_domain(frame(pts), LIG)
    b = build_domain(frame(pts, ligand_col=f"ligand_M.{LIG}"), LIG)
    c_df = frame(pts).drop(columns=["ligand_M"])
    c_df["ligand_M"] = [{LIG: 10.0 ** p[1]} for p in pts]
    c = build_domain(c_df, LIG)
    d_df = frame(pts).drop(columns=["ligand_M", "metal_initial_mM"])
    d_df["ligand_M"] = [f'{{"{LIG}": {10.0 ** p[1]}}}' for p in pts]
    d_df[f"metals_initial_mM.Nd"] = [1.0, 2.0, 3.0]
    d = build_domain(d_df, LIG)
    for dom in (b, c, d):
        assert dom.log_acid == a.log_acid and dom.log_ligand == a.log_ligand
        assert set(dom.hull_vertices) == set(a.hull_vertices)
    assert d.log_metal_total_mM == (0.0, math.log10(3.0))
    # explicit categorical keywords win over the columns
    e = build_domain(frame(pts), LIG, anion="chloride", diluent_family="x", modifiers=("m",))
    assert (e.anion, e.diluent_family, e.modifiers) == ("chloride", "x", ("m",))
    # hull helpers
    verts = hull_of_points(np.array([[0, 0], [1, 0], [0, 1], [0.2, 0.2], [1, 1]], dtype=float))
    assert set(verts) == {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
    assert hull_distance(verts, (0.5, 0.5)) == 0.0
    assert abs(hull_distance(verts, (2.0, 0.5)) - 1.0) < ABS
    assert abs(hull_distance(verts, (2.0, 2.0)) - math.sqrt(2.0)) < ABS
    assert math.isnan(hull_distance(None, (0.0, 0.0)))
    assert hull_of_points(np.zeros((0, 2))) is None
    assert interval_distance(0.5, (0.0, 1.0)) == 0.0
    assert interval_distance(1.5, (0.0, 1.0)) == 0.5 and interval_distance(-2.0, (0.0, 1.0)) == 2.0
    assert math.isnan(interval_distance(0.5, None))


def test_coerce_domain_from_json_block() -> None:
    block = {"anion": "chloride", "diluent_family": "aliphatic_hydrocarbon", "modifiers": [],
             "log_acid": [-1.42, -1.02], "log_ligand": {"PC88A": [-0.699, 0.0]},
             "log_metal_total_mM": [1.0, 1.7], "loading_fraction": [0.0, 0.2],
             "log_complexant": None, "oa_ratio": [1.0, 1.0], "temperature_C": [25.0, 25.0],
             "saponification_degree": None, "hull_vertices": None, "n_records": 0,
             "n_publications": 0, "_status": "assumed", "_assumed_label": "ASSUMED_PLACEHOLDER",
             "_range_note": "box transcribed from S2 conditions"}
    dom = coerce_domain(block)
    assert isinstance(dom, ApplicabilityDomain)
    assert dom.log_acid == (-1.42, -1.02) and dom.log_ligand == {"PC88A": (-0.699, 0.0)}
    assert dom.hull_vertices is None and dom.log_complexant is None and dom.modifiers == ()
    assert coerce_domain(dom) is dom and coerce_domain(None) is None
    with_hull = coerce_domain({**block, "hull_vertices": [[-1.4, -0.7], [-1.0, -0.7], [-1.2, 0.0]]})
    assert with_hull.hull_vertices == ((-1.4, -0.7), (-1.0, -0.7), (-1.2, 0.0))
    with pytest.raises(ValueError):
        coerce_domain(3.0)
    # the coerced literature box behaves as a box (no hull axis; no complexant was studied)
    flags, dist = domain_flags(dom, replace(state(c=0.0), ligand_total={"PC88A": 0.5},
                                            ligand_free={"PC88A": 0.45},
                                            y_by_ligand={"PC88A": {"Nd": 0.0}},
                                            acid_in_org={"PC88A": 0.0}), "PC88A")
    assert "hull" not in dist and flags == frozenset()
