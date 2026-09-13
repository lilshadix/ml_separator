"""``domain.py`` — applicability domains of a parameter set (DESIGN.md section 5.5).

``build_domain(records, ligand)`` derives an ``ApplicabilityDomain`` from the fit-eligible
records of one (ligand, temperature band): a box on log10 acid (mol/L), log10 ligand (mol/L,
formal concentration), log10 total metal (mM, non-NaN rows), loading fraction (computed as
``n_prior * metal_mM * 1e-3 / ligand_M`` with the ASSUMED O/A = 1 of section 4.4), temperature
(degC) and the categorical anion / diluent family / modifiers, plus the convex hull of the fit
points in (log10 acid, log10 primary ligand) via ``scipy.spatial.ConvexHull`` (``QJ``; fewer than
three distinct points or collinear points give a segment or a point whose endpoints are stored in
``hull_vertices``).

``domain_flags(domain, state, ligand)`` returns the ``OOD_*`` flags of a stage state with the
per-axis distance outside the recorded interval: log10 units on the concentration axes and O/A,
degC on the temperature axis, the loading fraction as a plain fraction, 1.0 for a categorical
mismatch, and the Euclidean distance (log10 units) to the hull polygon for ``OOD_HULL``.  A
distance of 0 means inside.  Literature domains have ``hull_vertices = None`` (box only).

Records layout: the flat ``corpus_records.csv`` mirror (one row per ``DistributionRecord``,
section 3.2).  Because the flattening of the mapping fields is the writer's choice, the column
resolvers below accept several spellings (see ``addenda/WB2.md``): the ligand concentration is
read from ``ligand_M__<ligand>``, ``ligand_M.<ligand>``, ``ligand_M_<ligand>``, a dict / JSON
``ligand_M`` column, ``extractant_M`` or ``cond__extractant_concentration_M``; the acid from
``acid_nominal_M`` or ``cond__acid_concentration_M``; the metal concentration from
``metal_initial_mM``, ``metals_initial_mM__<metal>`` (summed), a dict / JSON ``metals_initial_mM``
column or ``cond__metal_concentration_mM``.

Sources: the definitions are DESIGN.md sections 3.2, 5.5 and 12.2; no literature number enters
this module.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .types import ApplicabilityDomain, Flag, StageState

__all__ = [
    "AXIS_FLAGS", "HULL_TOL", "build_domain", "domain_flags", "hull_of_points", "hull_distance",
    "interval_distance", "coerce_domain", "resolve_column", "ligand_series", "acid_series",
    "metal_mm_series", "temperature_series", "publication_series", "truthy",
]

AXIS_FLAGS: dict[str, Flag] = {
    "acid": Flag.OOD_ACID,
    "ligand": Flag.OOD_LIGAND,
    "metal": Flag.OOD_METAL,
    "loading": Flag.OOD_LOADING,
    "anion": Flag.OOD_ANION,
    "complexant": Flag.OOD_COMPLEXANT,
    "oa": Flag.OOD_OA,
    "temperature": Flag.OOD_TEMPERATURE,
    "diluent": Flag.OOD_DILUENT,
    "modifier": Flag.OOD_MODIFIER,
    "hull": Flag.OOD_HULL,
}
"""Axis name of ``ood_distance`` -> the ``OOD_*`` flag raised when the distance exceeds
``HULL_TOL``."""

HULL_TOL = 1e-12
"""Distances at or below this (log10 units) count as inside (round-off at a vertex is not OOD)."""

_ACID_COLUMNS = ("acid_nominal_M", "cond__acid_concentration_M", "acid_M")
_METAL_MM_COLUMNS = ("metal_initial_mM", "metals_initial_mM", "cond__metal_concentration_mM",
                     "metal_mM")
_TEMPERATURE_COLUMNS = ("temperature_C", "cond__temperature_C")
_PUBLICATION_COLUMNS = ("publication_id",)
_COMPLEXANT_COLUMNS = ("complexant_M",)
_OA_COLUMNS = ("oa_ratio",)
_ANION_COLUMNS = ("anion", "medium_anion", "acid_class", "geom_cond__acid_class")
_DILUENT_COLUMNS = ("diluent_family", "geom_cond__diluent_family")
_MODIFIER_COLUMNS = ("modifiers", "modifier_names")


# ---------------------------------------------------------------------------------------------
# column resolution for the flat records layout
# ---------------------------------------------------------------------------------------------

def resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """First column name of ``candidates`` present in ``df`` (None when none is)."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def truthy(series: pd.Series) -> pd.Series:
    """Boolean view of a bool / int / string column (``"True"`` and ``"False"`` from CSV)."""
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y", "t"))


def _mapping_lookup(cell: Any, key: str) -> float:
    """Value of ``key`` inside a dict-valued or JSON-string cell (NaN when absent)."""
    if isinstance(cell, Mapping):
        value = cell.get(key)
    elif isinstance(cell, str):
        try:
            value = json.loads(cell).get(key)
        except (ValueError, AttributeError):
            return math.nan
    else:
        return math.nan
    return math.nan if value is None else float(value)


def _mapping_sum(cell: Any) -> float:
    """Sum of the values of a dict-valued or JSON-string cell (NaN when absent or empty)."""
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except ValueError:
            return math.nan
    if isinstance(cell, Mapping):
        values = [float(v) for v in cell.values() if v is not None]
        return float(sum(values)) if values else math.nan
    return math.nan


def ligand_series(df: pd.DataFrame, ligand: str) -> pd.Series:
    """Formal ligand concentration (mol/L) per record for ``ligand`` under the accepted spellings.

    Raises ``ValueError`` when no spelling is present.
    """
    for name in (f"ligand_M__{ligand}", f"ligand_M.{ligand}", f"ligand_M_{ligand}"):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").astype(float)
    if "ligand_M" in df.columns:
        col = df["ligand_M"]
        if pd.api.types.is_numeric_dtype(col):
            # WB1's corpus_records.csv: one numeric column for the primary ligand named by
            # ``ligand_name``; rows of another ligand carry no concentration for ``ligand``.
            values = col.astype(float)
            if "ligand_name" in df.columns:
                values = values.where(df["ligand_name"].astype(str) == ligand, np.nan)
            return values
        return col.map(lambda cell: _mapping_lookup(cell, ligand)).astype(float)
    name = resolve_column(df, ("extractant_M", "cond__extractant_concentration_M"))
    if name is None:
        raise ValueError(f"records carry no ligand concentration column for {ligand!r}")
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def other_ligand_columns(df: pd.DataFrame, ligand: str) -> dict[str, pd.Series]:
    """Explicit ``ligand_M__<name>`` columns of ligands other than ``ligand``."""
    out: dict[str, pd.Series] = {}
    for col in df.columns:
        for prefix in ("ligand_M__", "ligand_M."):
            if col.startswith(prefix):
                name = col[len(prefix):]
                if name != ligand:
                    out[name] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return out


def acid_series(df: pd.DataFrame) -> pd.Series:
    """Nominal acid concentration (mol/L) per record."""
    name = resolve_column(df, _ACID_COLUMNS)
    if name is None:
        raise ValueError("records carry no acid concentration column")
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def metal_mm_series(df: pd.DataFrame) -> pd.Series:
    """Total initial metal concentration (mM) per record; NaN when not recorded."""
    per_metal = [c for c in df.columns
                 if c.startswith("metals_initial_mM__") or c.startswith("metals_initial_mM.")]
    if per_metal:
        block = df[per_metal].apply(pd.to_numeric, errors="coerce").astype(float)
        total = block.sum(axis=1, min_count=1)
        return total.astype(float)
    name = resolve_column(df, _METAL_MM_COLUMNS)
    if name is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    col = df[name]
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float)
    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.notna().any():
        return numeric.astype(float)
    return col.map(_mapping_sum).astype(float)


def temperature_series(df: pd.DataFrame) -> pd.Series:
    name = resolve_column(df, _TEMPERATURE_COLUMNS)
    if name is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def publication_series(df: pd.DataFrame) -> pd.Series | None:
    name = resolve_column(df, _PUBLICATION_COLUMNS)
    return None if name is None else df[name]


def _optional_numeric(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    name = resolve_column(df, candidates)
    if name is None:
        return None
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def _first_categorical(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    name = resolve_column(df, candidates)
    if name is None:
        return None
    values = df[name].dropna().astype(str)
    values = values[values.str.len() > 0]
    if values.empty:
        return None
    return str(values.mode().iloc[0])


# ---------------------------------------------------------------------------------------------
# intervals, hull
# ---------------------------------------------------------------------------------------------

def interval_distance(value: float, interval: tuple[float, float] | None) -> float:
    """Distance of ``value`` outside ``[lo, hi]`` (0 inside; NaN value -> NaN)."""
    if interval is None or value is None or not math.isfinite(value):
        return math.nan
    lo, hi = float(interval[0]), float(interval[1])
    if value < lo:
        return lo - value
    if value > hi:
        return value - hi
    return 0.0


def _interval(values: pd.Series | np.ndarray) -> tuple[float, float] | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return (float(arr.min()), float(arr.max()))


def hull_of_points(points: np.ndarray) -> tuple[tuple[float, float], ...] | None:
    """Convex hull vertices (counter-clockwise) of 2-D ``points``; a segment's two endpoints or a
    single point for degenerate inputs; None for no points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array")
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] == 0:
        return None
    pts = np.unique(pts, axis=0)
    if pts.shape[0] == 1:
        return ((float(pts[0, 0]), float(pts[0, 1])),)
    centred = pts - pts.mean(axis=0)
    svals = np.linalg.svd(centred, compute_uv=False)
    if pts.shape[0] == 2 or svals[-1] <= 1e-9 * max(svals[0], 1e-300):
        # collinear: the two extreme points along the principal direction
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        proj = centred @ vt[0]
        lo, hi = int(np.argmin(proj)), int(np.argmax(proj))
        return ((float(pts[lo, 0]), float(pts[lo, 1])), (float(pts[hi, 0]), float(pts[hi, 1])))
    from scipy.spatial import ConvexHull  # local import keeps module import light

    hull = ConvexHull(pts, qhull_options="QJ")
    verts = [(float(pts[i, 0]), float(pts[i, 1])) for i in hull.vertices]
    if _signed_area(verts) < 0:
        verts.reverse()
    return tuple(verts)


def _signed_area(verts: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0.0:
        return float(np.hypot(*(p - a)))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.hypot(*(p - (a + t * ab))))


def hull_distance(vertices: Sequence[tuple[float, float]] | None,
                  point: tuple[float, float]) -> float:
    """Euclidean distance from ``point`` to the polygon (segment, point) described by
    ``vertices``; 0 inside or on the boundary; NaN when ``vertices`` is None."""
    if vertices is None:
        return math.nan
    p = np.asarray(point, dtype=float)
    if not np.all(np.isfinite(p)):
        return math.nan
    verts = np.asarray(vertices, dtype=float)
    n = verts.shape[0]
    if n == 0:
        return math.nan
    if n == 1:
        return float(np.hypot(*(p - verts[0])))
    if n == 2:
        return _segment_distance(p, verts[0], verts[1])
    if _signed_area([tuple(v) for v in verts]) < 0:
        verts = verts[::-1]
    edges = np.roll(verts, -1, axis=0) - verts
    rel = p[None, :] - verts
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    if np.all(cross >= -HULL_TOL):
        return 0.0
    return min(_segment_distance(p, verts[i], verts[(i + 1) % n]) for i in range(n))


# ---------------------------------------------------------------------------------------------
# build_domain
# ---------------------------------------------------------------------------------------------

def build_domain(records: pd.DataFrame, ligand: str, *, anion: str | None = None,
                 diluent_family: str | None = None, modifiers: Iterable[str] | None = None,
                 band: str | None = None, n_prior: float = 3.0) -> ApplicabilityDomain:
    """Applicability domain of one (ligand, band) from its fit-eligible records.

    ``records`` is the flat records frame (module docstring); rows with ``fit_eligible`` false
    are dropped when the column exists; ``band`` (e.g. ``"20-30C"``) filters on the temperature
    column when given (NaN temperature counts as ``"20-30C"``, section 3.4).  The categorical
    fields are taken from the keyword arguments, else from the ``anion`` / ``acid_class``,
    ``diluent_family`` and ``modifiers`` columns, else ``"unknown"`` / empty.
    """
    df = records
    if "fit_eligible" in df.columns:
        df = df[truthy(df["fit_eligible"])]
    if band is not None:
        temp = temperature_series(df)
        lo, hi = _parse_band(band)
        filled = temp.fillna(25.0)
        df = df[(filled >= lo) & (filled < hi)]
    acid = acid_series(df)
    lig = ligand_series(df, ligand)
    ok = (acid > 0) & (lig > 0) & acid.notna() & lig.notna()
    df = df[ok]
    acid = acid[ok]
    lig = lig[ok]
    la = np.log10(acid.to_numpy(dtype=float))
    le = np.log10(lig.to_numpy(dtype=float))

    log_ligand: dict[str, tuple[float, float]] = {}
    li = _interval(le)
    if li is not None:
        log_ligand[ligand] = li
    for name, series in other_ligand_columns(df, ligand).items():
        vals = series.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            log_ligand[name] = (float(np.log10(vals.min())), float(np.log10(vals.max())))

    metal_mm = metal_mm_series(df).to_numpy(dtype=float)
    has_metal = np.isfinite(metal_mm) & (metal_mm > 0)
    log_metal = _interval(np.log10(metal_mm[has_metal])) if has_metal.any() else None
    loading = None
    if has_metal.any():
        lam = n_prior * metal_mm[has_metal] * 1e-3 / lig.to_numpy(dtype=float)[has_metal]
        loading = _interval(lam)

    comp = _optional_numeric(df, _COMPLEXANT_COLUMNS)
    log_complexant = None
    if comp is not None:
        vals = comp.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            log_complexant = (float(np.log10(vals.min())), float(np.log10(vals.max())))
    oa = _optional_numeric(df, _OA_COLUMNS)
    oa_ratio = _interval(oa) if oa is not None else None
    temperature = _interval(temperature_series(df))

    if anion is None:
        anion = _first_categorical(df, _ANION_COLUMNS) or "unknown"
    if diluent_family is None:
        diluent_family = _first_categorical(df, _DILUENT_COLUMNS) or "unknown"
    if modifiers is None:
        mod_text = _first_categorical(df, _MODIFIER_COLUMNS)
        modifiers = tuple(m for m in (mod_text or "").split("|") if m) if mod_text else ()
    pubs = publication_series(df)
    n_pub = int(pubs.dropna().astype(str).nunique()) if pubs is not None else 0

    hull = hull_of_points(np.column_stack([la, le])) if la.size else None
    return ApplicabilityDomain(
        anion=str(anion), diluent_family=str(diluent_family), modifiers=tuple(sorted(modifiers)),
        log_acid=_interval(la) or (math.nan, math.nan), log_ligand=log_ligand,
        log_metal_total_mM=log_metal, loading_fraction=loading, log_complexant=log_complexant,
        oa_ratio=oa_ratio, temperature_C=temperature, saponification_degree=None,
        hull_vertices=hull, n_records=int(len(df)), n_publications=n_pub,
    )


def _parse_band(band: str) -> tuple[float, float]:
    """``"<20C"`` -> (-inf, 20); ``"20-30C"`` -> (20, 30); ``">=50C"`` -> (50, inf)."""
    text = band.strip().rstrip("C").rstrip("c")
    if text.startswith("<="):
        return (-math.inf, float(text[2:]) + 1e-9)
    if text.startswith("<"):
        return (-math.inf, float(text[1:]))
    if text.startswith(">="):
        return (float(text[2:]), math.inf)
    if text.startswith(">"):
        return (float(text[1:]) + 1e-9, math.inf)
    lo, hi = text.split("-")
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------------------------
# domain_flags
# ---------------------------------------------------------------------------------------------

def _log10(x: float) -> float:
    return math.log10(x) if x > 0 else -math.inf


def domain_flags(domain: ApplicabilityDomain | None, state: StageState, ligand: str, *,
                 anion: str | None = None, diluent_family: str | None = None,
                 modifiers: Iterable[str] | None = None, ligand_scale: float = 1.0,
                 loading: float | None = None,
                 ) -> tuple[frozenset[Flag], dict[str, float]]:
    """``OOD_*`` flags and per-axis distances of ``state`` against ``domain`` (module docstring).

    Axes whose interval is None in the domain are absent from the distance dict, except the
    complexant axis: a stage with free complexant against a domain without a complexant interval
    is out of domain at distance 1.0.  ``ligand`` selects the ligand axis and, when it is the
    domain's primary ligand (first key of ``log_ligand``), the hull test; ``ligand_scale``
    converts ``state.ligand_total`` (code basis: dimer for dimers) to the formal concentration
    of the records (2 for a dimeric ligand).  ``loading`` is the loading fraction ``lambda_k`` of
    section 5.4 when the caller knows it (``dmodel.ligand_loading_fraction``); otherwise it is
    read off the ligand balance as ``1 - L_f / L_T`` (exact without ``EffectiveCapacity``).  The
    categorical axes are checked only when the corresponding keyword is given.
    """
    if domain is None:
        return frozenset(), {}
    dist: dict[str, float] = {}
    la = _log10(state.h)
    dist["acid"] = interval_distance(la, domain.log_acid)
    lt = state.ligand_total.get(ligand)
    if lt is not None and ligand in domain.log_ligand:
        dist["ligand"] = interval_distance(_log10(lt * ligand_scale), domain.log_ligand[ligand])
    r = state.v_org_L / state.v_aq_L if state.v_aq_L > 0 else math.nan
    if domain.log_metal_total_mM is not None:
        total = sum(state.x_total.values()) + r * sum(state.y.values())
        if total > 0:
            dist["metal"] = interval_distance(math.log10(total * 1e3), domain.log_metal_total_mM)
    if domain.loading_fraction is not None and lt:
        if loading is None or not math.isfinite(loading):
            lf = state.ligand_free.get(ligand, lt)
            loading = 1.0 - lf / lt
        dist["loading"] = interval_distance(loading, domain.loading_fraction)
    if state.c_free > 0:
        if domain.log_complexant is None:
            dist["complexant"] = 1.0
        else:
            dist["complexant"] = interval_distance(math.log10(state.c_free), domain.log_complexant)
    if domain.oa_ratio is not None and r > 0:
        lo, hi = domain.oa_ratio
        dist["oa"] = interval_distance(math.log10(r), (_log10(lo), _log10(hi)))
    if domain.temperature_C is not None:
        dist["temperature"] = interval_distance(state.temperature_C, domain.temperature_C)
    if anion is not None:
        dist["anion"] = 0.0 if anion == domain.anion else 1.0
    if diluent_family is not None:
        dist["diluent"] = 0.0 if diluent_family == domain.diluent_family else 1.0
    if modifiers is not None:
        same = tuple(sorted(modifiers)) == tuple(sorted(domain.modifiers))
        dist["modifier"] = 0.0 if same else 1.0
    if domain.hull_vertices is not None and lt is not None and domain.log_ligand:
        primary = next(iter(domain.log_ligand))
        if primary == ligand:
            dist["hull"] = hull_distance(domain.hull_vertices, (la, _log10(lt * ligand_scale)))
    dist = {k: float(v) for k, v in dist.items() if not (isinstance(v, float) and math.isnan(v))}
    flags = frozenset(AXIS_FLAGS[k] for k, v in dist.items() if v > HULL_TOL)
    return flags, dist


# ---------------------------------------------------------------------------------------------
# coercion of a JSON-shaped applicability block
# ---------------------------------------------------------------------------------------------

def _pair(obj: Any) -> tuple[float, float] | None:
    if obj is None:
        return None
    return (float(obj[0]), float(obj[1]))


def coerce_domain(obj: Any) -> ApplicabilityDomain | None:
    """An ``ApplicabilityDomain`` from a dataclass instance or the JSON block of section 3.7
    (keys starting with ``_`` are ignored); None stays None."""
    if obj is None or isinstance(obj, ApplicabilityDomain):
        return obj
    if not isinstance(obj, Mapping):
        raise ValueError("applicability block must be a mapping or ApplicabilityDomain")
    log_ligand = {str(k): _pair(v) for k, v in (obj.get("log_ligand") or {}).items()}
    hull = obj.get("hull_vertices")
    return ApplicabilityDomain(
        anion=str(obj.get("anion", "unknown")),
        diluent_family=str(obj.get("diluent_family", "unknown")),
        modifiers=tuple(obj.get("modifiers") or ()),
        log_acid=_pair(obj.get("log_acid")) or (math.nan, math.nan),
        log_ligand=log_ligand,
        log_metal_total_mM=_pair(obj.get("log_metal_total_mM")),
        loading_fraction=_pair(obj.get("loading_fraction")),
        log_complexant=_pair(obj.get("log_complexant")),
        oa_ratio=_pair(obj.get("oa_ratio")),
        temperature_C=_pair(obj.get("temperature_C")),
        saponification_degree=_pair(obj.get("saponification_degree")),
        hull_vertices=None if hull is None else tuple((float(a), float(b)) for a, b in hull),
        n_records=int(obj.get("n_records", 0) or 0),
        n_publications=int(obj.get("n_publications", 0) or 0),
    )
