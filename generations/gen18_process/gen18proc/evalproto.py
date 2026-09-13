"""``evalproto.py`` -- the pre-registered evaluation of the corpus-fitted D model (DESIGN.md
section 10.1-10.3, 10.5; PRE_REGISTRATION.md sections 3-6).

What lives here
---------------
* ``fit_mass_action`` -- model **M1**: the pooled-slope mass-action fit of one (system, ligand,
  temperature band): ``log D_k = a_m(k) + n * log10[L]_k + p_eff * log10[acid]_k + delta_p(k)``,
  one intercept per metal, slopes shared across metals, sum-to-zero publication effects for
  publications with >= 2 aggregated points (single-point publications: effect 0), weighted
  unpenalised least squares (weight = number of replicate rows), a slope fixed at the mechanism
  prior when its axis has < 3 distinct levels.  Classical and HC1 covariances.
* ``lopo_evaluate`` -- leave-one-publication-out scoring of M1 against the baselines **B0**
  (training mean of the metal) and **B1** (same-metal nearest condition in (log acid, log ligand)),
  plus the labelled variants ``M1_offset`` (one held-out point sets the publication effect) and
  ``B1_crossmetal`` (exploratory X3).  ``in_sample_evaluate`` gives the labelled in-sample rows.
* ``macro_over_systems``, ``paired_system_bootstrap`` -- the E1 averaging order and the R1
  interval.
* ``jackknife_by_publication``, ``split_half_by_publication`` -- reliability before interpretation
  (E3).
* ``ComparisonCounter`` -- every contrast counted; Benjamini-Hochberg over the exploratory family.
* ``as_solvating_params`` / ``as_cation_exchange_params`` -- the fitted block of DESIGN.md 3.8.
* ``loading_series_evaluate`` -- the loading endpoint E2 (WB4b): C0 (tracer value everywhere)
  against C1 (ideal ligand depletion through ``equilibrium.solve_stage`` with ``log K`` anchored
  at the tracer point), the descriptive Spearman, and the exploratory ``EffectiveCapacity(phi)``
  arm; the stage solver is imported lazily so this module stays light.

Input layout
------------
Every function consumes a pandas DataFrame in the flat ``systems/corpus_records.csv`` layout of
DESIGN.md section 3.2 (``DistributionRecord``): ``record_id, system_id, metal, d, log_d,
acid_nominal_M, anion_M, metals_initial_mM, temperature_C, publication_id, loading_series_id,
is_tracer, fit_eligible, duplicate_flag`` and the ligand concentrations in any of three forms
(``addenda/WB4a.md`` A1): (i) the form WB1 writes -- a **scalar** ``ligand_M`` column (the
primary extractant's formal concentration) beside ``ligand_name``, and a scalar
``metal_initial_mM`` (the row's own metal); (ii) one flattened column per ligand named
``ligand_M.<name>`` (``ligand_M__<name>`` is accepted too) and ``metals_initial_mM.<metal>``;
(iii) a single nested column ``ligand_M`` / ``metals_initial_mM`` holding a mapping or its JSON
string.  The replicate-group key is ``replicate_group_id`` when WB1 wrote it (A2).

Units: concentrations mol/L (``metals_initial_mM`` mM); ``log`` means log10; temperature degC.
Seeds default to 18 (``numpy.random.default_rng``).  Nothing here raises on a degenerate fit:
degenerate quantities are NaN and statuses say why.  ``ValueError`` only on malformed input.
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gen18proc.types import (
    ASSUMED_LABEL,
    ApplicabilityDomain,
    CationExchangeParams,
    Flag,
    Mechanism,
    Provenance,
    ProvStatus,
    SolvatingParams,
    Source,
    Sourced,
)

__all__ = [
    "FitResult", "MacroResult", "ComparisonCounter",
    "fit_mass_action", "lopo_evaluate", "in_sample_evaluate", "macro_over_systems",
    "paired_system_bootstrap", "jackknife_by_publication", "split_half_by_publication",
    "as_solvating_params", "as_cation_exchange_params", "loading_series_evaluate",
    "prepare_records", "ligand_column", "metals_initial_column", "aggregate_replicates",
    "band_contains", "DEFAULT_BAND", "MAE_COLUMNS",
]

DEFAULT_BAND = "20-30C"
"""Band assigned to records with NaN temperature (DESIGN.md section 3.4)."""

MAE_COLUMNS = ("mae_M1", "mae_B0", "mae_B1", "mae_M1_offset", "mae_B1_crossmetal")
"""Error columns of the LOPO table, in the order of DESIGN.md section 10.2."""

_CONDITION_COLUMNS = ("acid_nominal_M", "anion_M", "temperature_C", "diluent_name",
                      "contact_time_min", "complexant_M")
"""Flat condition columns that (with the ligand and metal-concentration columns, the metal and
the publication) define a replicate group when no explicit ``replicate_group`` column exists."""


# =============================================================================================
# Input handling: nested / flattened mapping columns, band filter, replicate aggregation
# =============================================================================================

def _parse_mapping(obj: Any) -> dict[str, float]:
    """A mapping cell: dict, JSON string, or NaN/None -> dict."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return {str(k): float(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, float) and math.isnan(obj):
        return {}
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {}
        parsed = json.loads(s)
        if not isinstance(parsed, Mapping):
            raise ValueError(f"mapping column holds a non-object JSON value: {obj!r}")
        return {str(k): float(v) for k, v in parsed.items() if v is not None}
    raise ValueError(f"cannot interpret {type(obj).__name__} as a mapping column")


_SCALAR_FORM = {"ligand_M": ("ligand_M", "ligand_name"),
                "metals_initial_mM": ("metal_initial_mM", "metal")}
"""Mapping base -> (scalar column, selector column) of the scalar form WB1 writes: the scalar
column holds the value for the entity named in the selector column (the primary ligand's name,
the row's own metal); every other key is NaN."""


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _mapping_column(df: pd.DataFrame, base: str, key: str) -> pd.Series:
    """Values of ``base[key]`` per row under the three accepted forms (module docstring):
    flattened ``<base>.<key>`` / ``<base>__<key>`` column; the scalar form (``ligand_M`` +
    ``ligand_name``, ``metal_initial_mM`` + ``metal``); a nested ``<base>`` column (mapping or
    JSON string).  NaN where the key is absent from a row."""
    for flat in (f"{base}.{key}", f"{base}__{key}"):
        if flat in df.columns:
            return pd.to_numeric(df[flat], errors="coerce").astype(float)
    scalar_col, selector = _SCALAR_FORM[base]
    if scalar_col in df.columns and (scalar_col != base
                                     or pd.api.types.is_numeric_dtype(df[scalar_col])):
        vals = pd.to_numeric(df[scalar_col], errors="coerce").astype(float)
        if selector in df.columns:
            vals = vals.where(df[selector].astype(str) == key, np.nan)
        return vals
    if base in df.columns:
        out = []
        sel = df[selector].astype(str).tolist() if selector in df.columns else None
        for i, v in enumerate(df[base].tolist()):
            if _is_number(v):   # a scalar cell inside an object column: the scalar rule
                out.append(float(v) if (sel is None or sel[i] == key) else np.nan)
            else:
                out.append(_parse_mapping(v).get(key, np.nan))
        return pd.Series(out, index=df.index, dtype=float)
    raise ValueError(f"records carry no '{base}.{key}', '{scalar_col}' or nested '{base}' column")


def ligand_column(df: pd.DataFrame, ligand: str) -> pd.Series:
    """Formal ligand concentration (mol/L) of ``ligand`` per row (flattened or nested form)."""
    return _mapping_column(df, "ligand_M", ligand)


def metals_initial_column(df: pd.DataFrame, metal: str) -> pd.Series:
    """Initial aqueous concentration (mM) of ``metal`` per row (flattened or nested form)."""
    return _mapping_column(df, "metals_initial_mM", metal)


def _mapping_key_series(df: pd.DataFrame, base: str) -> pd.Series:
    """A canonical hashable string per row of a mapping column (any of the three forms), for
    grouping when no explicit replicate-group column exists."""
    flat_cols = sorted(c for c in df.columns
                       if c.startswith(base + ".") or c.startswith(base + "__"))
    if flat_cols:
        parts = df[flat_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        return parts.apply(
            lambda r: json.dumps({c.split(".", 1)[-1].split("__", 1)[-1]:
                                  (None if pd.isna(v) else float(v)) for c, v in r.items()},
                                 sort_keys=True), axis=1)
    scalar_col, selector = _SCALAR_FORM[base]
    if scalar_col in df.columns and (scalar_col != base
                                     or pd.api.types.is_numeric_dtype(df[scalar_col])):
        vals = pd.to_numeric(df[scalar_col], errors="coerce").astype(float).map(
            lambda v: "nan" if pd.isna(v) else repr(float(v)))
        if selector in df.columns:
            return df[selector].astype(str) + "=" + vals
        return vals
    if base in df.columns:
        return pd.Series([repr(float(v)) if _is_number(v) else
                          json.dumps(_parse_mapping(v), sort_keys=True) for v in df[base]],
                         index=df.index)
    return pd.Series([""] * len(df), index=df.index)


def band_contains(band: str | None, temperature_C: float | None) -> bool:
    """Whether ``temperature_C`` falls in ``band`` (``"<lo>-<hi>C"`` as ``lo <= T < hi``,
    ``"<20C"``, ``">=50C"``); NaN temperature is assigned to ``DEFAULT_BAND`` (section 3.4);
    ``band None`` accepts everything."""
    if band is None:
        return True
    if temperature_C is None or (isinstance(temperature_C, float) and math.isnan(temperature_C)):
        return band == DEFAULT_BAND
    t = float(temperature_C)
    b = band.strip()
    if b.startswith(">="):
        return t >= float(b[2:].rstrip("C"))
    if b.startswith("<"):
        return t < float(b[1:].rstrip("C"))
    lo, hi = b.rstrip("C").split("-")
    return float(lo) <= t < float(hi)


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"records are missing required columns: {missing}")


def _as_bool(s: pd.Series, default: bool) -> pd.Series:
    if s.dtype == bool:
        return s
    out = s.map(lambda v: default if (v is None or (isinstance(v, float) and math.isnan(v)))
                else (str(v).strip().lower() in {"true", "1", "yes", "t"}))
    return out.astype(bool)


def prepare_records(records: pd.DataFrame, *, ligand: str, band: str | None,
                    mechanism: Mechanism | str = Mechanism.SOLVATING) -> pd.DataFrame:
    """Fit-eligible rows of one ligand and band with the fit axes attached.

    Keeps rows with ``fit_eligible`` true, ``duplicate_flag != "UNIT_SLIP_DUPLICATE"``, finite
    ``log_d``, positive acid and ligand concentration, temperature inside ``band`` (NaN -> the
    default band).  Adds ``lE`` = log10 of the ligand concentration (dimer basis, i.e. of
    ``ligand_M / 2``, for cation exchange; DESIGN.md section 1.1) and ``lA`` = log10 of the
    nominal acid.  Never raises on an empty result.
    """
    _require_columns(records, ("record_id", "metal", "log_d", "acid_nominal_M",
                               "publication_id"))
    df = records.copy()
    mech = Mechanism(mechanism) if not isinstance(mechanism, Mechanism) else mechanism
    lig = ligand_column(df, ligand)
    if mech == Mechanism.CATION_EXCHANGE:
        lig = lig / 2.0
    df["_ligand_M"] = lig
    df["_acid_M"] = pd.to_numeric(df["acid_nominal_M"], errors="coerce").astype(float)
    df["log_d"] = pd.to_numeric(df["log_d"], errors="coerce").astype(float)
    if "system_id" not in df.columns:
        df["system_id"] = "unspecified"
    keep = np.isfinite(df["log_d"].to_numpy()) & (df["_acid_M"].to_numpy() > 0) \
        & (df["_ligand_M"].to_numpy() > 0)
    if "fit_eligible" in df.columns:
        keep &= _as_bool(df["fit_eligible"], True).to_numpy()
    if "duplicate_flag" in df.columns:
        keep &= (df["duplicate_flag"].astype("string").fillna("") != "UNIT_SLIP_DUPLICATE") \
            .to_numpy()
    if "temperature_C" in df.columns:
        temps = pd.to_numeric(df["temperature_C"], errors="coerce").astype(float)
        keep &= np.array([band_contains(band, t) for t in temps.tolist()], dtype=bool)
    elif "temperature_band" in df.columns and band is not None:
        # WB1 also writes the band string; used only when the temperature itself is absent
        keep &= (df["temperature_band"].astype("string").fillna(DEFAULT_BAND) == band).to_numpy()
    df = df.loc[keep].copy()
    df["lE"] = np.log10(df["_ligand_M"].to_numpy(dtype=float))
    df["lA"] = np.log10(df["_acid_M"].to_numpy(dtype=float))
    df["record_id"] = df["record_id"].astype(str)
    df["publication_id"] = df["publication_id"].astype("string").fillna("pub_unknown").astype(str)
    df["metal"] = df["metal"].astype(str)
    return df


def _replicate_key(df: pd.DataFrame) -> pd.Series:
    """Replicate-group key per row: an explicit ``replicate_group_id`` (WB1's hash of the exact
    64-column condition key + publication + SMILES + metal), ``replicate_group`` or
    ``condition_key`` column when present, else the tuple of every flat condition column
    available plus the ligand and metal-concentration mappings, the metal and the publication."""
    for explicit in ("replicate_group_id", "replicate_group", "condition_key"):
        if explicit in df.columns:
            return df["publication_id"] + "|" + df["metal"] + "|" + df[explicit].astype(str)
    parts = [df["publication_id"], df["metal"]]
    for c in _CONDITION_COLUMNS:
        if c in df.columns:
            parts.append(df[c].astype("string").fillna("nan").astype(str))
    parts.append(_mapping_key_series(df, "ligand_M"))
    parts.append(_mapping_key_series(df, "metals_initial_mM"))
    key = parts[0].astype(str)
    for p in parts[1:]:
        key = key + "|" + p.astype(str)
    return key


def aggregate_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate groups (DESIGN.md section 10.1 step 1) -> one point each.

    Output columns: ``system_id, publication_id, metal, record_id`` (the smallest of the group),
    ``lE, lA, log_d`` (mean over the group), ``n_rep`` (weight), ``record_ids`` (tuple).
    """
    if df.empty:
        return pd.DataFrame(columns=["system_id", "publication_id", "metal", "record_id", "lE",
                                     "lA", "log_d", "n_rep", "record_ids"])
    work = df.assign(_key=_replicate_key(df))
    has_w = "n_rep_weight" in work.columns   # pre-aggregated points carry their weight
    rows = []
    for _, g in work.groupby("_key", sort=True, dropna=False):
        ids = sorted(set(g["record_id"].tolist()))
        wts = g["n_rep_weight"].to_numpy(dtype=float) if has_w else np.ones(len(g))
        rows.append({
            "system_id": str(g["system_id"].iloc[0]),
            "publication_id": str(g["publication_id"].iloc[0]),
            "metal": str(g["metal"].iloc[0]),
            "record_id": ids[0],
            "lE": float(np.average(g["lE"].to_numpy(dtype=float), weights=wts)),
            "lA": float(np.average(g["lA"].to_numpy(dtype=float), weights=wts)),
            "log_d": float(np.average(g["log_d"].to_numpy(dtype=float), weights=wts)),
            "n_rep": int(round(wts.sum())),
            "record_ids": tuple(ids),
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["metal", "publication_id", "record_id"], kind="stable")
    return out.reset_index(drop=True)


# =============================================================================================
# 10.1  Model M1
# =============================================================================================

@dataclass(frozen=True)
class FitResult:
    """Result of ``fit_mass_action`` (DESIGN.md section 10.1).

    ``intercepts`` per metal (log D at [L] = 1 M and [acid] = 1 M, publication effect 0);
    ``n`` slope on log10 ligand, ``p_eff`` slope on log10 nominal acid (for cation exchange
    ``p_eff = -b_proton``); ``publication_effects`` per publication (0 for single-point
    publications); ``covariance`` classical ``(X'WX)^-1 s^2`` over ``param_names``; ``se`` per
    parameter name; ``jackknife_se`` / ``split_half`` from section 10.3 (None when not computed
    or undefined); ``interpretable`` per slope; ``n_points`` aggregated points;
    ``n_publications``; ``domain`` (None unless ``gen18proc.domain`` is importable);
    ``fit_manifest_sha256`` filled by the fit script.  Extra fields (after the contract's):
    ``slope_status`` ``{"n": "fitted"|"assumed", "p_eff": ...}``, ``flags``, ``se_hc1``,
    ``covariance_hc1``, ``param_names``, ``residual_sd``, ``distinct_levels``,
    ``record_ids_by_metal`` (training record ids per metal), ``mechanism``, ``status``.
    """

    system_id: str
    ligand: str
    band: str | None
    intercepts: dict[str, float]
    n: float
    p_eff: float
    publication_effects: dict[str, float]
    covariance: np.ndarray
    se: dict[str, float]
    jackknife_se: dict[str, tuple[float, float]] | None
    split_half: dict[str, Any] | None
    interpretable: dict[str, bool]
    n_points: int
    n_publications: int
    domain: ApplicabilityDomain | None
    fit_manifest_sha256: str | None
    slope_status: dict[str, str] = field(default_factory=dict)
    flags: frozenset[Flag] = frozenset()
    se_hc1: dict[str, float] = field(default_factory=dict)
    covariance_hc1: np.ndarray | None = None
    param_names: tuple[str, ...] = ()
    residual_sd: float = float("nan")
    distinct_levels: dict[str, int] = field(default_factory=dict)
    record_ids_by_metal: dict[str, tuple[str, ...]] = field(default_factory=dict)
    mechanism: str = Mechanism.SOLVATING.value
    status: str = "fitted"
    n_rows: int = 0
    priors: dict[str, float] = field(default_factory=dict)

    def predict(self, metal: str, lE: float | np.ndarray, lA: float | np.ndarray,
                publication_id: str | None = None) -> float | np.ndarray:
        """``a_metal + n lE + p_eff lA + delta_pub`` (effect 0 for an unknown publication)."""
        a = self.intercepts[metal]
        d = self.publication_effects.get(publication_id, 0.0) if publication_id else 0.0
        lE_arr = np.asarray(lE, dtype=float)
        lA_arr = np.asarray(lA, dtype=float)
        return a + self.n * lE_arr + self.p_eff * lA_arr + d


def _design(points: pd.DataFrame, metals: Sequence[str], pubs_multi: Sequence[str],
            fit_n: bool, fit_p: bool) -> tuple[np.ndarray, list[str]]:
    """Design matrix: metal intercepts, shared slopes, sum-to-zero publication effects."""
    N = len(points)
    cols: list[np.ndarray] = []
    names: list[str] = []
    metal_arr = points["metal"].to_numpy()
    for m in metals:
        cols.append((metal_arr == m).astype(float))
        names.append(f"a.{m}")
    if fit_n:
        cols.append(points["lE"].to_numpy(dtype=float))
        names.append("n")
    if fit_p:
        cols.append(points["lA"].to_numpy(dtype=float))
        names.append("p_eff")
    pub_arr = points["publication_id"].to_numpy()
    if len(pubs_multi) >= 2:
        last = pubs_multi[-1]
        ref = (pub_arr == last).astype(float)
        for p in pubs_multi[:-1]:
            cols.append((pub_arr == p).astype(float) - ref)
            names.append(f"pub.{p}")
    X = np.column_stack(cols) if cols else np.zeros((N, 0))
    return X, names


def fit_mass_action(records: pd.DataFrame, *, mechanism: Mechanism | str = Mechanism.SOLVATING,
                    ligand: str, band: str | None = DEFAULT_BAND, pooled_slopes: bool = True,
                    n_prior: float = 3.0, p_prior: float = 2.0, seed: int = 18,
                    reliability: bool = True, split_half_repeats: int = 20,
                    fit_manifest_sha256: str | None = None) -> FitResult:
    """Model M1 of PRE_REGISTRATION.md section 3 on the records of ONE system.

    Steps (DESIGN.md section 10.1): (1) fit-eligible records of the (ligand, band), replicate
    groups aggregated to mean log D with weight ``n_rep``; (2) ``lE = log10 ligand``, ``lA =
    log10 acid``; (3) intercept per metal, shared slopes, sum-to-zero publication effects for
    publications with >= 2 aggregated points; (4) an axis with < 3 distinct levels has its slope
    fixed at the prior (status ``assumed``, flagged); (5) weighted unpenalised least squares by
    ``numpy.linalg.lstsq``; covariance ``(X'WX)^-1 s^2`` and the HC1 sandwich; (6) reliability
    (section 10.3) when ``reliability`` is true.

    ``pooled_slopes=False`` is the exploratory arm X2 (metal-specific slopes); it is accepted and
    fitted with one ``n``/``p_eff`` column per metal (names ``n.<metal>``, ``p_eff.<metal>``),
    ``n`` / ``p_eff`` of the result then hold the unweighted mean over metals.

    Raises ``ValueError`` only on malformed input (missing columns, more than one ``system_id``).
    A frame with no usable point returns a result with ``status = "no_data"`` and NaN values.
    """
    mech = Mechanism(mechanism) if not isinstance(mechanism, Mechanism) else mechanism
    prepared = prepare_records(records, ligand=ligand, band=band, mechanism=mech)
    systems = sorted(set(prepared["system_id"].tolist()))
    if len(systems) > 1:
        raise ValueError(f"fit_mass_action fits one system at a time; got {systems}")
    system_id = systems[0] if systems else (
        str(records["system_id"].iloc[0]) if "system_id" in records.columns and len(records)
        else "unspecified")
    points = aggregate_replicates(prepared)
    flags = {Flag.EQUILIBRIUM_ACID_ASSUMED_NOMINAL, Flag.OA_ASSUMED}
    priors = {"n": float(n_prior), "p_eff": float(p_prior)}
    if points.empty:
        return FitResult(system_id, ligand, band, {}, float("nan"), float("nan"), {},
                         np.zeros((0, 0)), {}, None, None, {"n": False, "p_eff": False}, 0, 0,
                         None, fit_manifest_sha256, {"n": "assumed", "p_eff": "assumed"},
                         frozenset(flags), {}, None, (), float("nan"), {"lE": 0, "lA": 0}, {},
                         mech.value, "no_data", int(len(prepared)), priors)

    metals = sorted(set(points["metal"].tolist()))
    pub_counts = points["publication_id"].value_counts()
    pubs_all = sorted(pub_counts.index.tolist())
    pubs_multi = sorted(p for p in pubs_all if pub_counts[p] >= 2)
    levels_E = int(np.unique(np.round(points["lE"].to_numpy(dtype=float), 9)).size)
    levels_A = int(np.unique(np.round(points["lA"].to_numpy(dtype=float), 9)).size)
    fit_n = levels_E >= 3
    fit_p = levels_A >= 3

    y = points["log_d"].to_numpy(dtype=float).copy()
    w = points["n_rep"].to_numpy(dtype=float)
    slope_status = {"n": "fitted" if fit_n else "assumed",
                    "p_eff": "fitted" if fit_p else "assumed"}
    if not fit_n:
        y = y - n_prior * points["lE"].to_numpy(dtype=float)
    if not fit_p:
        y = y - p_prior * points["lA"].to_numpy(dtype=float)

    if pooled_slopes:
        X, names = _design(points, metals, pubs_multi, fit_n, fit_p)
    else:
        X, names = _design(points, metals, pubs_multi, False, False)
        extra_cols, extra_names = [], []
        metal_arr = points["metal"].to_numpy()
        for m in metals:
            ind = (metal_arr == m).astype(float)
            if fit_n:
                extra_cols.append(ind * points["lE"].to_numpy(dtype=float))
                extra_names.append(f"n.{m}")
            if fit_p:
                extra_cols.append(ind * points["lA"].to_numpy(dtype=float))
                extra_names.append(f"p_eff.{m}")
        if extra_cols:
            X = np.column_stack([X] + extra_cols)
            names = names + extra_names

    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    N, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    dof = N - k
    s2 = float(np.sum(w * resid ** 2) / dof) if dof > 0 else float("nan")
    XtWX = Xw.T @ Xw
    if rank == k and k > 0:
        XtWX_inv = np.linalg.inv(XtWX)
    else:
        XtWX_inv = np.linalg.pinv(XtWX)
    cov = XtWX_inv * s2 if k > 0 else np.zeros((0, 0))
    meat = (Xw * (resid * sw)[:, None]).T @ (Xw * (resid * sw)[:, None])
    hc1_scale = (N / dof) if dof > 0 else float("nan")
    cov_hc1 = XtWX_inv @ meat @ XtWX_inv * hc1_scale if k > 0 else np.zeros((0, 0))
    se = {nm: float(math.sqrt(max(cov[i, i], 0.0))) if np.isfinite(cov[i, i]) else float("nan")
          for i, nm in enumerate(names)}
    se_hc1 = {nm: float(math.sqrt(max(cov_hc1[i, i], 0.0))) if np.isfinite(cov_hc1[i, i])
              else float("nan") for i, nm in enumerate(names)}
    coef = dict(zip(names, beta.tolist()))

    intercepts = {m: float(coef[f"a.{m}"]) for m in metals}
    if pooled_slopes:
        n_val = float(coef["n"]) if fit_n else float(n_prior)
        p_val = float(coef["p_eff"]) if fit_p else float(p_prior)
    else:
        n_vals = [coef[f"n.{m}"] for m in metals if f"n.{m}" in coef]
        p_vals = [coef[f"p_eff.{m}"] for m in metals if f"p_eff.{m}" in coef]
        n_val = float(np.mean(n_vals)) if n_vals else float(n_prior)
        p_val = float(np.mean(p_vals)) if p_vals else float(p_prior)
    effects: dict[str, float] = {p: 0.0 for p in pubs_all}
    if len(pubs_multi) >= 2:
        free = [f"pub.{p}" for p in pubs_multi[:-1]]
        for p, nm in zip(pubs_multi[:-1], free):
            effects[p] = float(coef[nm])
        effects[pubs_multi[-1]] = float(-sum(coef[nm] for nm in free))
        idx = [names.index(nm) for nm in free]
        one = np.ones(len(idx))
        var_last = float(one @ cov[np.ix_(idx, idx)] @ one)
        var_last_hc1 = float(one @ cov_hc1[np.ix_(idx, idx)] @ one)
        se[f"pub.{pubs_multi[-1]}"] = math.sqrt(max(var_last, 0.0)) if np.isfinite(var_last) \
            else float("nan")
        se_hc1[f"pub.{pubs_multi[-1]}"] = math.sqrt(max(var_last_hc1, 0.0)) \
            if np.isfinite(var_last_hc1) else float("nan")
    if not fit_n:
        se["n"] = 0.0
        se_hc1["n"] = 0.0
    if not fit_p:
        se["p_eff"] = 0.0
        se_hc1["p_eff"] = 0.0

    record_ids_by_metal = {
        m: tuple(sorted(rid for ids in points.loc[points["metal"] == m, "record_ids"]
                        for rid in ids)) for m in metals}

    domain = None
    try:  # WB2's domain builder; optional at this layer
        from gen18proc.domain import build_domain  # type: ignore
        domain = build_domain(prepared, ligand)
    except Exception:  # noqa: BLE001  -- absent module or an incompatible frame: box stays None
        domain = None

    result = FitResult(
        system_id, ligand, band, intercepts, n_val, p_val, effects, cov, se, None, None,
        {"n": False, "p_eff": False}, int(N), int(len(pubs_all)), domain, fit_manifest_sha256,
        slope_status, frozenset(flags), se_hc1, cov_hc1, tuple(names),
        float(math.sqrt(s2)) if np.isfinite(s2) else float("nan"),
        {"lE": levels_E, "lA": levels_A}, record_ids_by_metal, mech.value, "fitted",
        int(len(prepared)), priors)
    if not reliability:
        return result

    fit_kwargs = dict(mechanism=mech, ligand=ligand, band=band, pooled_slopes=pooled_slopes,
                      n_prior=n_prior, p_prior=p_prior)
    jk = jackknife_by_publication(records, **fit_kwargs)
    sh = split_half_by_publication(records, repeats=split_half_repeats, seed=seed, **fit_kwargs)
    interpretable = {}
    for key in ("n", "p_eff"):
        ok = slope_status[key] == "fitted" and jk is not None and np.isfinite(jk[key][1]) \
            and jk[key][1] < 0.5
        if ok and sh is not None:
            ok = sh[f"{key}_sign_agreement"] >= math.ceil(0.9 * sh["repeats"])
        interpretable[key] = bool(ok)
    return FitResult(
        system_id, ligand, band, intercepts, n_val, p_val, effects, cov, se, jk, sh,
        interpretable, int(N), int(len(pubs_all)), domain, fit_manifest_sha256, slope_status,
        frozenset(flags), se_hc1, cov_hc1, tuple(names),
        float(math.sqrt(s2)) if np.isfinite(s2) else float("nan"),
        {"lE": levels_E, "lA": levels_A}, record_ids_by_metal, mech.value, "fitted",
        int(len(prepared)), priors)


# =============================================================================================
# 10.3  Reliability
# =============================================================================================

def _slopes_of(fit: FitResult) -> tuple[float, float]:
    return fit.n, fit.p_eff


def jackknife_by_publication(records: pd.DataFrame, *, ligand: str,
                             mechanism: Mechanism | str = Mechanism.SOLVATING,
                             band: str | None = DEFAULT_BAND, pooled_slopes: bool = True,
                             n_prior: float = 3.0, p_prior: float = 2.0,
                             ) -> dict[str, tuple[float, float]] | None:
    """Leave-one-publication-out jackknife of ``n`` and ``p_eff`` (DESIGN.md section 10.3).

    Returns ``{"n": (estimate, se), "p_eff": (estimate, se)}`` where the estimate is the full
    fit's value and ``se = sqrt((P - 1) / P * sum (theta_(-i) - mean)^2)``; None when fewer than
    2 publications are available.  A slope fixed at its prior in the full fit reports SE 0 and is
    never ``interpretable`` (the SE then measures nothing).
    """
    kw = dict(mechanism=mechanism, ligand=ligand, band=band, pooled_slopes=pooled_slopes,
              n_prior=n_prior, p_prior=p_prior, reliability=False)
    full = fit_mass_action(records, **kw)
    if full.status != "fitted":
        return None
    prepared = prepare_records(records, ligand=ligand, band=band, mechanism=mechanism)
    pubs = sorted(set(prepared["publication_id"].tolist()))
    if len(pubs) < 2:
        return None
    est = []
    for p in pubs:
        sub = records.loc[records["publication_id"].astype(str) != p]
        f = fit_mass_action(sub, **kw)
        est.append(_slopes_of(f) if f.status == "fitted" else (float("nan"), float("nan")))
    arr = np.asarray(est, dtype=float)
    P = arr.shape[0]
    out: dict[str, tuple[float, float]] = {}
    for j, key in enumerate(("n", "p_eff")):
        col = arr[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2:
            se = float("nan")
        else:
            se = float(math.sqrt((col.size - 1) / col.size * np.sum((col - col.mean()) ** 2)))
        out[key] = (float(getattr(full, key)), se if P >= 2 else float("nan"))
    return out


def split_half_by_publication(records: pd.DataFrame, repeats: int = 20, seed: int = 18, *,
                              ligand: str, mechanism: Mechanism | str = Mechanism.SOLVATING,
                              band: str | None = DEFAULT_BAND, pooled_slopes: bool = True,
                              n_prior: float = 3.0, p_prior: float = 2.0,
                              ) -> dict[str, Any] | None:
    """Split-half **by publication** (DESIGN.md section 10.3; never within a publication).

    For systems with >= 4 publications: ``repeats`` seeded random halves of the publication set;
    per repeat the sign agreement of ``n`` and of ``p_eff`` between the two half fits and the
    Pearson r of the metal-intercept vectors over the metals both halves fitted (NaN with < 3
    common metals).  Returns ``{"repeats", "n_sign_agreement", "p_eff_sign_agreement",
    "intercept_r_median", "intercept_r_mean", "n_repeats_both_slopes_fitted", "per_repeat"}``
    (``per_repeat`` a DataFrame); None for systems with 2-3 publications (undefined, not passed).
    """
    prepared = prepare_records(records, ligand=ligand, band=band, mechanism=mechanism)
    pubs = sorted(set(prepared["publication_id"].tolist()))
    if len(pubs) < 4:
        return None
    kw = dict(mechanism=mechanism, ligand=ligand, band=band, pooled_slopes=pooled_slopes,
              n_prior=n_prior, p_prior=p_prior, reliability=False)
    rng = np.random.default_rng(seed)
    pub_col = records["publication_id"].astype(str)
    rows = []
    for r in range(int(repeats)):
        perm = rng.permutation(len(pubs))
        half = len(pubs) // 2
        a = sorted(pubs[i] for i in perm[:half])
        b = sorted(pubs[i] for i in perm[half:])
        fa = fit_mass_action(records.loc[pub_col.isin(a)], **kw)
        fb = fit_mass_action(records.loc[pub_col.isin(b)], **kw)
        common = sorted(set(fa.intercepts) & set(fb.intercepts))
        if len(common) >= 3:
            va = np.array([fa.intercepts[m] for m in common])
            vb = np.array([fb.intercepts[m] for m in common])
            if va.std() > 0 and vb.std() > 0:
                r_int = float(np.corrcoef(va, vb)[0, 1])
            else:
                r_int = float("nan")
        else:
            r_int = float("nan")
        rows.append({
            "repeat": r, "pubs_a": ",".join(a), "pubs_b": ",".join(b),
            "n_a": fa.n, "n_b": fb.n, "p_eff_a": fa.p_eff, "p_eff_b": fb.p_eff,
            "n_agree": bool(np.sign(fa.n) == np.sign(fb.n)),
            "p_eff_agree": bool(np.sign(fa.p_eff) == np.sign(fb.p_eff)),
            "n_fitted_both": fa.slope_status["n"] == "fitted" and fb.slope_status["n"] == "fitted",
            "p_eff_fitted_both": (fa.slope_status["p_eff"] == "fitted"
                                  and fb.slope_status["p_eff"] == "fitted"),
            "n_common_metals": len(common), "intercept_r": r_int,
        })
    per = pd.DataFrame(rows)
    return {
        "repeats": int(repeats),
        "n_publications": len(pubs),
        "n_sign_agreement": int(per["n_agree"].sum()),
        "p_eff_sign_agreement": int(per["p_eff_agree"].sum()),
        "intercept_r_median": float(np.nanmedian(per["intercept_r"].to_numpy(dtype=float)))
        if per["intercept_r"].notna().any() else float("nan"),
        "intercept_r_mean": float(np.nanmean(per["intercept_r"].to_numpy(dtype=float)))
        if per["intercept_r"].notna().any() else float("nan"),
        "n_repeats_both_slopes_fitted": int(
            (per["n_fitted_both"] & per["p_eff_fitted_both"]).sum()),
        "per_repeat": per,
    }


# =============================================================================================
# 10.2  Baselines and LOPO
# =============================================================================================

def _nearest_index(test_lA: float, test_lE: float, train: pd.DataFrame) -> int:
    """Position (in ``train``) of the 1-NN in (lA, lE), Euclidean; ties by smaller |delta lA|,
    then smaller ``record_id`` (DESIGN.md section 10.2)."""
    dA = train["lA"].to_numpy(dtype=float) - test_lA
    dE = train["lE"].to_numpy(dtype=float) - test_lE
    dist = np.sqrt(dA ** 2 + dE ** 2)
    rid_rank = np.argsort(np.argsort(train["record_id"].to_numpy().astype(str), kind="stable"),
                          kind="stable")
    order = np.lexsort((rid_rank, np.abs(dA), dist))
    return int(order[0])


def _score_holdout(train: pd.DataFrame, test: pd.DataFrame, fit: FitResult,
                   system_id: str, holdout_pub: str, holdout_label: str) -> list[dict[str, Any]]:
    """Rows of the LOPO table for one held-out publication."""
    rows = []
    train_metals = set(fit.intercepts)
    scorable = test.loc[test["metal"].isin(train_metals)]
    offset_delta = float("nan")
    calib_rid = None
    if len(scorable) >= 1:
        calib = scorable.sort_values("record_id", kind="stable").iloc[0]
        pred0 = float(fit.predict(calib["metal"], calib["lE"], calib["lA"]))
        offset_delta = float(calib["log_d"]) - pred0
        calib_rid = str(calib["record_id"])
    for metal, g in test.groupby("metal", sort=True):
        base = {"system_id": system_id, "holdout": holdout_label,
                "publication_id": holdout_pub, "metal": metal, "n_points": int(len(g)),
                "n_rows": int(g["n_rep"].sum())}
        if metal not in train_metals:
            rows.append({**base, "status": "metal_absent_from_training",
                         **{c: float("nan") for c in MAE_COLUMNS}, "n_train_same_metal": 0})
            continue
        tr_m = train.loc[train["metal"] == metal].reset_index(drop=True)
        obs = g["log_d"].to_numpy(dtype=float)
        pred_m1 = np.asarray(fit.predict(metal, g["lE"].to_numpy(), g["lA"].to_numpy()),
                             dtype=float)
        b0 = float(tr_m["log_d"].mean())
        pred_b1 = np.array([tr_m["log_d"].iloc[_nearest_index(a, e, tr_m)]
                            for a, e in zip(g["lA"], g["lE"])], dtype=float)
        tr_all = train.reset_index(drop=True)
        pred_b1x = np.array([tr_all["log_d"].iloc[_nearest_index(a, e, tr_all)]
                             for a, e in zip(g["lA"], g["lE"])], dtype=float)
        rest = g.loc[g["record_id"].astype(str) != calib_rid] if calib_rid is not None else g
        if len(rest) >= 1 and np.isfinite(offset_delta):
            pred_off = np.asarray(fit.predict(metal, rest["lE"].to_numpy(),
                                              rest["lA"].to_numpy()), dtype=float) + offset_delta
            mae_off = float(np.mean(np.abs(pred_off - rest["log_d"].to_numpy(dtype=float))))
        else:
            mae_off = float("nan")
        rows.append({
            **base, "status": "scored",
            "mae_M1": float(np.mean(np.abs(pred_m1 - obs))),
            "mae_B0": float(np.mean(np.abs(b0 - obs))),
            "mae_B1": float(np.mean(np.abs(pred_b1 - obs))),
            "mae_M1_offset": mae_off,
            "mae_B1_crossmetal": float(np.mean(np.abs(pred_b1x - obs))),
            "n_train_same_metal": int(len(tr_m)),
            "n_offset_points": int(len(rest)) if np.isfinite(offset_delta) else 0,
            "slope_status_n": fit.slope_status["n"],
            "slope_status_p": fit.slope_status["p_eff"],
        })
    return rows


def _points_to_records(points: pd.DataFrame, ligand: str) -> pd.DataFrame:
    """Aggregated points back into the flat record layout (so ``fit_mass_action`` can be reused
    on them without re-aggregating anything: every point is its own replicate group carrying
    its weight in ``n_rep``)."""
    return pd.DataFrame({
        "record_id": points["record_id"].to_numpy(),
        "system_id": points["system_id"].to_numpy(),
        "metal": points["metal"].to_numpy(),
        "log_d": points["log_d"].to_numpy(dtype=float),
        "d": np.power(10.0, points["log_d"].to_numpy(dtype=float)),
        "acid_nominal_M": np.power(10.0, points["lA"].to_numpy(dtype=float)),
        f"ligand_M.{ligand}": np.power(10.0, points["lE"].to_numpy(dtype=float)),
        "publication_id": points["publication_id"].to_numpy(),
        "replicate_group": points["record_id"].to_numpy(),
        "n_rep_weight": points["n_rep"].to_numpy(dtype=int),
    })


def _fit_on_points(points: pd.DataFrame, ligand: str, **kw: Any) -> FitResult:
    """Fit M1 on aggregated points, honouring their replicate weights."""
    recs = _points_to_records(points, ligand)
    return fit_mass_action(recs, ligand=ligand, band=None, reliability=False, **kw)


def lopo_evaluate(records: pd.DataFrame, *, mechanism: Mechanism | str = Mechanism.SOLVATING,
                  ligand: str, band: str | None = DEFAULT_BAND, seed: int = 18,
                  pooled_slopes: bool = True, n_prior: float = 3.0, p_prior: float = 2.0,
                  ) -> pd.DataFrame:
    """Leave-one-publication-out table (DESIGN.md section 10.2; PRE_REGISTRATION.md section 4).

    One row per (system, held-out publication, metal) with ``mae_M1`` (publication effect 0),
    ``mae_B0``, ``mae_B1``, ``mae_M1_offset`` (one held-out point -- the smallest ``record_id``
    among the publication's scorable points -- sets the effect; scored on the rest; NaN when the
    publication has one point), ``mae_B1_crossmetal`` (exploratory; 1-NN over all metals, no
    intercept correction), ``n_points`` (aggregated), ``n_rows`` (raw), ``status``
    (``"scored"`` or ``"metal_absent_from_training"``, the latter with NaN errors), and
    ``holdout = "lopo"``.  Systems are handled one by one (``system_id`` column); the seed is
    accepted for interface symmetry (the procedure is deterministic).
    """
    mech = Mechanism(mechanism) if not isinstance(mechanism, Mechanism) else mechanism
    prepared = prepare_records(records, ligand=ligand, band=band, mechanism=mech)
    kw = dict(mechanism=mech, pooled_slopes=pooled_slopes, n_prior=n_prior, p_prior=p_prior)
    rows: list[dict[str, Any]] = []
    for system_id, sub in prepared.groupby("system_id", sort=True):
        points = aggregate_replicates(sub)
        pubs = sorted(set(points["publication_id"].tolist()))
        if len(pubs) < 2:
            continue
        for p in pubs:
            train = points.loc[points["publication_id"] != p].reset_index(drop=True)
            test = points.loc[points["publication_id"] == p].reset_index(drop=True)
            fit = _fit_on_points(train, ligand, **kw)
            if fit.status != "fitted":
                continue
            rows.extend(_score_holdout(train, test, fit, str(system_id), p, "lopo"))
    cols = ["system_id", "holdout", "publication_id", "metal", "status", *MAE_COLUMNS,
            "n_points", "n_rows", "n_train_same_metal", "n_offset_points", "slope_status_n",
            "slope_status_p"]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = pd.Series(dtype=float if c.startswith("mae") else object)
    return out[cols].sort_values(["system_id", "publication_id", "metal"],
                                 kind="stable").reset_index(drop=True)


def in_sample_evaluate(records: pd.DataFrame, *, mechanism: Mechanism | str = Mechanism.SOLVATING,
                       ligand: str, band: str | None = DEFAULT_BAND, seed: int = 18,
                       pooled_slopes: bool = True, n_prior: float = 3.0, p_prior: float = 2.0,
                       ) -> pd.DataFrame:
    """The labelled ``in_sample`` rows: fit on all publications, scored on each (PRE_REGISTRATION
    section 4).  M1 uses the fitted publication effect here (it is in-sample by definition); B0
    and B1 use all points including the scored one.  Same columns as ``lopo_evaluate``."""
    mech = Mechanism(mechanism) if not isinstance(mechanism, Mechanism) else mechanism
    prepared = prepare_records(records, ligand=ligand, band=band, mechanism=mech)
    kw = dict(mechanism=mech, pooled_slopes=pooled_slopes, n_prior=n_prior, p_prior=p_prior)
    rows: list[dict[str, Any]] = []
    for system_id, sub in prepared.groupby("system_id", sort=True):
        points = aggregate_replicates(sub)
        fit = _fit_on_points(points, ligand, **kw)
        if fit.status != "fitted":
            continue
        for p in sorted(set(points["publication_id"].tolist())):
            test = points.loc[points["publication_id"] == p].reset_index(drop=True)
            fit_p = dataclasses.replace(
                fit, intercepts={m: a + fit.publication_effects.get(p, 0.0)
                                 for m, a in fit.intercepts.items()})
            rows.extend(_score_holdout(points, test, fit_p, str(system_id), p, "in_sample"))
    out = pd.DataFrame(rows)
    cols = ["system_id", "holdout", "publication_id", "metal", "status", *MAE_COLUMNS,
            "n_points", "n_rows", "n_train_same_metal", "n_offset_points", "slope_status_n",
            "slope_status_p"]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.Series(dtype=float if c.startswith("mae") else object)
    return out[cols].sort_values(["system_id", "publication_id", "metal"],
                                 kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class MacroResult:
    """``macro_over_systems`` output: ``per_system`` (one row per system: mean of each error
    column over its scored (publication, metal) rows, ``n_pairs``, ``n_points``), ``macro``
    (mean of each column over systems), ``n_systems``, ``point_weighted`` (labelled alternative:
    every aggregated point weight 1)."""

    per_system: pd.DataFrame
    macro: pd.Series
    n_systems: int
    point_weighted: pd.Series


def macro_over_systems(table: pd.DataFrame, columns: Sequence[str] = MAE_COLUMNS) -> MacroResult:
    """E1 averaging (PRE_REGISTRATION section 5): mean over (publication, metal) rows within a
    system (equal weight), then macro over systems.  Rows with ``status != "scored"`` (or NaN in
    a column) are excluded column-wise."""
    cols = [c for c in columns if c in table.columns]
    scored = table.loc[table["status"] == "scored"] if "status" in table.columns else table
    per = scored.groupby("system_id", sort=True)[cols].mean()
    per["n_pairs"] = scored.groupby("system_id", sort=True).size()
    if "n_points" in scored.columns:
        per["n_points"] = scored.groupby("system_id", sort=True)["n_points"].sum()
    macro = per[cols].mean(axis=0) if len(per) else pd.Series({c: float("nan") for c in cols})
    if "n_points" in scored.columns and len(scored):
        wts = scored["n_points"].to_numpy(dtype=float)
        pw = {}
        for c in cols:
            v = scored[c].to_numpy(dtype=float)
            ok = np.isfinite(v)
            pw[c] = float(np.sum(v[ok] * wts[ok]) / np.sum(wts[ok])) if ok.any() else float("nan")
        point_weighted = pd.Series(pw)
    else:
        point_weighted = pd.Series({c: float("nan") for c in cols})
    return MacroResult(per, macro, int(len(per)), point_weighted)


def paired_system_bootstrap(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray,
                            n_boot: int = 2000, seed: int = 18, alpha: float = 0.05,
                            ) -> tuple[float, float, float]:
    """Percentile bootstrap of ``mean(a - b)`` over paired units (systems or series) resampled
    with replacement (PRE_REGISTRATION section 5 (ii) and section 7 (ii)).

    Returns ``(mean, lo, hi)`` with the ``1 - alpha`` percentile interval; deterministic for a
    given ``seed``.  Pairs with a NaN in either vector are dropped; fewer than 2 pairs -> NaN
    interval.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have the same length")
    d = a - b
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(d.mean())
    if d.size < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lo), float(hi)


# =============================================================================================
# 10.5  ComparisonCounter
# =============================================================================================

_FAMILIES = ("primary", "secondary", "exploratory")


class ComparisonCounter:
    """Every statistical contrast of the generation, counted (DESIGN.md section 10.5;
    PRE_REGISTRATION section 9).  ``record(name, family, p_value)``; ``table()`` returns one row
    per contrast with Benjamini-Hochberg adjusted p-values over the **exploratory** family (raw
    and adjusted both reported; primary and secondary carry their raw value and ``p_adjusted =
    NaN``); ``write(path, regime=...)`` delegates to ``report.write_table``."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def record(self, name: str, family: str, p_value: float | None,
               note: str = "") -> None:
        if family not in _FAMILIES:
            raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")
        if any(r["name"] == name for r in self._rows):
            raise ValueError(f"contrast {name!r} already recorded")
        self._rows.append({"name": name, "family": family,
                           "p_value": float("nan") if p_value is None else float(p_value),
                           "note": note})

    @staticmethod
    def benjamini_hochberg(p: Sequence[float] | np.ndarray) -> np.ndarray:
        """BH step-up adjusted p-values (monotone, capped at 1); NaN stays NaN and does not
        count toward ``m``."""
        p = np.asarray(p, dtype=float)
        out = np.full(p.shape, np.nan)
        ok = np.isfinite(p)
        m = int(ok.sum())
        if m == 0:
            return out
        vals = p[ok]
        order = np.argsort(vals, kind="stable")
        ranked = vals[order] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(ranked[::-1])[::-1]
        adj = np.minimum(adj, 1.0)
        res = np.empty(m)
        res[order] = adj
        out[ok] = res
        return out

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=["name", "family", "p_value", "note"])
        df["p_adjusted"] = np.nan
        df["n_in_family"] = df.groupby("family")["name"].transform("size") if len(df) else 0
        expl = df["family"] == "exploratory"
        if expl.any():
            df.loc[expl, "p_adjusted"] = self.benjamini_hochberg(df.loc[expl, "p_value"])
        df["adjustment"] = np.where(expl, "benjamini_hochberg_exploratory_family", "none")
        return df[["name", "family", "p_value", "p_adjusted", "adjustment", "n_in_family",
                   "note"]]

    def counts(self) -> dict[str, int]:
        t = self.table()
        return {f: int((t["family"] == f).sum()) for f in _FAMILIES} | {"total": int(len(t))}

    def write(self, path: str, *, regime: Mapping[str, Any]) -> str:
        from gen18proc.report import write_table
        return write_table(self.table(), path, regime=dict(regime))


# =============================================================================================
# Fitted parameter blocks (DESIGN.md section 3.8 shape)
# =============================================================================================

def _model_id(fit: FitResult) -> str:
    return f"M1_pooled_{fit.system_id}_{fit.band or 'all'}"


def _fitted(value: float, unit: str, rng: tuple[float, float], fit: FitResult, locator: str,
            note: str, safe_exp_ids: tuple[str, ...] = ()) -> Sourced:
    return Sourced(value, unit, Provenance(
        ProvStatus.FITTED_FROM_CORPUS,
        Source("model", None, locator, None, safe_exp_ids),
        (float(rng[0]), float(rng[1])), None, _model_id(fit), fit.fit_manifest_sha256, note))


def _assumed(value: float, unit: str, rng: tuple[float, float], locator: str, note: str,
             ) -> Sourced:
    return Sourced(value, unit, Provenance(
        ProvStatus.ASSUMED, Source("none", None, locator), (float(rng[0]), float(rng[1])),
        ASSUMED_LABEL, None, None, note))


def _slope_sourced(fit: FitResult, key: str, unit: str = "1") -> Sourced:
    value = getattr(fit, key)
    if fit.slope_status.get(key) != "fitted":
        prior = fit.priors.get(key, value)
        return _assumed(float(prior), unit, (prior, prior),
                        "mechanism prior: fewer than 3 distinct levels on this axis",
                        f"slope {key} fixed at the pre-specified prior (PRE_REGISTRATION 3)")
    jk = fit.jackknife_se.get(key) if fit.jackknife_se else None
    se = jk[1] if jk is not None and np.isfinite(jk[1]) else fit.se.get(key, float("nan"))
    half = 2.0 * se if np.isfinite(se) else 0.0
    note = ("range = estimate +/- 2 jackknife SE; reliability in results/dmodels"
            if jk is not None and np.isfinite(jk[1]) else
            "range = estimate +/- 2 classical SE (jackknife not computed)")
    return _fitted(float(value), unit, (value - half, value + half), fit,
                   "M1 pooled-slope fit, publication effects summed to zero", note)


def _logk_block(fit: FitResult, note: str) -> dict[str, Sourced]:
    out = {}
    for m, a in fit.intercepts.items():
        se = fit.se.get(f"a.{m}", float("nan"))
        half = 2.0 * se if np.isfinite(se) else 0.0
        out[m] = _fitted(float(a), "1", (a - half, a + half), fit,
                         "M1 pooled-slope fit, publication effects summed to zero", note,
                         fit.record_ids_by_metal.get(m, ()))
    return out


def as_solvating_params(fit: FitResult, *, medium_anion: str = "nitrate",
                        k_acid_uptake: Sourced | None = None) -> SolvatingParams:
    """The fitted solvating block of DESIGN.md section 3.8: ``log_k`` per metal (intercept at
    [L] = 1 M, [anion] = 1 M, tracer limit; range +/- 2 SE), ``n_solvation`` = ``n``,
    ``p_anion`` = ``p_eff`` (the corpus acid exponent lands on the anion axis;
    ``ANION_ACID_CONFOUNDED`` applies when [anion] != [H+]), ``p_h`` = 0 assumed,
    ``k_acid_uptake`` unknown unless supplied."""
    if fit.status != "fitted":
        raise ValueError("cannot build parameters from a fit with status " + fit.status)
    return SolvatingParams(
        medium_anion=medium_anion,
        temperature_band=fit.band or DEFAULT_BAND,
        log_k=_logk_block(fit, f"intercept at [L] = 1 M, [{medium_anion}] = 1 M; tracer limit"),
        n_solvation=_slope_sourced(fit, "n"),
        p_anion=_slope_sourced(fit, "p_eff"),
        p_h=_assumed(0.0, "1", (0.0, 0.0),
                     "assignment of the confounded exponent to the anion axis", ""),
        k_acid_uptake=k_acid_uptake or Sourced.unknown(
            "L2/mol2", note="HNO3 uptake not transcribed; ACID_UPTAKE_UNMODELLED above 1 M"),
        delta_h_kj_mol=None,
    )


def as_cation_exchange_params(fit: FitResult, *, medium_anion: str = "chloride",
                              ) -> CationExchangeParams:
    """The cation-exchange analogue: ``a_dimer = n`` (slope on the dimer concentration),
    ``b_proton = -p_eff`` (the acid slope enters with a minus sign, section 5.2 a), ``log_k`` on
    the dimer basis at [(HA)2] = 1 M and [H+] = 1 M."""
    if fit.status != "fitted":
        raise ValueError("cannot build parameters from a fit with status " + fit.status)
    b = _slope_sourced(fit, "p_eff")
    b_val = -float(b.value) if b.value is not None else None
    b_rng = (-b.range[1], -b.range[0]) if b.range else None
    b_proton = Sourced(b_val, "1", Provenance(b.status, b.source, b_rng,
                                              b.provenance.assumed_label, b.provenance.model_id,
                                              b.provenance.fit_manifest_sha256,
                                              "b_proton = -p_eff; " + b.provenance.note))
    return CationExchangeParams(
        medium_anion=medium_anion,
        temperature_band=fit.band or DEFAULT_BAND,
        log_k=_logk_block(fit, "intercept at [(HA)2] = 1 M, [H+] = 1 M; dimer basis"),
        a_dimer=_slope_sourced(fit, "n"),
        b_proton=b_proton,
        k_reported_as="log_k",
        beta_anion=None,
        saponification_degree_studied=None,
        delta_h_kj_mol=None,
    )


# =============================================================================================
# 10.4  Loading evaluation (WB4b; addendum WB4b)
# =============================================================================================

LOADING_ACTIVE_RANGE = 0.3
"""log D range above which a series is loading-active (PRE_REGISTRATION section 2)."""

_ANCHOR_TOL = 1e-10
"""The tracer log D must be reproduced to this absolute tolerance (DESIGN.md section 10.4)."""


def _loading_n_of(fit: Any, n_fallback: float) -> tuple[float, str]:
    """``(n, source)`` for the depletion term: a ``FitResult`` (or a mapping with ``n`` and
    ``slope_status``) whose slope was fitted, else the prior ``n_fallback`` (PRE_REGISTRATION
    section 3, C1: prior ``n0`` when the system has no fittable group)."""
    if fit is None:
        return float(n_fallback), "prior_n0"
    if isinstance(fit, Mapping):
        n = fit.get("n")
        raw = fit.get("slope_status")
    else:
        n = getattr(fit, "n", None)
        raw = getattr(fit, "slope_status", None)
    if raw is None:
        status = "fitted" if n is not None else None
    elif isinstance(raw, Mapping):
        status = raw.get("n", "fitted" if n is not None else None)
    else:
        raise ValueError(
            f"slope_status must be a mapping keyed by parameter name, got {type(raw).__name__}"
            f" ({raw!r}); use {{'n': 'fitted'}}")
    if n is None or not np.isfinite(float(n)) or status != "fitted":
        return float(n_fallback), "prior_n0"
    return float(n), "m1_in_sample"


def _loading_series_model(metal: str, ligand: str, n: float, log_k: float, anion: str,
                          phi: float | None) -> Any:
    """A single-metal solvating ``SystemModel`` for C1: ``log D = log K + n log L_f``
    (``p_anion = p_h = 0``: the series is at fixed acid, so the acid term is inside ``log K``),
    ideal stoichiometry ``q = n``, ``p = 0``, ``z = 3``, ``K_H`` unknown (uptake unmodelled,
    flagged above 1 M), ``EffectiveCapacity(phi)`` as the activity when ``phi < 1``."""
    from gen18proc.dmodel import ASSUMPTIONS, EffectiveCapacity, SolvatingMassAction, SystemModel

    def sv(value: float, note: str) -> Sourced:
        return Sourced(float(value), "1", Provenance(
            ProvStatus.ASSUMED, Source(kind="none", locator=note), (float(value), float(value)),
            ASSUMED_LABEL, note=note))

    params = SolvatingParams(
        medium_anion=anion, temperature_band=DEFAULT_BAND,
        log_k={metal: sv(log_k, "anchored at the tracer point of the loading series (C1)")},
        n_solvation=sv(n, "depletion exponent of C1"),
        p_anion=sv(0.0, "fixed acid within a series: the acid term sits inside log K"),
        p_h=sv(0.0, "fixed acid within a series"),
        k_acid_uptake=Sourced.unknown("L2/mol2", note="HNO3 uptake unmodelled (C1)"),
        delta_h_kj_mol=None)
    model = SolvatingMassAction(params, ligand, None, None)
    activity = EffectiveCapacity(phi) if (phi is not None and phi < 1.0) else None
    return SystemModel(entry=None, dmodels={ligand: model}, complexant=None, activity=activity,
                       temperature_C=25.0, assumptions=ASSUMPTIONS,
                       params_source="anchored_tracer")


def _loading_stage(system: Any, metal: str, ligand: str, mm: float, ligand_total: float,
                   acid: float, oa: float) -> tuple[float, str, frozenset[Flag], float]:
    """One single-metal stage solve at initial aqueous ``mm`` (mM), ``L_T = ligand_total``,
    ``h = nu = acid``, O/A ``oa``: ``(log10 D, status, flags, loading_fraction)``."""
    from gen18proc.equilibrium import solve_stage
    from gen18proc.types import AqStream, OrgStream

    aq = AqStream(1.0, {metal: float(mm) * 1e-3}, float(acid), float(acid), 0.0, 0.0)
    org = OrgStream(float(oa), {metal: 0.0}, {ligand: {metal: 0.0}},
                    {ligand: float(ligand_total)}, {ligand: float(ligand_total)},
                    {ligand: 0.0}, 0.0)
    _, _, diag = solve_stage(aq, org, system)
    d = float(diag.d.get(metal, math.nan))
    logd = math.log10(d) if d > 0 else -math.inf
    return logd, diag.status, diag.flags, float(diag.loading_fraction.get(ligand, math.nan))


def _loading_anchor(metal: str, ligand: str, n: float, anion: str, phi: float | None,
                    tracer_mm: float, ligand_total: float, acid: float, oa: float,
                    target_logd: float, tol: float = _ANCHOR_TOL,
                    ) -> tuple[float, str, float]:
    """1-D root in ``log K`` so that the stage solve at the tracer point reproduces
    ``target_logd``: ``(log_k, status, residual)`` with status ``anchored`` (|residual| <= tol),
    ``anchor_tolerance`` (root found, residual above tol) or ``tracer_beyond_cap`` (no log K
    reaches the tracer D: the tracer point already exceeds the fixed-stoichiometry capacity)."""
    from scipy.optimize import brentq

    cap = ligand_total * (phi if phi is not None else 1.0)

    def f(log_k: float) -> float:
        system = _loading_series_model(metal, ligand, n, log_k, anion, phi)
        logd, status, _, _ = _loading_stage(system, metal, ligand, tracer_mm, ligand_total,
                                            acid, oa)
        if status != "converged" or not math.isfinite(logd):
            return math.nan
        return logd - target_logd

    lo = target_logd - n * math.log10(cap) - 1e-9    # L_f <= cap  =>  log K >= this
    f_lo = f(lo)
    for _ in range(20):                               # roundoff guard: f(lo) must be < 0
        if np.isfinite(f_lo) and f_lo < 0.0:
            break
        lo -= 1e-6
        f_lo = f(lo)
    if not np.isfinite(f_lo):
        return math.nan, "anchor_failed", math.nan
    if abs(f_lo) <= tol:
        return lo, "anchored", f_lo
    hi, f_hi, step = lo, f_lo, 1.0
    for _ in range(40):
        hi = hi + step
        f_hi = f(hi)
        if not np.isfinite(f_hi):
            return math.nan, "anchor_failed", math.nan
        if f_hi > 0.0:
            break
        step *= 2.0
    else:
        return math.nan, "tracer_beyond_cap", f_hi
    try:
        root = float(brentq(f, lo, hi, xtol=1e-14, rtol=8.9e-16, maxiter=500))
    except (RuntimeError, ValueError):
        return math.nan, "anchor_failed", math.nan
    resid = f(root)
    return root, ("anchored" if abs(resid) <= tol else "anchor_tolerance"), resid


def _loading_predict(metal: str, ligand: str, n: float, anion: str, phi: float | None,
                     tracer_mm: float, ligand_total: float, acid: float, oa: float,
                     target_logd: float, mms: np.ndarray,
                     ) -> tuple[np.ndarray, list[str], list[frozenset[Flag]], np.ndarray,
                                float, str]:
    """Anchor then predict every ``mms`` (mM): ``(log D, statuses, flags, loading fractions,
    log K, anchor status)``; NaN predictions when the anchor fails."""
    log_k, status, _ = _loading_anchor(metal, ligand, n, anion, phi, tracer_mm, ligand_total,
                                       acid, oa, target_logd)
    preds = np.full(len(mms), np.nan)
    statuses = ["not_predicted"] * len(mms)
    flags: list[frozenset[Flag]] = [frozenset()] * len(mms)
    lams = np.full(len(mms), np.nan)
    if not np.isfinite(log_k):
        return preds, statuses, flags, lams, log_k, status
    system = _loading_series_model(metal, ligand, n, log_k, anion, phi)
    for i, mm in enumerate(mms):
        preds[i], statuses[i], flags[i], lams[i] = _loading_stage(system, metal, ligand, mm,
                                                                  ligand_total, acid, oa)
    return preds, statuses, flags, lams, log_k, status


def loading_series_evaluate(records: pd.DataFrame, fits: Mapping[str, Any], *,
                            oa: float = 1.0, n_fallback: float = 3.0, phi_arm: bool = True,
                            phi_min_points: int = 4, series_ids: Iterable[str] | None = None,
                            ) -> pd.DataFrame:
    """Loading endpoint E2 per series (DESIGN.md section 10.4; PRE_REGISTRATION section 7).

    ``records`` in the flat ``corpus_records.csv`` layout with ``loading_series_id`` set on the
    series rows (``is_tracer`` marks the tracer; else the smallest metal concentration, ties by
    the smaller ``record_id``); ``fits`` maps ``system_id`` to the system's in-sample
    ``FitResult`` (or a mapping with ``n`` and ``slope_status``) whose ``n`` is the depletion
    exponent of C1 -- the prior ``n_fallback`` when absent or when the slope was fixed at its
    prior.  Per series: ``log K`` is anchored so that ``equilibrium.solve_stage`` at the tracer
    point (one metal, initial aqueous ``metals_initial_mM * 1e-3``, ``L_T = ligand_M``,
    ``h = nu = acid_nominal_M``, O/A ``oa``, ``K_H`` None) reproduces the tracer log D to 1e-10;
    every non-tracer point is predicted by ``solve_stage`` (C1); C0 keeps the tracer value.
    Columns: ``mae_constant`` (C0), ``mae_ideal`` (C1), ``c1_wins``, the descriptive
    ``spearman_logD_vs_logmM``, the anchor status and log K, ``n_used`` / ``n_source``, the
    union of stage flags, ``max_loading_fraction_pred``; the exploratory
    ``EffectiveCapacity(phi)`` arm (X1) for series with >= ``phi_min_points`` non-tracer points:
    leave-one-point-out ``phi`` in [0.01, 1] fitted on the rest (``minimize_scalar(bounded)``
    on the MAE, ``log K`` re-anchored for every ``phi``) -> ``mae_phi_loo``, ``phi_median``,
    plus the in-sample ``phi_insample`` / ``mae_phi_insample``.  Every series counts as it falls:
    nothing is excluded or re-weighted here.  ``df.attrs["points"]`` holds the per-point rows
    (observed, C0, C1, flags) as a list of records (``pd.DataFrame(df.attrs["points"])``).
    Series with fewer than 2 usable points or a failed anchor are kept with ``status`` saying
    why and NaN errors.  Malformed input (missing columns) raises
    ``ValueError``.
    """
    from scipy.optimize import minimize_scalar
    from scipy.stats import spearmanr

    _require_columns(records, ("record_id", "metal", "log_d", "acid_nominal_M",
                               "loading_series_id"))
    df = records.copy()
    df["loading_series_id"] = df["loading_series_id"].astype("string")
    df = df.loc[df["loading_series_id"].notna() & (df["loading_series_id"] != "")].copy()
    if series_ids is not None:
        wanted = set(str(s) for s in series_ids)
        df = df.loc[df["loading_series_id"].isin(wanted)].copy()
    if "system_id" not in df.columns:
        df["system_id"] = "unspecified"
    df["record_id"] = df["record_id"].astype(str)
    df["metal"] = df["metal"].astype(str)
    df["log_d"] = pd.to_numeric(df["log_d"], errors="coerce").astype(float)
    df["_acid"] = pd.to_numeric(df["acid_nominal_M"], errors="coerce").astype(float)
    rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for sid, sub in df.groupby("loading_series_id", sort=True):
        sid = str(sid)
        system_id = str(sub["system_id"].iloc[0])
        metal = str(sub["metal"].iloc[0])
        fit = fits.get(system_id)
        if "ligand_name" in sub.columns and sub["ligand_name"].notna().any():
            ligand = str(sub["ligand_name"].dropna().iloc[0])
        elif fit is not None and getattr(fit, "ligand", None):
            ligand = str(fit.ligand)
        else:
            raise ValueError(f"series {sid}: no ligand_name column and no fit to name the ligand")
        anion = str(sub["anion"].dropna().iloc[0]) if "anion" in sub.columns and \
            sub["anion"].notna().any() else "nitrate"
        pub = str(sub["publication_id"].iloc[0]) if "publication_id" in sub.columns else ""
        lig = ligand_column(sub, ligand).to_numpy(dtype=float)
        mm = metals_initial_column(sub, metal).to_numpy(dtype=float)
        acid = sub["_acid"].to_numpy(dtype=float)
        logd = sub["log_d"].to_numpy(dtype=float)
        keep = np.isfinite(lig) & (lig > 0) & np.isfinite(mm) & (mm > 0) & np.isfinite(acid) \
            & (acid > 0) & np.isfinite(logd)
        if "fit_eligible" in sub.columns:
            keep &= _as_bool(sub["fit_eligible"], True).to_numpy()
        s = sub.loc[keep].copy()
        lig, mm, acid, logd = lig[keep], mm[keep], acid[keep], logd[keep]
        rid = s["record_id"].to_numpy().astype(str)
        n_used, n_source = _loading_n_of(fit, n_fallback)
        base = {"loading_series_id": sid, "system_id": system_id, "ligand": ligand,
                "metal": metal, "publication_id": pub, "oa": float(oa), "n_points": int(len(s)),
                "n_used": n_used, "n_source": n_source}
        if len(s) < 2:
            rows.append({**base, "status": "insufficient_points"})
            continue
        if "is_tracer" in s.columns and _as_bool(s["is_tracer"], False).any():
            t_idx = int(np.flatnonzero(_as_bool(s["is_tracer"], False).to_numpy())[0])
        else:
            order = np.lexsort((rid, mm))
            t_idx = int(order[0])
        tracer_mm, tracer_logd, tracer_rid = float(mm[t_idx]), float(logd[t_idx]), rid[t_idx]
        ligand_total = float(np.median(lig))
        acid_m = float(np.median(acid))
        non = np.array([i for i in range(len(s)) if i != t_idx], dtype=int)
        obs = logd[non]
        rho, p_rho = (spearmanr(np.log10(mm), logd) if len(s) >= 3 else (math.nan, math.nan))
        base.update({"acid_nominal_M": acid_m, "ligand_M": ligand_total,
                     "n_non_tracer": int(len(non)), "tracer_record_id": tracer_rid,
                     "tracer_mM": tracer_mm, "log_d_tracer": tracer_logd,
                     "mM_min": float(mm.min()), "mM_max": float(mm.max()),
                     "log_d_range": float(logd.max() - logd.min()),
                     "loading_active": bool(logd.max() - logd.min() >= LOADING_ACTIVE_RANGE),
                     "spearman_logD_vs_logmM": float(rho), "spearman_p": float(p_rho)})
        preds, statuses, flags, lams, log_k, a_status = _loading_predict(
            metal, ligand, n_used, anion, None, tracer_mm, ligand_total, acid_m, oa,
            tracer_logd, mm[non])
        mae_c0 = float(np.mean(np.abs(tracer_logd - obs)))
        ok = np.isfinite(preds)
        mae_c1 = float(np.mean(np.abs(preds[ok] - obs[ok]))) if ok.any() else math.nan
        union: set[Flag] = set()
        for fl in flags:
            union |= set(fl)
        for i, j in enumerate(non):
            point_rows.append({
                "loading_series_id": sid, "system_id": system_id, "metal": metal,
                "record_id": rid[j], "metal_initial_mM": float(mm[j]), "log_d": float(obs[i]),
                "log_d_c0": tracer_logd, "log_d_c1": float(preds[i]),
                "abs_err_c0": float(abs(tracer_logd - obs[i])),
                "abs_err_c1": float(abs(preds[i] - obs[i])) if np.isfinite(preds[i]) else math.nan,
                "loading_fraction_pred": float(lams[i]), "stage_status": statuses[i],
                "flags": "|".join(sorted(f.value for f in flags[i])), "oa": float(oa)})
        base.update({"log_k_anchored": float(log_k), "anchor_status": a_status,
                     "mae_constant": mae_c0, "mae_ideal": mae_c1,
                     "c1_wins": bool(np.isfinite(mae_c1) and mae_c1 < mae_c0),
                     "delta_mae_c0_minus_c1": (mae_c0 - mae_c1) if np.isfinite(mae_c1)
                     else math.nan,
                     "n_predicted": int(ok.sum()),
                     "max_loading_fraction_pred": float(np.nanmax(lams)) if np.isfinite(lams).any()
                     else math.nan,
                     "flags": "|".join(sorted(f.value for f in union)),
                     "status": "scored" if np.isfinite(mae_c1) else a_status})
        # exploratory X1: effective capacity phi, leave-one-point-out
        phi_cols = {"phi_loo_n": 0, "phi_median": math.nan, "phi_min": math.nan,
                    "phi_max": math.nan, "mae_phi_loo": math.nan, "phi_insample": math.nan,
                    "mae_phi_insample": math.nan}
        if phi_arm and len(non) >= phi_min_points and np.isfinite(mae_c1):

            def fit_phi(idx: np.ndarray) -> float:
                def objective(phi: float) -> float:
                    p, _, _, _, _, _ = _loading_predict(metal, ligand, n_used, anion, float(phi),
                                                        tracer_mm, ligand_total, acid_m, oa,
                                                        tracer_logd, mm[non][idx])
                    good = np.isfinite(p)
                    if not good.any():
                        return 1e6
                    return float(np.mean(np.abs(p[good] - obs[idx][good]))) + (
                        0.0 if good.all() else 1e3)

                res = minimize_scalar(objective, bounds=(0.01, 1.0), method="bounded",
                                      options={"xatol": 1e-4, "maxiter": 60})
                return float(res.x)

            phis, errs = [], []
            for j in range(len(non)):
                rest = np.array([k for k in range(len(non)) if k != j], dtype=int)
                phi_j = fit_phi(rest)
                p, _, _, _, _, _ = _loading_predict(metal, ligand, n_used, anion, phi_j,
                                                    tracer_mm, ligand_total, acid_m, oa,
                                                    tracer_logd, mm[non][[j]])
                phis.append(phi_j)
                errs.append(abs(float(p[0]) - float(obs[j])) if np.isfinite(p[0]) else math.nan)
            phi_all = fit_phi(np.arange(len(non)))
            p_all, _, _, _, _, _ = _loading_predict(metal, ligand, n_used, anion, phi_all,
                                                    tracer_mm, ligand_total, acid_m, oa,
                                                    tracer_logd, mm[non])
            good = np.isfinite(p_all)
            phi_cols = {"phi_loo_n": int(len(phis)), "phi_median": float(np.median(phis)),
                        "phi_min": float(np.min(phis)), "phi_max": float(np.max(phis)),
                        "mae_phi_loo": float(np.nanmean(errs)) if np.isfinite(errs).any()
                        else math.nan,
                        "phi_insample": phi_all,
                        "mae_phi_insample": float(np.mean(np.abs(p_all[good] - obs[good])))
                        if good.any() else math.nan}
        base.update(phi_cols)
        rows.append(base)
    cols = ["loading_series_id", "system_id", "ligand", "metal", "publication_id",
            "acid_nominal_M", "ligand_M", "oa", "n_points", "n_non_tracer", "tracer_record_id",
            "tracer_mM", "log_d_tracer", "mM_min", "mM_max", "log_d_range", "loading_active",
            "n_used", "n_source", "log_k_anchored", "anchor_status", "mae_constant",
            "mae_ideal", "c1_wins", "delta_mae_c0_minus_c1", "n_predicted",
            "spearman_logD_vs_logmM", "spearman_p", "max_loading_fraction_pred", "flags",
            "status", "phi_loo_n", "phi_median", "phi_min", "phi_max", "mae_phi_loo",
            "phi_insample", "mae_phi_insample"]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols].sort_values("loading_series_id", kind="stable").reset_index(drop=True)
    points = pd.DataFrame(point_rows)
    if not points.empty:
        points = points.sort_values(["loading_series_id", "metal_initial_mM", "record_id"],
                                    kind="stable").reset_index(drop=True)
    # a list of records (not a DataFrame) so that pandas can compare / concatenate the attrs
    out.attrs["points"] = points.to_dict("records")
    return out
