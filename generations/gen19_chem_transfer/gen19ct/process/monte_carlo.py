"""``monte_carlo.py`` -- D-table draws through gen18's cascade, paired over one LHS design (brief section 17;
pre-registration section 14 "Monte Carlo" and "Outputs per operating point").

* **Draws.**  ``K`` joint draws (section 14: 64, seeded).  One draw is one D table.  **Registered mode** (the table
  carries the M7 member means ``member_logD_k`` and the calibrated conformal multiplier ``conformal_q95``,
  ``gen18_adapter.PredictionTable.has_members``): section 14's "one ensemble member plus a residual drawn from the
  calibrated conformal scale" -- one member ``k`` per draw (uniform over the members, one seeded stream, the SAME member
  for every metal because a member is one model that predicts both) plus, per metal, one residual ``s * z`` with
  ``s = std_logD * conformal_q95 / 1.96`` (the Gaussian whose 95 % half-width equals the calibrated conformal one,
  :data:`REGISTRATION_READINGS` ``draw_residual``) and ``z`` a standard normal correlated across metals with the section
  14 Pr/Nd residual correlation ``rho`` (Gaussian copula, exchangeable correlation matrix); within a metal the residual
  is common over the grid (one residual per metal and draw, not a jagged surface).  **Fallback** (no member columns;
  refused by a registered run, allowed with ``--exploratory`` and flagged): Gaussian ``N(mean_logD, std_logD)``
  truncated at the record's 95 % interval, one common quantile per metal.  ``std_logD = 0`` gives the mean (or the
  member); a missing interval gives the untruncated Gaussian (counted in ``draws.attrs``).  ``draws.attrs['draw_mode']``
  names the mode (task X finding V-07).
* **Paired design.**  gen18's ``optimize.lhs_pareto`` is run once per draw with the SAME LHS seed
  (``scripts/g18_case_prnd.py`` pattern), so candidate ``i`` is the same operating point in every draw and the per
  operating-point distributions are paired.  gen18's cascade, metrics and optimiser are imported read-only; the D
  source is :class:`gen19ct.process.gen18_adapter.Gen19DModel`.  The statuses of the grid cells each candidate's
  stages interpolated between are recorded per candidate (section 14 support rank input).
* **Aggregation** (:func:`aggregate_operating_points`): per operating point the median and 5 / 50 / 95 % purity and
  recovery over converged draws, ``P(purity >= t)``, ``P(recovery >= t)``, ``P(both)`` and ``P(both and feasible)`` per
  spec cell, ``P(feasible)`` and ``P(phase / loading constraints satisfied)`` -- a failed draw counts as a miss --
  reagent consumption (median ``consumption_index``, mol acid + base + complexant per kg oxide), stage count and
  throughput (median kg oxide / h).  Constraint flags: gen18's ``THIRD_PHASE_RISK`` (sourced LOC exceeded) and the
  loading flags ``LOADING_CAP_HIT`` / ``HIGH_LOADING`` (:data:`CONSTRAINT_FLAGS`); ``PHASE_BEHAVIOUR_UNKNOWN`` is
  reported separately (unknown, not violated).  Feasible = converged, constraints satisfied and regime not
  ``INADMISSIBLE``.
* **Determinism.**  Everything is a function of ``(seed, table, design space)``; the same seed reproduces the process
  table byte for byte (``tests/test_process.py``).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

from gen19ct import paths
from gen19ct.process import gen18_adapter as GA

paths.add_gen18_to_path()

from gen18proc import optimize as OPT  # noqa: E402  (read-only import)
from gen18proc.dmodel import SystemModel  # noqa: E402
from gen18proc.metrics import Prices  # noqa: E402
from gen18proc.types import AqStream, CascadeSpec, ModelBuildError, SystemEntry  # noqa: E402

__all__ = [
    "K_DRAWS_REGISTERED", "N_DESIGNS_DEFAULT", "LHS_SEED_DEFAULT", "TRUNCATION_LEVEL", "CONSTRAINT_FLAGS",
    "DRAW_MODE_REGISTERED", "DRAW_MODE_FALLBACK", "Z_975", "REGISTRATION_READINGS", "member_choices",
    "PHASE_UNKNOWN_FLAG", "PROCESS_COLUMNS", "CaseSetup", "MonteCarloResult", "exchangeable_correlation",
    "draw_quantiles", "truncated_normal_quantiles", "draw_tables", "base_spec_for", "run_monte_carlo",
    "cell_key", "aggregate_operating_points", "feasible_mask", "constraints_mask",
]

K_DRAWS_REGISTERED = 64          # section 14: 64 joint draws per operating point
DRAW_MODE_REGISTERED = "member_plus_conformal_residual"      # section 14 (needs member_logD_k + conformal_q95)
DRAW_MODE_FALLBACK = "truncated_gaussian_fallback"          # a reading; exploratory only
Z_975 = 1.959963984540054        # the standard-normal 97.5 % point: conformal 95 % half-width -> Gaussian scale
REGISTRATION_READINGS: dict[str, str] = {
    "draw_residual": "section 14 says 'a residual drawn from the calibrated conformal scale' without naming the residual "
                     "distribution; the residual is Gaussian with scale std_logD * conformal_q95 / 1.96, i.e. the "
                     "Gaussian whose central 95 % interval equals the calibrated normalised-conformal 95 % interval "
                     "(mean +- q95 std_logD) of the M7 record; the conformal distribution itself is not Gaussian and its "
                     "50 / 80 % quantiles are not used (INFERRED; needs a POST-HOC addendum)",
    "member_choice": "one M7 member per draw, uniform over the members from a second seeded stream "
                     "(default_rng(seed * 1000003 + 1)), shared by every metal of the draw because a member is one model "
                     "predicting both metals; the residual stream is default_rng(seed) as before (INFERRED)",
    "fallback": "a table without member_logD_k / conformal_q95 can only be drawn as a truncated Gaussian on the record "
                "(the pre-fix reading); a registered run refuses it and --exploratory flags it (task X finding V-07)",
}
N_DESIGNS_DEFAULT = 1000         # not registered: gen18's g18_case_prnd.py default (--n-lhs 1000)
LHS_SEED_DEFAULT = 18            # gen18's DEFAULT_SEED (every gen18 stochastic step)
TRUNCATION_LEVEL = 0.95          # truncate the Gaussian at the record's 95 % interval

#: gen18 flags whose presence violates the phase / loading constraints (section 14 "P(phase and loading constraints
#: satisfied), taken from gen18's THIRD_PHASE_RISK and loading flags")
CONSTRAINT_FLAGS: tuple[str, ...] = ("THIRD_PHASE_RISK", "LOADING_CAP_HIT", "HIGH_LOADING")
PHASE_UNKNOWN_FLAG = "PHASE_BEHAVIOUR_UNKNOWN"

#: columns of one draw's row kept in the process table (the LHS variables are added from the design space)
PROCESS_COLUMNS: tuple[str, ...] = (
    "draw", "candidate", "status", "regime_status", "in_domain", "on_spec", "purity_mol", "recovery_from_feed",
    "recovery_total", "net_product_mol_h", "n_stages_total", "n_ext", "n_scr", "n_str", "oa_ext", "s_over_a",
    "w_over_a", "consumption_index", "consumption_index_incomplete", "acid_mol_per_kg_oxide",
    "throughput_kg_oxide_per_h", "throughput_mol_T_per_h_per_L_org", "max_loading", "flags", "n_flags",
    "ood_distance_max", "balance_rel_max", "iterations", "front")


# --------------------------------------------------------------------------------------------- #
# the case
# --------------------------------------------------------------------------------------------- #

@dataclass
class CaseSetup:
    """Everything one Monte Carlo run needs besides the prediction table."""

    entry: SystemEntry
    feed: AqStream
    feed_anion: str
    temperature_C: float
    target: str
    impurities: tuple[str, ...]
    purity_grid: tuple[float, ...]
    recovery_grid: tuple[float, ...]
    space: OPT.DesignSpace
    base_spec: CascadeSpec
    prices: Prices | None = None
    n_prior: float = 3.0
    band: str = "20-30C"
    solver_kwargs: dict[str, Any] = field(default_factory=lambda: {"max_newton": 40, "max_sweeps": 10})

    @property
    def ligand(self) -> str:
        return GA._extractant_ligand(self.entry).name

    @property
    def loosest(self) -> dict[str, float]:
        return {"purity_min": min(self.purity_grid), "recovery_min": min(self.recovery_grid)}

    @property
    def cells(self) -> list[tuple[float, float]]:
        return [(float(p), float(r)) for p in self.purity_grid for r in self.recovery_grid]


def base_spec_for(feed: AqStream, target: str, ligand: str, ligand_total: float, *, scrub_acid_M: float = 3.0,
                  strip_acid_M: float = 0.01) -> CascadeSpec:
    """gen18's reference 6 / 3 / 3 cascade around ``feed`` (``scripts/g18_case_prnd.py::base_spec_for`` with the
    TODGA exploration's scrub / strip liquors: scrub 0.3 A at ``scrub_acid_M``, strip 0.5 A at ``strip_acid_M``).
    The LHS candidates overwrite every variable of the design space; the base supplies what they do not."""
    zeros = {m: 0.0 for m in feed.metals}
    scrub = AqStream(0.3 * feed.flow_L_h, dict(zeros), scrub_acid_M, scrub_acid_M, 0.0, 0.0)
    strip = AqStream(0.5 * feed.flow_L_h, dict(zeros), strip_acid_M, strip_acid_M, 0.0, 0.0)
    return CascadeSpec(6, 3, 3, feed, scrub, strip, feed.flow_L_h, {ligand: ligand_total}, 0.0, target=target)


# --------------------------------------------------------------------------------------------- #
# draws
# --------------------------------------------------------------------------------------------- #

def exchangeable_correlation(n: int, rho: float) -> np.ndarray:
    """``rho`` off the diagonal, 1 on it; must be positive definite (``rho > -1 / (n - 1)``)."""
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must lie in (-1, 1), got {rho}")
    if n > 1 and rho <= -1.0 / (n - 1):
        raise ValueError(f"rho = {rho} is not positive definite for {n} metals")
    C = np.full((n, n), float(rho))
    np.fill_diagonal(C, 1.0)
    return C


def draw_quantiles(n_draws: int, metals: Sequence[str], *, rho: float, seed: int) -> pd.DataFrame:
    """``draw x metal`` frame of correlated standard normals ``z`` and their uniforms ``u = Phi(z)`` (Gaussian
    copula, exchangeable correlation ``rho``; ``numpy.random.default_rng(seed)``)."""
    metals = list(metals)
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    rng = np.random.default_rng(int(seed))
    Z = rng.standard_normal((int(n_draws), len(metals)))
    if len(metals) > 1 and rho != 0.0:
        L = np.linalg.cholesky(exchangeable_correlation(len(metals), rho))
        Z = Z @ L.T
    U = norm.cdf(Z)
    rows = []
    for k in range(int(n_draws)):
        row: dict[str, Any] = {"draw": k}
        for m_i, m in enumerate(metals):
            row[f"z_{m}"] = float(Z[k, m_i])
            row[f"u_{m}"] = float(U[k, m_i])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs.update({"seed": int(seed), "rho": float(rho), "metals": metals})
    return df


def truncated_normal_quantiles(u: np.ndarray, mean: np.ndarray, std: np.ndarray, lower: np.ndarray,
                               upper: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantile ``u`` (per metal, broadcast over the grid) of ``N(mean, std)`` truncated to ``[lower, upper]``:
    ``std = 0`` -> ``mean``; a non-finite bound -> untruncated.  Returns ``(values, n_untruncated_cells)``."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    u = np.broadcast_to(np.asarray(u, dtype=float).reshape((-1,) + (1,) * (mean.ndim - 1)), mean.shape)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    out = np.array(mean, copy=True)
    pos = std > 0
    trunc = pos & np.isfinite(lower) & np.isfinite(upper)
    if trunc.any():
        a = (lower[trunc] - mean[trunc]) / std[trunc]
        b = (upper[trunc] - mean[trunc]) / std[trunc]
        out[trunc] = truncnorm.ppf(u[trunc], a, b, loc=mean[trunc], scale=std[trunc])
    plain = pos & ~trunc
    if plain.any():
        out[plain] = norm.ppf(u[plain], loc=mean[plain], scale=std[plain])
    return out, int(plain.sum())


def member_choices(n_draws: int, n_members: int, *, seed: int) -> np.ndarray:
    """One M7 member index per draw (uniform; :data:`REGISTRATION_READINGS` ``member_choice``)."""
    if n_members < 1:
        raise ValueError("n_members must be >= 1")
    rng = np.random.default_rng(int(seed) * 1000003 + 1)
    return rng.integers(0, int(n_members), size=int(n_draws))


def draw_tables(table: GA.PredictionTable, n_draws: int, *, seed: int, rho: float, allow_fallback: bool = True,
                ) -> tuple[list[np.ndarray], pd.DataFrame]:
    """``(values_per_draw, draws)``: one ``(M, A, L)`` log D array per draw and the draw frame of
    :func:`draw_quantiles` with the per-draw member index (registered mode) and the per-metal mean log D offset
    appended.  Registered mode when ``table.has_members`` (module docstring); otherwise the truncated-Gaussian fallback,
    refused with ``allow_fallback=False``."""
    draws = draw_quantiles(n_draws, table.metals, rho=rho, seed=seed)
    values: list[np.ndarray] = []
    n_plain = 0
    if table.has_members:
        ks = member_choices(len(draws), table.n_members, seed=seed)
        scale = table.std * table.conformal_q95 / Z_975                     # (M, A, L) Gaussian residual scale
        for k in range(len(draws)):
            z = np.array([draws.loc[k, f"z_{m}"] for m in table.metals]).reshape((-1,) + (1,) * (table.mean.ndim - 1))
            v = table.members[int(ks[k])] + scale * z
            values.append(v)
            draws.loc[k, "member"] = int(ks[k])
            for m_i, m in enumerate(table.metals):
                draws.loc[k, f"mean_offset_{m}"] = float(np.mean(v[m_i] - table.mean[m_i]))
        draws["member"] = draws["member"].astype(int)
        draws.attrs["draw_mode"] = DRAW_MODE_REGISTERED
        draws.attrs["n_members"] = int(table.n_members)
        draws.attrs["residual_scale"] = "std_logD * conformal_q95 / 1.96"
    else:
        if not allow_fallback:
            raise ValueError("the prediction table carries no M7 member columns (member_logD_k) / conformal_q95: section "
                             "14's draw (one ensemble member plus a conformal residual) is impossible and the "
                             "truncated-Gaussian fallback is exploratory only (task X finding V-07)")
        for k in range(len(draws)):
            u = np.array([draws.loc[k, f"u_{m}"] for m in table.metals])
            v, n = truncated_normal_quantiles(u, table.mean, table.std, table.lower, table.upper)
            n_plain += n
            values.append(v)
            for m_i, m in enumerate(table.metals):
                draws.loc[k, f"mean_offset_{m}"] = float(np.mean(v[m_i] - table.mean[m_i]))
        draws.attrs["draw_mode"] = DRAW_MODE_FALLBACK
    draws.attrs["n_untruncated_cells_total"] = n_plain
    draws.attrs["truncation_level"] = TRUNCATION_LEVEL
    draws.attrs["readings"] = dict(REGISTRATION_READINGS)
    return values, draws


# --------------------------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------------------------- #

@dataclass
class MonteCarloResult:
    process_table: pd.DataFrame          # one row per (candidate, draw)
    candidates: pd.DataFrame             # one row per candidate: the decoded design variables
    draws: pd.DataFrame                  # the draw frame
    usage: pd.DataFrame                  # candidate -> statuses used ('|'-joined), n_evaluations
    attrs: dict[str, Any] = field(default_factory=dict)

    def statuses_by_candidate(self) -> dict[int, frozenset[str]]:
        return {int(r.candidate): frozenset(s for s in str(r.statuses_used).split("|") if s)
                for r in self.usage.itertuples(index=False)}


def _build(table: GA.PredictionTable, setup: CaseSetup, values: np.ndarray | None, allow_unsupported: bool,
           ) -> SystemModel:
    sysm = GA.build_gen19_system(table, setup.entry, feed_anion=setup.feed_anion, temperature_C=setup.temperature_C,
                                 n_prior=setup.n_prior, band=setup.band, allow_unsupported=allow_unsupported,
                                 values=values, feed_metals=(setup.target, *setup.impurities))
    if isinstance(sysm, ModelBuildError):
        raise ValueError(f"Gen19 system could not be built: {sysm.reason}")
    return sysm


def run_monte_carlo(table: GA.PredictionTable, setup: CaseSetup, *, n_draws: int = K_DRAWS_REGISTERED,
                    n_designs: int = N_DESIGNS_DEFAULT, seed: int = LHS_SEED_DEFAULT, rho: float = 0.0,
                    allow_unsupported: bool = False, progress: Callable[[int, int, float], None] | None = None,
                    allow_fallback: bool = True) -> MonteCarloResult:
    """Section 14 Monte Carlo: ``n_draws`` D tables (:func:`draw_tables`, seed ``seed``), each through
    ``optimize.lhs_pareto`` on ``setup.space`` with the SAME LHS seed ``seed`` and ``n_designs`` rows (paired
    candidates).  ``progress(draw, n_draws, seconds)`` after every draw.  ``allow_unsupported`` is passed to the
    adapter (F5(ii) audit only)."""
    import time

    values, draws = draw_tables(table, n_draws, seed=seed, rho=rho, allow_fallback=allow_fallback)
    ligand = setup.ligand
    frames: list[pd.DataFrame] = []
    usage: dict[int, set[str]] = {}
    n_eval: dict[int, int] = {}
    candidates: pd.DataFrame | None = None
    n_failed = n_invalid = 0
    for k, v in enumerate(values):
        t0 = time.perf_counter()
        system = _build(table, setup, v, allow_unsupported)
        model = system.dmodels[ligand]
        model.reset_usage()

        def _hook(i_done: int, n_total: int) -> None:
            i = i_done - 1
            used = model.statuses_used()
            usage.setdefault(i, set()).update(used)
            n_eval[i] = n_eval.get(i, 0) + int(model.n_evaluations)
            model.reset_usage()

        df = OPT.lhs_pareto(setup.space, setup.base_spec, system, setup.target, setup.impurities, setup.loosest,
                            setup.prices, n=int(n_designs), seed=int(seed), solver_kwargs=setup.solver_kwargs,
                            progress=_hook)
        n_failed += int(df.attrs["n_failed"])
        n_invalid += int(df.attrs["n_invalid_spec"])
        df = df.sort_values("candidate", kind="mergesort").reset_index(drop=True)
        var_cols = [c for c in setup.space.variables] + [c for c in setup.space.fixed if c in df.columns]
        if candidates is None:
            candidates = df[["candidate", *var_cols]].copy()
        else:
            # the paired design: identical variables in every draw (the LHS seed is the same)
            same = np.allclose(df[var_cols].to_numpy(dtype=float), candidates[var_cols].to_numpy(dtype=float),
                               equal_nan=True)
            if not same:
                raise AssertionError("LHS design differs between draws: the pairing is broken")
        df.insert(0, "draw", int(k))
        lig_col = f"max_loading_{ligand}"
        df["max_loading"] = df[lig_col] if lig_col in df.columns else np.nan
        keep = list(dict.fromkeys([c for c in PROCESS_COLUMNS if c in df.columns] + var_cols))
        frames.append(df[keep])
        if progress is not None:
            progress(k + 1, len(values), time.perf_counter() - t0)
    table_df = pd.concat(frames, ignore_index=True).sort_values(["candidate", "draw"], kind="mergesort")
    table_df = table_df.reset_index(drop=True)
    use = pd.DataFrame({"candidate": sorted(usage),
                        "statuses_used": ["|".join(sorted(usage[i])) for i in sorted(usage)],
                        "n_evaluations": [n_eval[i] for i in sorted(usage)]})
    attrs = {"n_draws": int(n_draws), "n_designs": int(n_designs), "seed": int(seed), "rho": float(rho),
             "n_failed": n_failed, "n_invalid_spec": n_invalid, "allow_unsupported": bool(allow_unsupported),
             "ligand": ligand, "space": setup.space.to_json(), "table_statuses": table.statuses(),
             "table_box": table.box(), "n_untruncated_cells_total": draws.attrs.get("n_untruncated_cells_total"),
             "draw_mode": draws.attrs.get("draw_mode"), "n_members": draws.attrs.get("n_members", 0),
             "constraint_flags": list(CONSTRAINT_FLAGS)}
    return MonteCarloResult(process_table=table_df, candidates=candidates.reset_index(drop=True), draws=draws,
                            usage=use, attrs=attrs)


# --------------------------------------------------------------------------------------------- #
# aggregation per operating point
# --------------------------------------------------------------------------------------------- #

def cell_key(purity_min: float, recovery_min: float) -> str:
    return f"{float(purity_min):.2f}_{float(recovery_min):.2f}"


def _flags_of(series: pd.Series) -> list[set[str]]:
    return [set(str(s).split("|")) - {"", "nan"} for s in series.fillna("").astype(str)]


def constraints_mask(df: pd.DataFrame) -> np.ndarray:
    """Converged and none of :data:`CONSTRAINT_FLAGS` raised."""
    conv = df["status"].astype(str).str.startswith("converged").to_numpy()
    fl = _flags_of(df["flags"])
    ok = np.array([not (f & set(CONSTRAINT_FLAGS)) for f in fl], dtype=bool)
    return conv & ok


def feasible_mask(df: pd.DataFrame) -> np.ndarray:
    """:func:`constraints_mask` and regime not ``INADMISSIBLE``."""
    return constraints_mask(df) & (df["regime_status"].astype(str) != "INADMISSIBLE").to_numpy()


def _pct(values: np.ndarray, q: float) -> float:
    v = values[np.isfinite(values)]
    return float(np.percentile(v, q)) if v.size else math.nan


def aggregate_operating_points(process_table: pd.DataFrame, purity_grid: Sequence[float],
                               recovery_grid: Sequence[float], usage: pd.DataFrame | None = None,
                               variables: Iterable[str] = ()) -> pd.DataFrame:
    """Section 14 outputs per operating point (module docstring).  Probabilities are means over ALL draws of the
    candidate (a failed or invalid draw is a miss); percentiles and medians are over converged draws.  ``usage``
    (candidate -> ``statuses_used``) is merged when given."""
    df = process_table
    conv_all = df["status"].astype(str).str.startswith("converged").to_numpy()
    feas_all = feasible_mask(df)
    cons_all = constraints_mask(df)
    fl_all = _flags_of(df["flags"])
    unknown_all = np.array([PHASE_UNKNOWN_FLAG in f for f in fl_all], dtype=bool)
    third_all = np.array(["THIRD_PHASE_RISK" in f for f in fl_all], dtype=bool)
    pur_all = pd.to_numeric(df["purity_mol"], errors="coerce").to_numpy(dtype=float)
    rec_all = pd.to_numeric(df["recovery_from_feed"], errors="coerce").to_numpy(dtype=float)
    cons_idx_all = pd.to_numeric(df["consumption_index"], errors="coerce").to_numpy(dtype=float)
    thr_all = pd.to_numeric(df["throughput_kg_oxide_per_h"], errors="coerce").to_numpy(dtype=float)
    cand_all = df["candidate"].to_numpy()
    var_cols = [v for v in variables if v in df.columns]
    rows = []
    for cand in sorted(set(cand_all.tolist())):
        sel = cand_all == cand
        n = int(sel.sum())
        conv, feas, cons = conv_all[sel], feas_all[sel], cons_all[sel]
        pur = np.where(conv, pur_all[sel], np.nan)
        rec = np.where(conv, rec_all[sel], np.nan)
        first = df.loc[sel].iloc[0]
        row: dict[str, Any] = {"candidate": int(cand), "n_draws": n, "n_converged": int(conv.sum()),
                               "n_failed": int(n - conv.sum()),
                               "purity_p5": _pct(pur, 5), "purity_p50": _pct(pur, 50), "purity_p95": _pct(pur, 95),
                               "recovery_p5": _pct(rec, 5), "recovery_p50": _pct(rec, 50),
                               "recovery_p95": _pct(rec, 95),
                               "p_converged": float(conv.mean()), "p_constraints": float(cons.mean()),
                               "p_feasible": float(feas.mean()),
                               "p_phase_unknown": float(unknown_all[sel].mean()),
                               "p_third_phase_risk": float(third_all[sel].mean())}
        pur0 = np.nan_to_num(pur, nan=-np.inf)
        rec0 = np.nan_to_num(rec, nan=-np.inf)
        for pt in purity_grid:
            row[f"p_purity_ge_{float(pt):.2f}"] = float((conv & (pur0 >= pt)).mean())
        for rt in recovery_grid:
            row[f"p_recovery_ge_{float(rt):.2f}"] = float((conv & (rec0 >= rt)).mean())
        for pt in purity_grid:
            for rt in recovery_grid:
                both = conv & (pur0 >= pt) & (rec0 >= rt)
                row[f"p_both_{cell_key(pt, rt)}"] = float(both.mean())
                row[f"p_both_feasible_{cell_key(pt, rt)}"] = float((both & feas).mean())
        ci = np.where(conv, cons_idx_all[sel], np.nan)
        thr = np.where(conv, thr_all[sel], np.nan)
        row["consumption_index_median"] = _pct(ci, 50)
        row["throughput_median"] = _pct(thr, 50)
        row["n_stages_total"] = int(round(float(first["n_stages_total"])))
        for c in ("n_ext", "n_scr", "n_str", "oa_ext", "s_over_a", "w_over_a"):
            if c in df.columns:
                row[c] = float(first[c])
        for c in var_cols:
            if c not in row:
                row[c] = float(first[c])
        rows.append(row)
    out = pd.DataFrame(rows)
    if usage is not None:
        use = usage[["candidate", "statuses_used"]].copy() if len(usage) else pd.DataFrame(columns=["candidate", "statuses_used"])
        out = out.merge(use, on="candidate", how="left")
        conv = pd.to_numeric(out["n_converged"], errors="coerce").fillna(0).to_numpy() > 0
        st = out["statuses_used"]
        empty = st.isna() | (st.astype(str).str.strip() == "") | (st.astype(str).str.lower() == "nan")
        missing = sorted(int(c) for c in out.loc[conv & empty.to_numpy(), "candidate"])
        if missing:
            raise ValueError(f"aggregate_operating_points: converged candidate(s) {missing[:10]} have no D-source usage "
                             "record (absent from `usage` or an empty statuses_used); the section 14 support rank cannot "
                             "be computed for them (task X finding VL2-03)")
        out["statuses_used"] = st.fillna("").astype(str)
    else:
        # no usage record at all: the table cannot be ranked (rank_recipes refuses an untracked converged candidate)
        out["statuses_used"] = ""
    return out
