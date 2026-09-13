"""Script 8 of DESIGN.md section 11: the Pr/Nd case, PC88A versus Cyanex 272 (section 13).

Inputs are the literature entries ``systems/sys_29976921e156a0a0.json`` (PC88A) and
``systems/sys_f02db527a94a5e86.json`` (Cyanex 272) -- every number ``ASSUMED_PLACEHOLDER`` with a
range -- plus ``cases/prnd_feed.json`` and ``cases/prnd_spec.json`` (feed and spec grid, both
``assumed``, open item U1).  ``--allow-placeholders`` is required to run at all (13.2); the script
refuses when any ``assumed`` value lacks a range (validator V9) and stamps every output
``PLACEHOLDER_PARAMETERS``.

Per system: ``--draws`` parameter draws (seed 18; ``log_k.Nd``, ``SF(Nd/Pr)`` -- PC88A uniform in
[1.3, 1.5], Cyanex 272 the assumed sweep {1.1, 1.2, 1.3, 1.4} assigned to the draws in turn --
``a_dimer``, ``b_proton``, each uniform inside its declared range; the extractant concentration is
a design variable); per draw ``lhs_pareto`` with ``--n-lhs`` rows over ``n_ext, n_scr, n_str``
(1 to ``--max-stages``), ``oa_ext, s_over_a, w_over_a, scrub_acid_M, scrub_target_mM``
(displacement scrub), ``strip_acid_M`` (0.5-6 M), ``ligand_total_M`` (the entry's range),
``saponification_degree`` in {0, 0.3, 0.5} (``SAPONIFICATION_RANGE_UNKNOWN`` when > 0),
``feed_stage_offset``, ``scrub_return_offset``; the same LHS design (same seed) for every draw
so draws are paired.  Outputs (13.3, all under ``results/case_prnd/`` with regime tag and
parameter status): ``parameters.md``, per-draw tables, ``cell_best.csv`` (best on-spec regime per
draw and spec cell), ``intervals.csv`` / ``intervals.md`` (min / median / max over draws),
``sensitivity.csv`` (one factor at a time at the median draw), ``displacement_scrub.csv`` (all
four origin quantities), ``sanity_limit.csv`` (per-section constant D, no acid balance),
``cyanex_vs_pc88a.md`` (conditional comparison per draw pair), ``consistency_checks.md`` (13.4
a-c with their verdict strings), ``manifest.json``.  With ``--todga-feed`` the exploratory DGA +
aqueous-ligand study of 13.5 is written under ``todga_exploration/``.

Cost: the cascade solver is capped per candidate (``--solver-max-newton``,
``--solver-max-sweeps``); failed candidates are counted, never raised.  The projected time is
printed from ``results/bench/timing.json`` and re-estimated after the first draw (printed only;
no wall-clock value is written).

Usage (from the repository root, one process):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_case_prnd.py \\
        --feed cases/prnd_feed.json --spec cases/prnd_spec.json --allow-placeholders \\
        [--draws 64] [--n-lhs 1000] [--seed 18] [--max-stages 40] [--out-dir results/case_prnd] \\
        [--todga-feed cases/todga_prnd_feed.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc import optimize as OPT  # noqa: E402
from gen18proc.cascade import section_tracer_d, solve_cascade  # noqa: E402
from gen18proc.dmodel import ASSUMPTIONS, ConstantD, SystemModel, build_system_model  # noqa: E402
from gen18proc.literature import (  # noqa: E402
    SF_ND_PR_PLACEHOLDER,
    todga_hydrophilic_complexant_placeholder,
)
from gen18proc.metrics import Prices, compute_metrics  # noqa: E402
from gen18proc.report import markdown_table, regime_line, write_manifest, write_table  # noqa: E402
from gen18proc.systems import iter_sourced, load_system  # noqa: E402
from gen18proc.types import (  # noqa: E402
    AqStream,
    CascadeSpec,
    ModelBuildError,
    ProvStatus,
    Sourced,
)

CASE_SYSTEMS: tuple[str, ...] = ("sys_29976921e156a0a0", "sys_f02db527a94a5e86")
"""PC88A then Cyanex 272 (DESIGN.md sections 3.1 and 3.7)."""

PLACEHOLDER_STAMP = "PLACEHOLDER_PARAMETERS"
LADDER: tuple[int, ...] = (3, 5, 7, 10, 15, 20, 30, 40)
"""Stage-count ladder (n_ext = n_scr = N, n_str = 8) of consistency check (b)."""
STRIP_STAGES_B = 8
SCALAR_KEYS = ("n_stages_total", "n_ext", "n_scr", "n_str", "oa_ext", "s_over_a", "w_over_a",
               "scrub_acid_M", "scrub_target_mM", "strip_acid_M", "ligand_total_M",
               "saponification_degree", "purity_mol", "recovery_from_feed", "recovery_total",
               "acid_mol_per_kg_oxide", "base_mol_per_kg_oxide", "water_L_per_kg_oxide",
               "extractant_makeup_mol_per_kg_oxide", "consumption_index",
               "cost_proxy_per_kg_oxide")


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def sourced_value(obj: Any) -> float:
    return float(Sourced.from_json(obj).value)


def feed_stream(feed: dict[str, Any]) -> AqStream:
    """The feed ``AqStream`` of a case file (WB1 addendum A8 shape; metals mM -> mol/L)."""
    return AqStream(flow_L_h=sourced_value(feed["flow_L_h"]),
                    metals={m: sourced_value(v) * 1e-3 for m, v in feed["metals_mM"].items()},
                    h=sourced_value(feed["h_M"]), anion=sourced_value(feed["anion_M"]),
                    complexant_total=sourced_value(feed["complexant_total_M"]),
                    sodium=sourced_value(feed["sodium_M"]))


def base_spec_for(feed: AqStream, target: str, ligand: str, ligand_total: float) -> CascadeSpec:
    zeros = {m: 0.0 for m in feed.metals}
    scrub = AqStream(0.3 * feed.flow_L_h, dict(zeros), 0.2, 0.2, 0.0, 0.0)
    strip = AqStream(0.5 * feed.flow_L_h, dict(zeros), 3.0, 3.0, 0.0, 0.0)
    return CascadeSpec(6, 3, 3, feed, scrub, strip, feed.flow_L_h, {ligand: ligand_total}, 0.0,
                       target=target)


def placeholder_guard(entry, feed: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """V9: every ``assumed`` numeric of the entry and the case files must declare a range.
    Returns the offending paths (empty when the guard passes)."""
    bad = []
    for path, s in iter_sourced(entry):
        if s.status == ProvStatus.ASSUMED and s.value is not None and s.range is None:
            bad.append(f"{entry.system_id}:{path}")

    def walk(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            if "status" in obj and "value" in obj:
                if obj.get("status") == "assumed" and obj.get("value") is not None \
                        and obj.get("range") is None:
                    bad.append(prefix)
                return
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}")

    walk(feed, "feed")
    walk(spec, "spec")
    return bad


def parameter_table(entries: list, feed: dict[str, Any], spec: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        for path, s in iter_sourced(entry):
            if s.value is None and s.status == ProvStatus.UNKNOWN and "records" in path:
                continue
            rows.append({"system_id": entry.system_id, "parameter": path, "value": s.value,
                         "unit": s.unit, "status": s.status.value,
                         "range": "" if s.range is None else f"[{s.range[0]}, {s.range[1]}]",
                         "source": (s.source.doi or s.source.kind) + (
                             f" | {s.source.locator}" if s.source.locator else ""),
                         "note": s.provenance.note})
    for name, obj in (("feed", feed), ("spec", spec)):
        def walk(o: Any, prefix: str) -> None:
            if isinstance(o, dict):
                if "status" in o and ("value" in o or "values" in o):
                    rows.append({"system_id": name, "parameter": prefix,
                                 "value": o.get("value", o.get("values")),
                                 "unit": o.get("unit", ""), "status": o.get("status"),
                                 "range": "" if o.get("range") is None else str(o["range"]),
                                 "source": ((o.get("source") or {}).get("doi") or
                                            (o.get("source") or {}).get("kind") or "") + (
                                     " | " + str((o.get("source") or {}).get("locator"))
                                     if (o.get("source") or {}).get("locator") else ""),
                                 "note": o.get("note", "")})
                    return
                for k, v in o.items():
                    walk(v, f"{prefix}.{k}")
        walk(obj, name)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# draws
# ---------------------------------------------------------------------------------------------

def draw_table(entry, ligand: str, n_draws: int, seed: int) -> pd.DataFrame:
    """Parameter draws of section 13.2 for one system: ``log_k.Nd``, SF(Nd/Pr), ``a_dimer``,
    ``b_proton`` uniform inside their declared ranges (seed 18); ``log_k.Pr = log_k.Nd -
    log10 SF`` (clipped into its own declared range when the two windows disagree, counted)."""
    rng = np.random.default_rng(seed)
    block = entry.params[ligand]["20-30C"]
    r_nd = block.log_k["Nd"].range
    r_pr = block.log_k["Pr"].range
    r_a = block.a_dimer.range
    r_b = block.b_proton.range
    sf_info = SF_ND_PR_PLACEHOLDER[entry.system_id]
    rows = []
    for i in range(n_draws):
        log_k_nd = float(rng.uniform(r_nd[0], r_nd[1]))
        if "sweep" in sf_info:
            sf = float(sf_info["sweep"][i % len(sf_info["sweep"])])
            sf_mode = "assumed sweep, assigned in turn"
        else:
            sf = float(rng.uniform(sf_info["range"][0], sf_info["range"][1]))
            sf_mode = "uniform in the placeholder range"
        log_k_pr_raw = log_k_nd - math.log10(sf)
        log_k_pr = min(max(log_k_pr_raw, r_pr[0]), r_pr[1])
        rows.append({"draw": i, "log_k_Nd": log_k_nd, "sf_nd_pr": sf, "sf_mode": sf_mode,
                     "log_k_Pr": log_k_pr, "log_k_Pr_clipped": bool(log_k_pr != log_k_pr_raw),
                     "a_dimer": float(rng.uniform(r_a[0], r_a[1])),
                     "b_proton": float(rng.uniform(r_b[0], r_b[1]))})
    return pd.DataFrame(rows)


def median_draw(draws: pd.DataFrame) -> dict[str, float]:
    return {"log_k_Nd": float(draws["log_k_Nd"].median()),
            "sf_nd_pr": float(draws["sf_nd_pr"].median()),
            "a_dimer": float(draws["a_dimer"].median()),
            "b_proton": float(draws["b_proton"].median())}


def draw_to_parameters(ligand: str, d: dict[str, float], entry) -> dict[str, float]:
    block = entry.params[ligand]["20-30C"]
    r_pr = block.log_k["Pr"].range
    log_k_pr = min(max(d["log_k_Nd"] - math.log10(d["sf_nd_pr"]), r_pr[0]), r_pr[1])
    return {f"{ligand}.log_k.Nd": d["log_k_Nd"], f"{ligand}.log_k.Pr": log_k_pr,
            f"{ligand}.a_dimer": d["a_dimer"], f"{ligand}.b_proton": d["b_proton"]}


def build(entry, feed_anion: str, temperature: float, draw: dict[str, float] | None,
          **kw: Any) -> SystemModel:
    model = build_system_model(entry, feed_anion=feed_anion, temperature_C=temperature,
                               params_source="literature", parameter_draw=draw, **kw)
    if isinstance(model, ModelBuildError):
        raise SystemExit(f"{entry.system_id}: {model.reason}")
    return model


# ---------------------------------------------------------------------------------------------
# the labelled sanity limit: constant D per section, no acid balance
# ---------------------------------------------------------------------------------------------

class SectionConstantD(ConstantD):
    """Constant D per metal and per section (extraction / scrub / strip): the tracer D of each
    section's inlet, selected by the stage's aqueous [H+] (nearest section inlet acid in log
    space); ``q = p = z = 0`` -- no ligand depletion, no acid release, no anion transport.  The
    labelled sanity limit of DESIGN.md section 13.3, never a D source."""

    def __init__(self, tracer: dict[str, dict[str, float]], inlet_h: dict[str, float],
                 ligand: str):
        metals = tuple(next(iter(tracer.values())))
        super().__init__({m: 0.0 for m in metals}, ligand=ligand)
        self._sections = tuple(tracer)
        self._log_h = np.array([math.log10(max(inlet_h[s], 1e-12)) for s in self._sections])
        self._logd = np.array([[math.log10(max(tracer[s][m], 1e-300)) for m in metals]
                               for s in self._sections])

    def core(self, L, h, nu, c, ligand_total, capacity=None):
        j = int(np.argmin(np.abs(self._log_h - math.log10(max(h, 1e-12)))))
        return self._logd[j], self._zeros, self._zeros, self._zeros, self._zeros


def sanity_system(spec: CascadeSpec, system: SystemModel) -> SystemModel:
    tracer = section_tracer_d(spec, system)
    inlet_h = {"extraction": spec.feed.h, "scrub": spec.scrub.h, "strip": spec.strip.h}
    ligand = system.ligands[0]
    model = SectionConstantD(tracer, {s: inlet_h[s] for s in tracer}, ligand)
    return SystemModel(entry=system.entry, dmodels={ligand: model}, complexant=None,
                       activity=None, temperature_C=system.temperature_C,
                       assumptions=ASSUMPTIONS, phase=system.phase, params_source="constant")


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def regime_of(status: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    reg = {"cohort": "Pr/Nd case: literature placeholder entries PC88A and Cyanex 272 "
                     "(no corpus rows); feed_status assumed (U1)",
           "holdout": "none (computed regimes; consistency checks are not validation)",
           "averaging_unit": "draw (parameter draw, seed 18)", "status_of_parameters": status}
    if extra:
        reg.update(extra)
    return reg


def evaluate_fixed(values: dict[str, float], base: CascadeSpec, system: SystemModel,
                   target: str, impurities: list[str], limits: dict[str, float],
                   prices: Prices, solver_kwargs: dict[str, Any]) -> dict[str, Any]:
    return OPT.evaluate_candidate(values, base, system, target, impurities, limits, prices,
                                  solver_kwargs=solver_kwargs)


def summarise_cells(cell_best: pd.DataFrame, n_draws: int) -> pd.DataFrame:
    """min / median / max over the draws where each cell is reachable."""
    rows = []
    keys = ["system_id", "subset", "purity_min", "recovery_min"]
    for key, g in cell_best.groupby(keys, sort=True):
        reach = g.loc[g["reachable"].astype(bool)]
        row = dict(zip(keys, key))
        row["n_draws"] = n_draws
        row["draws_reachable"] = int(len(reach))
        row["fraction_reachable"] = len(reach) / n_draws if n_draws else math.nan
        for col in SCALAR_KEYS:
            vals = pd.to_numeric(reach[col], errors="coerce") if len(reach) else pd.Series(
                dtype=float)
            row[f"{col}_min"] = float(vals.min()) if len(vals) and vals.notna().any() else math.nan
            row[f"{col}_median"] = float(vals.median()) if len(vals) and vals.notna().any() \
                else math.nan
            row[f"{col}_max"] = float(vals.max()) if len(vals) and vals.notna().any() else math.nan
        flags: dict[str, int] = {}
        for s in reach["flags"].astype(str) if len(reach) else []:
            for f in s.split("|"):
                if f:
                    flags[f] = flags.get(f, 0) + 1
        row["flags_over_draws"] = "; ".join(f"{k}:{v}" for k, v in sorted(flags.items()))
        row["regime_status_over_draws"] = "; ".join(
            f"{k}:{v}" for k, v in sorted(reach["regime_status"].value_counts().items())) \
            if len(reach) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def fenske_min_stages(purity: float, recovery: float, feed_ratio_nd_pr: float, sf: float,
                      ) -> float:
    """Fenske-type minimum number of theoretical stages at total reflux for a binary Nd/Pr split
    with product purity ``purity`` (Nd mol fraction) and Nd recovery ``recovery`` from a feed of
    mol ratio ``feed_ratio_nd_pr``, with a constant separation factor ``sf``:
    ``N_min = ln[(x_Nd/x_Pr)_product / (x_Nd/x_Pr)_raffinate] / ln SF``.  An analytic bound used
    only for the order-of-magnitude reading of consistency check (b); it is not the cascade."""
    nd_f, pr_f = feed_ratio_nd_pr, 1.0
    prod_nd = recovery * nd_f
    prod_pr = prod_nd * (1.0 - purity) / purity
    raff_nd = nd_f - prod_nd
    raff_pr = pr_f - prod_pr
    if raff_pr <= 0 or raff_nd <= 0 or prod_pr <= 0:
        return math.inf
    return math.log((prod_nd / prod_pr) / (raff_nd / raff_pr)) / math.log(sf)


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feed", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--n-lhs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--allow-placeholders", action="store_true")
    ap.add_argument("--out-dir", default=str(paths.RESULTS_CASE_PRND_DIR))
    ap.add_argument("--systems", nargs="*", default=list(CASE_SYSTEMS))
    ap.add_argument("--max-stages", type=int, default=40,
                    help="upper bound of n_ext, n_scr, n_str (DESIGN 13.2: 40)")
    ap.add_argument("--solver-max-newton", type=int, default=40)
    ap.add_argument("--solver-max-sweeps", type=int, default=10)
    ap.add_argument("--skip-ladder", action="store_true",
                    help="skip the stage ladder of consistency check (b)")
    ap.add_argument("--todga-feed", default=None,
                    help="cases/todga_prnd_feed.json: run the 13.5 exploration as well")
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = paths.G18_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    systems_dir = Path(args.systems_dir)
    feed_path = paths.G18_ROOT / args.feed if not Path(args.feed).is_absolute() else Path(args.feed)
    spec_path = paths.G18_ROOT / args.spec if not Path(args.spec).is_absolute() else Path(args.spec)
    feed_json, spec_json = load_json(feed_path), load_json(spec_path)
    entries = [load_system(systems_dir / f"{sid}.json") for sid in args.systems]
    bad = []
    for e in entries:
        bad += placeholder_guard(e, feed_json, spec_json)
    if bad:
        raise SystemExit("refused (V9): assumed values without a range: " + ", ".join(bad))
    any_assumed = any(s.status == ProvStatus.ASSUMED for e in entries for _, s in iter_sourced(e))
    if any_assumed and not args.allow_placeholders:
        raise SystemExit("refused: the case systems carry ASSUMED_PLACEHOLDER parameters; pass "
                         "--allow-placeholders to run (every output is then stamped "
                         f"{PLACEHOLDER_STAMP})")
    status = (f"{PLACEHOLDER_STAMP} (assumed with ranges, --allow-placeholders); feed_status "
              f"{feed_json.get('feed_status')}; spec_status {spec_json.get('spec_status')}")
    prices = Prices.load()
    solver_kwargs = {"max_newton": args.solver_max_newton, "max_sweeps": args.solver_max_sweeps}
    target = str(spec_json["target"])
    impurities = [str(m) for m in spec_json["impurities"]]
    purity_grid = [float(v) for v in spec_json["purity_min_grid"]["values"]]
    recovery_grid = [float(v) for v in spec_json["recovery_min_grid"]["values"]]
    cons_cell = (float(spec_json["consistency_cell"]["purity_min"]),
                 float(spec_json["consistency_cell"]["recovery_min"]))
    patent_cell = (float(spec_json["patent_cell"]["purity_min"]),
                   float(spec_json["patent_cell"]["recovery_min"]))
    loosest = {"purity_min": min(purity_grid), "recovery_min": min(recovery_grid)}
    feed = feed_stream(feed_json)
    temperature = sourced_value(feed_json["temperature_C"])
    outputs: list[Path] = []

    # bench projection (printed only)
    bench = paths.RESULTS_BENCH_DIR / "timing.json"
    if bench.exists():
        proj = load_json(bench).get("projection", {})
        per_1000 = float(proj.get("per_1000_cascades_s", math.nan))
        total = args.draws * len(entries) * args.n_lhs
        print(f"bench projection: {per_1000:.1f} s per 1000 reference cascades -> "
              f"{total} cascades = {total / 1000 * per_1000 / 60:.1f} min at the reference "
              "cost (the LHS mean cost with 1-%d stages is larger; re-estimated after draw 0)"
              % args.max_stages)

    # parameters.md
    ptab = parameter_table(entries, feed_json, spec_json)
    outputs.append(write_table(ptab, out_dir / "parameters.md", regime=regime_of(status)))

    cell_best_all: list[pd.DataFrame] = []
    draw_tables: dict[str, pd.DataFrame] = {}
    ref_rows: dict[str, dict[str, Any]] = {}
    ref_specs: dict[str, tuple[CascadeSpec, SystemModel, dict[str, float]]] = {}
    ladder_rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    t_draws = 0.0                      # seconds spent inside the per-draw LHS loops only
    for entry in entries:
        sid = entry.system_id
        ligand = entry.organic_ligands[0].name
        lig_conc = entry.organic_ligands[0].concentration
        lig_range = tuple(lig_conc.range) if lig_conc.range else (0.2, 1.5)
        dom = entry.applicability.get(f"{ligand}|20-30C")
        space = OPT.DesignSpace.from_config(
            family=entry.family, domain=dom, ligand=ligand,
            widen={"n_ext": (1, args.max_stages), "n_scr": (1, args.max_stages),
                   "n_str": (1, args.max_stages),
                   "feed_stage_offset": (0, args.max_stages - 1),
                   "scrub_return_offset": (0, args.max_stages - 1),
                   "scrub_acid_M": (0.01, 3.0), "scrub_target_mM": (0.0, 50.0),
                   "strip_acid_M": (0.5, 6.0), "ligand_total_M": lig_range},
            fixed={"strip_anion_M": 0.0, "scrub_complexant_M": 0.0, "feed_dilution": 0.0,
                   "f_bleed": 0.0},
            levels={"saponification_degree": (0.0, 0.3, 0.5)})
        draws = draw_table(entry, ligand, args.draws, args.seed)
        draw_tables[sid] = draws
        sys_dir = out_dir / sid
        sys_dir.mkdir(parents=True, exist_ok=True)
        outputs.append(write_table(draws, sys_dir / "draws.csv", regime=regime_of(status),
                                   regime_table=True))
        base = base_spec_for(feed, target, ligand, float(lig_conc.value) / 2.0)
        cell_rows = []
        for d in draws.itertuples():
            params = draw_to_parameters(ligand, d._asdict(), entry)
            system = build(entry, feed_json["anion"], temperature, params)
            t0 = time.perf_counter()
            df = OPT.lhs_pareto(space, base, system, target, impurities, loosest, prices,
                                n=args.n_lhs, seed=args.seed, solver_kwargs=solver_kwargs)
            dt = time.perf_counter() - t0
            t_draws += dt
            knees = OPT.epsilon_knees(df, purity_grid, recovery_grid)
            df.insert(0, "draw", int(d.draw))
            outputs.append(write_table(df, sys_dir / f"draw_{int(d.draw):03d}.csv",
                                       regime=regime_of(status, {"draw": str(int(d.draw))}),
                                       regime_table=True))
            knees.insert(0, "draw", int(d.draw))
            knees.insert(0, "system_id", sid)
            cell_rows.append(knees)
            if int(d.draw) == 0:
                remaining = (args.draws * len(entries) - 1) * dt
                print(f"{sid}: draw 0 took {dt:.1f} s for {args.n_lhs} cascades "
                      f"({df.attrs['n_failed']} failed, {df.attrs['n_invalid_spec']} invalid); "
                      f"projected remaining {remaining / 60:.1f} min")
        cell_best = pd.concat(cell_rows, ignore_index=True)
        cell_best_all.append(cell_best)

        # reference regime at the median draw: the consistency cell's knee, else the loosest
        # cell's knee, else the converged row of highest purity meeting the loosest recovery
        med = median_draw(draws)
        med_params = draw_to_parameters(ligand, med, entry)
        med_system = build(entry, feed_json["anion"], temperature, med_params)
        df_med = OPT.lhs_pareto(space, base, med_system, target, impurities, loosest, prices,
                                n=args.n_lhs, seed=args.seed, solver_kwargs=solver_kwargs)
        knees_med = OPT.epsilon_knees(df_med, purity_grid, recovery_grid)
        df_med.insert(0, "draw", -1)
        outputs.append(write_table(df_med, sys_dir / "draw_median.csv",
                                   regime=regime_of(status, {"draw": "median"}),
                                   regime_table=True))
        ref = None
        for pmin, rmin in (cons_cell, (min(purity_grid), min(recovery_grid))):
            k = knees_med.loc[(knees_med.subset == "all") & (knees_med.purity_min == pmin)
                              & (knees_med.recovery_min == rmin) & knees_med.reachable]
            if len(k):
                ref = k.iloc[0]
                ref_kind = f"knee of cell ({pmin}, {rmin}) at the median draw"
                break
        if ref is None:
            conv = df_med.loc[df_med["status"].str.startswith("converged")
                              & (df_med["recovery_from_feed"] >= min(recovery_grid))]
            if len(conv):
                ref = conv.loc[conv["purity_mol"].idxmax()]
                ref_kind = "highest-purity converged regime meeting the loosest recovery"
            else:
                conv = df_med.loc[df_med["status"].str.startswith("converged")]
                if len(conv):
                    score = conv["purity_mol"].fillna(0.0) * conv["recovery_from_feed"].fillna(0.0)
                    ref = conv.loc[score.idxmax()]
                else:
                    ref = df_med.iloc[0]
                ref_kind = "converged regime of largest purity x recovery (no cell reachable)"
        ref_values = {v: float(ref[v]) for v in space.variables}
        ref_values.update(space.fixed)
        ref_rows[sid] = {"system_id": sid, "reference_regime": ref_kind, **ref_values,
                         "purity_mol": float(ref["purity_mol"]),
                         "recovery_from_feed": float(ref["recovery_from_feed"])}
        ref_specs[sid] = (base, med_system, ref_values)
        print(f"{sid}: reference regime = {ref_kind}; purity {float(ref['purity_mol']):.3f}, "
              f"recovery {float(ref['recovery_from_feed']):.3f}")

        # consistency check (b): stage ladder at SF 1.4 (median draw otherwise), patent cell
        if not args.skip_ladder:
            sf14 = dict(med, sf_nd_pr=1.4)
            sys14 = build(entry, feed_json["anion"], temperature,
                          draw_to_parameters(ligand, sf14, entry))
            n_b = max(8, args.n_lhs // 50)
            patent_limits = {"purity_min": patent_cell[0], "recovery_min": patent_cell[1]}
            for N in [n for n in LADDER if n <= args.max_stages] or [args.max_stages]:
                sp = space.with_fixed(n_ext=N, n_scr=N, n_str=STRIP_STAGES_B)
                dfb = OPT.lhs_pareto(sp, base, sys14, target, impurities, patent_limits, prices,
                                     n=n_b, seed=args.seed, solver_kwargs=solver_kwargs)
                conv = dfb.loc[dfb["status"].str.startswith("converged")]
                reached = conv.loc[(conv["purity_mol"] >= patent_cell[0])
                                   & (conv["recovery_from_feed"] >= patent_cell[1])]
                best_p = float(conv["purity_mol"].max()) if len(conv) else math.nan
                ladder_rows.append({
                    "system_id": sid, "sf_nd_pr": 1.4, "n_ext": N, "n_scr": N,
                    "n_str": STRIP_STAGES_B, "n_lhs": n_b, "n_converged": int(len(conv)),
                    "n_failed": dfb.attrs["n_failed"], "cell_reached": bool(len(reached)),
                    "n_reached": int(len(reached)), "best_purity_mol": best_p,
                    "best_recovery_at_purity_ge_0.99": float(
                        conv.loc[conv["purity_mol"] >= 0.99, "recovery_from_feed"].max())
                    if len(conv) and (conv["purity_mol"] >= 0.99).any() else math.nan,
                    "max_purity_recovery_product": float(
                        (conv["purity_mol"] * conv["recovery_from_feed"]).max()) if len(conv)
                    else math.nan})
                print(f"{sid}: ladder N = {N}+{N}+{STRIP_STAGES_B}: reached = {bool(len(reached))}"
                      f" (best purity {best_p:.3f})")
                if len(reached):
                    break

    # ---- interval tables ---------------------------------------------------------------------
    if not cell_best_all:
        # ``--systems`` with no names: only the section 13.5 exploration is wanted (integration
        # 2026-09-13, so that the exploratory section can be regenerated without re-running the
        # whole case).  Everything above this point is per-system and has nothing to write.
        print("no case systems requested: skipping the interval tables and the comparison; "
              "running the section 13.5 exploration only")
        if args.todga_feed:
            outputs += todga_exploration(args, out_dir / "todga_exploration", prices,
                                         solver_kwargs)
        write_manifest(out_dir / "manifest_todga_only.json", outputs,
                       [feed_path, spec_path, paths.CONFIG_DIR / "prices.json",
                        paths.CONFIG_DIR / "design_spaces.json"],
                       args.seed, arguments=vars(args),
                       extra={"stamp": PLACEHOLDER_STAMP, "todga_exploration_only": True})
        return 0
    cell_best = pd.concat(cell_best_all, ignore_index=True)
    outputs.append(write_table(cell_best, out_dir / "cell_best.csv",
                               regime=regime_of(status, {"averaging_unit": "draw x spec cell"}),
                               regime_table=True))
    intervals = summarise_cells(cell_best, args.draws)
    outputs.append(write_table(intervals, out_dir / "intervals.csv",
                               regime=regime_of(status, {"averaging_unit":
                                                         "min / median / max over draws"}),
                               regime_table=True))
    short = intervals[["system_id", "subset", "purity_min", "recovery_min", "draws_reachable",
                       "n_draws", "n_stages_total_min", "n_stages_total_median",
                       "n_stages_total_max", "oa_ext_median", "s_over_a_median",
                       "w_over_a_median", "scrub_acid_M_median", "strip_acid_M_median",
                       "saponification_degree_median", "acid_mol_per_kg_oxide_median",
                       "base_mol_per_kg_oxide_median", "consumption_index_median",
                       "regime_status_over_draws"]]
    outputs.append(write_table(short, out_dir / "intervals.md",
                               regime=regime_of(status, {"averaging_unit":
                                                         "min / median / max over draws"}),
                               regime_table=True))

    # ---- sensitivities, displacement scrub, sanity limit at the reference regime ------------
    sens_rows, disp_rows, sanity_rows = [], [], []
    for entry in entries:
        sid = entry.system_id
        ligand = entry.organic_ligands[0].name
        base, med_system, ref_values = ref_specs[sid]
        med = median_draw(draw_tables[sid])
        block = entry.params[ligand]["20-30C"]
        cell_limits = {"purity_min": cons_cell[0], "recovery_min": cons_cell[1]}

        def run(values: dict[str, float], system: SystemModel, factor: str, value: Any,
                feed_override: AqStream | None = None) -> dict[str, Any]:
            b = base if feed_override is None else CascadeSpec(
                base.n_ext, base.n_scr, base.n_str, feed_override, base.scrub, base.strip,
                feed_override.flow_L_h, base.ligand_total, 0.0, target=target)
            row = evaluate_fixed(values, b, system, target, impurities, cell_limits, prices,
                                 solver_kwargs)
            keep = {k: row.get(k) for k in ("status", "purity_mol", "recovery_from_feed",
                                            "recovery_total", "scrub_target_return",
                                            "net_product_mol_h", "consumption_index",
                                            "acid_mol_per_kg_oxide", "base_mol_per_kg_oxide",
                                            "n_stages_total", "regime_status", "flags",
                                            "max_loading_" + ligand)}
            return {"system_id": sid, "factor": factor, "value": value, **keep}

        sens_rows.append(run(ref_values, med_system, "reference", "median draw"))
        sf_info = SF_ND_PR_PLACEHOLDER[sid]
        for sf in np.linspace(sf_info["range"][0], sf_info["range"][1], 5):
            sysx = build(entry, feed_json["anion"], temperature,
                         draw_to_parameters(ligand, dict(med, sf_nd_pr=float(sf)), entry))
            sens_rows.append(run(ref_values, sysx, "sf_nd_pr", float(sf)))
        r_nd = block.log_k["Nd"].range
        for lk in np.linspace(max(r_nd[0], med["log_k_Nd"] - 1.0),
                              min(r_nd[1], med["log_k_Nd"] + 1.0), 5):
            sysx = build(entry, feed_json["anion"], temperature,
                         draw_to_parameters(ligand, dict(med, log_k_Nd=float(lk)), entry))
            sens_rows.append(run(ref_values, sysx, "log_k_Nd (2-log window clipped to range)",
                                 float(lk)))
        lig_range = entry.organic_ligands[0].concentration.range or (0.2, 1.5)
        for ha in np.linspace(lig_range[0], lig_range[1], 5):
            sens_rows.append(run({**ref_values, "ligand_total_M": float(ha)}, med_system,
                                 "ligand_total_M", float(ha)))
        for s in (0.0, 0.3, 0.5):
            sens_rows.append(run({**ref_values, "saponification_degree": s}, med_system,
                                 "saponification_degree", s))
        for f in (0.5, 1.0, 2.0):
            sens_rows.append(run({**ref_values, "oa_ext": ref_values["oa_ext"] * f}, med_system,
                                 "oa_ext (x reference)", f))
        sens = feed_json.get("sensitivity", {})
        ratio_ref = feed.metals[target] / sum(v for m, v in feed.metals.items() if m != target)
        for tot in sens.get("total_metal_M", {}).get("values", []):
            nd = float(tot) * ratio_ref / (1.0 + ratio_ref)
            pr = float(tot) - nd
            fx = AqStream(feed.flow_L_h, {target: nd, impurities[0]: pr}, feed.h,
                          3.0 * float(tot) + feed.h, 0.0, 0.0)
            sens_rows.append(run(ref_values, med_system, "feed_total_metal_M", float(tot), fx))
        for ratio in sens.get("ratio_nd_pr", {}).get("values", []):
            a, b = (float(x) for x in str(ratio).split(":"))
            tot = sum(feed.metals.values())
            nd = tot * a / (a + b)
            fx = AqStream(feed.flow_L_h, {target: nd, impurities[0]: tot - nd}, feed.h,
                          3.0 * tot + feed.h, 0.0, 0.0)
            sens_rows.append(run(ref_values, med_system, "feed_ratio_nd_pr", str(ratio), fx))
        # displacement scrub: the scrub liquor carries the target
        for mm in (0.0, 10.0, 25.0, 50.0):
            disp_rows.append(run({**ref_values, "scrub_target_mM": mm}, med_system,
                                 "scrub_target_mM", mm))
        # labelled sanity limit
        spec_ref, _ = OPT.candidate_spec(dict(ref_values), base, med_system, target)
        loaded = run(ref_values, med_system, "loading-aware (mass action, acid balance)", "")
        sanity_rows.append(loaded)
        sane = sanity_system(spec_ref, med_system)
        with np.errstate(all="ignore"):
            res = solve_cascade(spec_ref, sane, **solver_kwargs)
            m = compute_metrics(res, spec_ref, sane, target, impurities, prices,
                                spec_limits=cell_limits)
        sanity_rows.append({"system_id": sid, "factor": "sanity limit: ConstantD per section at "
                                                        "the tracer D, no acid balance",
                            "value": "", "status": res.status, "purity_mol": m.purity_mol,
                            "recovery_from_feed": m.recovery_from_feed,
                            "recovery_total": m.recovery_total,
                            "scrub_target_return": m.scrub_target_return,
                            "net_product_mol_h": m.net_product_mol_h,
                            "consumption_index": OPT.consumption_index(m.consumption)[0],
                            "acid_mol_per_kg_oxide": m.consumption.get("acid_mol_per_kg_oxide"),
                            "base_mol_per_kg_oxide": m.consumption.get("base_mol_per_kg_oxide"),
                            "n_stages_total": m.n_stages_total, "regime_status": m.regime_status,
                            "flags": "|".join(sorted(f.value for f in m.flags)),
                            "max_loading_" + ligand: m.max_loading_fraction.get(ligand)})
    ref_table = pd.DataFrame(list(ref_rows.values()))
    outputs.append(write_table(ref_table, out_dir / "reference_regimes.csv",
                               regime=regime_of(status, {"averaging_unit": "median draw"}),
                               regime_table=True))
    outputs.append(write_table(pd.DataFrame(sens_rows), out_dir / "sensitivity.csv",
                               regime=regime_of(status, {"averaging_unit":
                                                         "one factor at a time, median draw"}),
                               regime_table=True))
    outputs.append(write_table(pd.DataFrame(disp_rows), out_dir / "displacement_scrub.csv",
                               regime=regime_of(status, {"averaging_unit": "median draw"}),
                               regime_table=True))
    outputs.append(write_table(pd.DataFrame(sanity_rows), out_dir / "sanity_limit.csv",
                               regime=regime_of(status, {"averaging_unit": "median draw"}),
                               regime_table=True))

    # ---- consistency checks (a)-(c) ---------------------------------------------------------
    lines = ["# Consistency checks against the cited numbers (DESIGN.md section 13.4) -- never "
             "validation", "",
             f"Regime: {regime_line(regime_of(status))}",
             "", "Feeds, acidities and loadings of the sources are unknown (LITERATURE_NOTES.md "
             "section 2); none of (a)-(c) can validate the model.", ""]
    pc = "sys_29976921e156a0a0"
    verdicts: dict[str, str] = {}
    cb = cell_best.loc[(cell_best.system_id == pc) & (cell_best.subset == "all")
                       & (cell_best.purity_min == cons_cell[0])
                       & (cell_best.recovery_min == cons_cell[1])]
    reach = cb.loc[cb["reachable"].astype(bool)] if len(cb) else cb
    if len(reach):
        n_lo, n_hi = int(reach["n_stages_total"].min()), int(reach["n_stages_total"].max())
        verdicts["a"] = (f"consistent with the cited outcome (reachable in {len(reach)}/"
                         f"{args.draws} draws with N = {n_lo}-{n_hi} stages)")
        detail = ("O/A %.2f-%.2f, S/A %.2f-%.2f, scrub acid %.3f-%.3f M, scrub Nd %.1f-%.1f mM, "
                  "saponification %s" % (
                      reach["oa_ext"].min(), reach["oa_ext"].max(), reach["s_over_a"].min(),
                      reach["s_over_a"].max(), reach["scrub_acid_M"].min(),
                      reach["scrub_acid_M"].max(), reach["scrub_target_mM"].min(),
                      reach["scrub_target_mM"].max(),
                      sorted(set(reach["saponification_degree"].round(2)))))
    else:
        verdicts["a"] = "not reachable within the assumed window"
        detail = "no draw reached the cell in the sweep"
    lines += ["## (a) Thakur 1993 (doi 10.1016/0304-386X(93)90084-Q): 97 % purity at > 85 % "
              "recovery, counter-current PC88A", "",
              f"Cell ({cons_cell[0]}, {cons_cell[1]}), PC88A, subset all: **{verdicts['a']}**; "
              f"{detail}.", ""]
    ladder = pd.DataFrame(ladder_rows)
    if len(ladder):
        outputs.append(write_table(ladder, out_dir / "stage_ladder_b.csv",
                                   regime=regime_of(status, {"averaging_unit": "ladder rung"}),
                                   regime_table=True))
    lp = ladder.loc[ladder.system_id == pc] if len(ladder) else ladder
    ratio = feed.metals[target] / sum(v for m, v in feed.metals.items() if m != target)
    n_min = fenske_min_stages(patent_cell[0], patent_cell[1], ratio, 1.4)
    if len(lp) and lp["cell_reached"].any():
        first = lp.loc[lp["cell_reached"]].iloc[0]
        n_reach = int(first["n_ext"])
        order = "of the order of 70 + 70" if n_reach >= 20 else "closer to 7 + 7 than to 70 + 70"
        verdicts["b"] = (f"at SF 1.4 the model reaches the ({patent_cell[0]}, {patent_cell[1]}) "
                         f"cell first at n_ext = n_scr = {n_reach} (+ {STRIP_STAGES_B} strip): "
                         f"{order}")
    elif len(lp):
        n_max = int(lp["n_ext"].max())
        verdicts["b"] = (f"at SF 1.4 the ({patent_cell[0]}, {patent_cell[1]}) cell is not "
                         f"reached up to n_ext = n_scr = {n_max} (+ {STRIP_STAGES_B} strip; the "
                         "solver budget ends there): the stage count the model needs is of the "
                         "order of 70 + 70 rather than 7 + 7")
    else:
        verdicts["b"] = "ladder skipped (--skip-ladder)"
    lines += ["## (b) EP2388344A1: PC-88A Nd/Pr circuit 72 extraction + 72 scrub + 8 strip "
              "stages at SF 1.4", "",
              f"Stage ladder (n_ext = n_scr = N, n_str = {STRIP_STAGES_B}, "
              f"{max(8, args.n_lhs // 50)} LHS rows per rung, SF 1.4, median draw otherwise): "
              "**%s**." % verdicts["b"],
              "", f"Analytic Fenske-type minimum at total reflux for the "
              f"({patent_cell[0]}, {patent_cell[1]}) cell with SF 1.4 and feed Nd:Pr {ratio:.2f}: "
              f"N_min = {n_min:.1f} theoretical stages (a bound from the constant-SF ideal, not "
              "the cascade); practical countercurrent circuits need a multiple of it, which is "
              "the order of the patent's 72 + 72.", ""]
    if len(lp):
        lines += [markdown_table(lp), ""]
    sf_hi = SF_ND_PR_PLACEHOLDER[pc]["range"][1]
    verdicts["c"] = (f"enters only as the upper end of the SF placeholder range "
                     f"[{SF_ND_PR_PLACEHOLDER[pc]['range'][0]}, {sf_hi}]")
    lines += ["## (c) Banda 2014 (doi 10.1016/j.jiec.2014.03.002): maximum SF about 1.5", "",
              f"**{verdicts['c']}**; no number of Banda 2014 is reproduced or compared.", ""]
    cons_md = out_dir / "consistency_checks.md"
    cons_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    outputs.append(cons_md)

    # ---- cyanex_vs_pc88a.md ------------------------------------------------------------------
    cy = "sys_f02db527a94a5e86"
    comp_rows = []
    if pc in set(cell_best.system_id) and cy in set(cell_best.system_id):
        for subset in ("all", "in_domain_only"):
            for pmin in purity_grid:
                for rmin in recovery_grid:
                    a = cell_best.loc[(cell_best.system_id == pc) & (cell_best.subset == subset)
                                      & (cell_best.purity_min == pmin)
                                      & (cell_best.recovery_min == rmin)].set_index("draw")
                    b = cell_best.loc[(cell_best.system_id == cy) & (cell_best.subset == subset)
                                      & (cell_best.purity_min == pmin)
                                      & (cell_best.recovery_min == rmin)].set_index("draw")
                    both = a.index.intersection(b.index)
                    ra, rb = a.loc[both, "reachable"].astype(bool), \
                        b.loc[both, "reachable"].astype(bool)
                    both_ok = both[ra.to_numpy() & rb.to_numpy()]
                    fewer = int((a.loc[both_ok, "n_stages_total"] < b.loc[both_ok,
                                                                          "n_stages_total"]).sum())
                    less_acid = int((a.loc[both_ok, "acid_mol_per_kg_oxide"]
                                     < b.loc[both_ok, "acid_mol_per_kg_oxide"]).sum())
                    comp_rows.append({
                        "subset": subset, "purity_min": pmin, "recovery_min": rmin,
                        "draws": int(len(both)), "pc88a_reachable": int(ra.sum()),
                        "cyanex272_reachable": int(rb.sum()), "both_reachable": int(len(both_ok)),
                        "only_pc88a_reachable": int((ra & ~rb).sum()),
                        "only_cyanex272_reachable": int((~ra & rb).sum()),
                        "pc88a_fewer_stages": fewer, "pc88a_less_acid": less_acid,
                        "fraction_pc88a_fewer_stages": fewer / len(both_ok) if len(both_ok)
                        else math.nan,
                        "fraction_pc88a_less_acid": less_acid / len(both_ok) if len(both_ok)
                        else math.nan})
    comp = pd.DataFrame(comp_rows)
    cmp_lines = ["# PC88A versus Cyanex 272 at identical feed, spec grid and price table "
                 "(DESIGN.md section 13.3)", "",
                 f"*regime: {status}; feed {feed_path.name}, spec {spec_path.name}, "
                 f"{args.draws} draws, seed {args.seed}, {args.n_lhs} LHS rows per draw, "
                 f"stage bounds 1-{args.max_stages}*", "",
                 "**Conditional on the placeholder ranges.** Every parameter of both systems is "
                 "`ASSUMED_PLACEHOLDER` (PC88A SF from EP2388344A1 / Banda 2014, Cyanex 272 SF an "
                 "assumed sweep {1.1, 1.2, 1.3, 1.4} with no source found); the ranking cannot be "
                 "decided until the literature values are transcribed by a person (open item U4) "
                 "or measured. Draw i of PC88A is paired with draw i of Cyanex 272 (same LHS "
                 "design, independent parameter draws).", ""]
    if len(comp):
        cmp_lines += [markdown_table(comp), ""]
        outputs.append(write_table(comp, out_dir / "cyanex_vs_pc88a.csv",
                                   regime=regime_of(status, {"averaging_unit": "draw pair"}),
                                   regime_table=True))
    else:
        cmp_lines += ["(one of the two systems was not run)", ""]
    cmp_md = out_dir / "cyanex_vs_pc88a.md"
    cmp_md.write_text("\n".join(cmp_lines), encoding="utf-8", newline="\n")
    outputs.append(cmp_md)

    # ---- 13.5 TODGA exploration (optional) ----------------------------------------------------
    if args.todga_feed:
        outputs += todga_exploration(args, out_dir / "todga_exploration", prices, solver_kwargs)

    summary = {"schema": "gen18.case_prnd.1", "stamp": PLACEHOLDER_STAMP,
               "verdicts": verdicts, "draws": args.draws, "n_lhs": args.n_lhs,
               "max_stages": args.max_stages, "systems": args.systems,
               "reference_regimes": {k: v["reference_regime"] for k, v in ref_rows.items()},
               "fenske_n_min_patent_cell_sf_1.4": n_min}
    sp = out_dir / "summary.json"
    sp.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n",
                  encoding="utf-8", newline="\n")
    outputs.append(sp)
    write_manifest(out_dir / "manifest.json", outputs,
                   [feed_path, spec_path, paths.CONFIG_DIR / "prices.json",
                    paths.CONFIG_DIR / "design_spaces.json"]
                   + [systems_dir / f"{sid}.json" for sid in args.systems],
                   args.seed, arguments=vars(args), extra={"stamp": PLACEHOLDER_STAMP,
                                                           "verdicts": verdicts})
    elapsed = time.perf_counter() - t_start
    n_cascades = max(1, args.draws * len(entries) * args.n_lhs)
    per_cascade = t_draws / n_cascades
    fixed = elapsed - t_draws
    print(f"case study done: {len(outputs)} files under {out_dir}; verdict (a): {verdicts['a']}; "
          f"(b): {verdicts['b']}")
    print(f"measured: {elapsed / 60:.1f} min in all; {t_draws / 60:.1f} min for the "
          f"{n_cascades} LHS cascades of the draws ({per_cascade * 1e3:.0f} ms each at stage "
          f"bounds 1-{args.max_stages}) and {fixed / 60:.1f} min for the median draw, "
          f"sensitivities, ladder and 13.5; a full 64 x 2 x 1000 run projects to about "
          f"{(per_cascade * 64 * 2 * 1000 + fixed * 1000 / max(1, args.n_lhs)) / 60:.0f} min at "
          "this cost (printed only, never written)")
    return 0


# ---------------------------------------------------------------------------------------------
# 13.5  DGA + aqueous ligand exploration (labelled exploratory)
# ---------------------------------------------------------------------------------------------

def todga_exploration(args: argparse.Namespace, out_dir: Path, prices: Prices,
                      solver_kwargs: dict[str, Any]) -> list[Path]:
    """TODGA / nitrate / aliphatic with the Pr/Nd nitrate feed: (1) the TODGA-only Pareto table
    (D source per the pre-registered decision; ``K_H`` null -> ``ACID_UPTAKE_UNMODELLED``;
    ``THIRD_PHASE_RISK`` above the sourced LOC); (2) the hold-back complexant with placeholder
    betas swept (``delta log beta`` in [0, 1.5], ``log beta_Nd`` in [1, 4], ``log K_H`` 2) in
    the feed and in the scrub only, at the TODGA-only reference regime: enrichment gain against
    the beta contrast, complexant consumption with the assumed regeneration fraction, and the
    cross-flow depression of extraction with the complexant in the scrub only.  A sensitivity
    study, not a recommendation (open item U3)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    feed_path = paths.G18_ROOT / args.todga_feed if not Path(args.todga_feed).is_absolute() \
        else Path(args.todga_feed)
    feed_json = load_json(feed_path)
    sid = str(feed_json["system_id"])
    entry = load_system(Path(args.systems_dir) / f"{sid}.json")
    target, impurities = "Nd", ["Pr"]
    feed = feed_stream(feed_json)
    temperature = sourced_value(feed_json["temperature_C"])
    dec_path = paths.RESULTS_EVAL_DIR / "decision.json"
    adopted = bool(load_json(dec_path).get("m1_adopted", False)) if dec_path.exists() else False
    system = build_system_model(entry, feed_anion=feed_json["anion"], temperature_C=temperature,
                                params_source="auto", decision_adopted=adopted,
                                feed_metals=(target, *impurities))
    if isinstance(system, ModelBuildError):
        raise SystemExit(f"{sid}: {system.reason}")
    status = (f"exploratory (DESIGN 13.5); D source {system.params_source} (decision "
              f"m1_adopted = {adopted}); complexant betas ASSUMED_PLACEHOLDER sweep (U3); "
              f"feed_status {feed_json.get('feed_status')}")
    ligand = system.ligands[0]
    dom = entry.applicability.get(f"{ligand}|{system.bands.get(ligand, '20-30C')}")
    space = OPT.DesignSpace.from_config(
        family=entry.family, domain=dom, ligand=ligand,
        widen={"n_ext": (1, min(20, args.max_stages)), "n_scr": (1, min(20, args.max_stages)),
               "n_str": (1, min(20, args.max_stages)), "feed_stage_offset": (0, 19),
               "scrub_return_offset": (0, 19), "scrub_acid_M": (0.01, 5.0),
               "strip_acid_M": (0.01, 0.5), "ligand_total_M": (0.05, 0.4)},
        fixed={"saponification_degree": 0.0, "strip_anion_M": 0.0, "scrub_complexant_M": 0.0,
               "feed_dilution": 0.0, "f_bleed": 0.0, "scrub_target_mM": 0.0})
    base = base_spec_for(feed, target, ligand, 0.1)
    base = CascadeSpec(base.n_ext, base.n_scr, base.n_str, base.feed,
                       AqStream(base.scrub.flow_L_h, dict(base.scrub.metals), 3.0, 3.0, 0.0, 0.0),
                       AqStream(base.strip.flow_L_h, dict(base.strip.metals), 0.01, 0.01, 0.0,
                                0.0), base.organic_flow_L_h, base.ligand_total, 0.0,
                       target=target)
    loosest = {"purity_min": 0.95, "recovery_min": 0.80}
    n = max(20, args.n_lhs // 5)
    df = OPT.lhs_pareto(space, base, system, target, impurities, loosest, prices, n=n,
                        seed=args.seed, solver_kwargs=solver_kwargs)
    reg = {"cohort": f"{sid} (TODGA / nitrate / aliphatic), corpus-parameterised; nitrate "
                     "Pr/Nd feed (assumed)", "holdout": "none (computed regimes)",
           "averaging_unit": "candidate", "status_of_parameters": status}
    outputs.append(write_table(df, out_dir / "todga_only_pareto.csv", regime=reg,
                               regime_table=True))
    fronts = OPT.pareto_fronts(df)
    for name, fr in fronts.items():
        outputs.append(write_table(fr, out_dir / f"todga_only_front_{name}.csv", regime=reg,
                                   regime_table=True))
    conv = df.loc[df["status"].str.startswith("converged")]
    if conv.empty:
        (out_dir / "NOTE.md").write_text("no converged TODGA-only candidate; the complexant "
                                         "sweep was not run\n", encoding="utf-8")
        outputs.append(out_dir / "NOTE.md")
        return outputs
    ref = conv.loc[(conv["purity_mol"] * conv["recovery_from_feed"]).idxmax()]
    ref_values = {v: float(ref[v]) for v in space.variables}
    ref_values.update(space.fixed)
    # complexant sweep
    comp = todga_hydrophilic_complexant_placeholder()
    rows = []
    for variant in ("feed", "scrub_only"):
        for lb_nd in (1.0, 2.5, 4.0):
            for delta in (0.0, 0.5, 1.0, 1.5):
                draw = {f"{comp.name}.log_beta.Nd": lb_nd,
                        f"{comp.name}.log_beta.Pr": lb_nd + delta,
                        f"{comp.name}.protonation_logk.1": 2.0}
                sysc = build_system_model(entry, feed_anion=feed_json["anion"],
                                          temperature_C=temperature, params_source="auto",
                                          decision_adopted=adopted, complexant=comp,
                                          parameter_draw=draw,
                                          feed_metals=(target, *impurities))
                if isinstance(sysc, ModelBuildError):
                    rows.append({"variant": variant, "log_beta_Nd": lb_nd,
                                 "delta_log_beta": delta, "status": "build_error",
                                 "reason": sysc.reason})
                    continue
                c_total = float(comp.concentration.value)
                if variant == "feed":
                    fx = AqStream(feed.flow_L_h, dict(feed.metals), feed.h, feed.anion, c_total,
                                  feed.sodium)
                    b = CascadeSpec(base.n_ext, base.n_scr, base.n_str, fx, base.scrub,
                                    base.strip, base.organic_flow_L_h, base.ligand_total, 0.0,
                                    target=target)
                    vals = dict(ref_values)
                else:
                    b = base
                    vals = {**ref_values, "scrub_complexant_M": c_total}
                row = OPT.evaluate_candidate(vals, b, sysc, target, impurities, loosest, prices,
                                             solver_kwargs=solver_kwargs)
                rows.append({"variant": variant, "log_beta_Nd": lb_nd, "delta_log_beta": delta,
                             "log_k_h": 2.0, "complexant_M": c_total,
                             **{k: row.get(k) for k in (
                                 "status", "purity_mol", "recovery_from_feed", "recovery_total",
                                 "enrichment_factor_Pr", "complexant_mol_per_kg_oxide",
                                 "consumption_index", "regime_status", "flags")}})
    base_row = OPT.evaluate_candidate(dict(ref_values), base, system, target, impurities,
                                      loosest, prices, solver_kwargs=solver_kwargs)
    rows.insert(0, {"variant": "none (TODGA only)", "log_beta_Nd": math.nan,
                    "delta_log_beta": math.nan, "log_k_h": math.nan, "complexant_M": 0.0,
                    **{k: base_row.get(k) for k in (
                        "status", "purity_mol", "recovery_from_feed", "recovery_total",
                        "enrichment_factor_Pr", "complexant_mol_per_kg_oxide",
                        "consumption_index", "regime_status", "flags")}})
    sweep = pd.DataFrame(rows)
    outputs.append(write_table(sweep, out_dir / "complexant_sweep.csv", regime={
        **reg, "averaging_unit": "reference regime x beta grid"}, regime_table=True))
    md = ["# DGA + aqueous hold-back ligand exploration (DESIGN.md section 13.5) -- "
          "sensitivity study, not a recommendation", "", f"*regime: {status}*", "",
          "Reference regime: the TODGA-only candidate of largest purity x recovery in the "
          f"{n}-row LHS (n_ext {int(ref['n_ext'])}, n_scr {int(ref['n_scr'])}, n_str "
          f"{int(ref['n_str'])}, O/A {float(ref['oa_ext']):.2f}, purity "
          f"{float(ref['purity_mol']):.3f}, recovery {float(ref['recovery_from_feed']):.3f}, "
          f"flags {ref['flags']}).", "",
          "The complexant enters the feed (`feed`) or the scrub liquor only (`scrub_only`) at "
          f"{float(comp.concentration.value)} M total; `delta_log_beta = log beta_Pr - log "
          "beta_Nd` is the beta contrast the hold-back ligand would need; the consumption uses "
          "the assumed regeneration fraction "
          f"{comp.regeneration_fraction.value} (range {comp.regeneration_fraction.range}). "
          "Until a sourced beta set exists (open item U3) no number here is a recommendation.",
          "", markdown_table(sweep), ""]
    (out_dir / "todga_exploration.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    outputs.append(out_dir / "todga_exploration.md")
    return outputs


if __name__ == "__main__":
    raise SystemExit(main())
