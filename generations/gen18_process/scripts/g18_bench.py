"""Script 7 of DESIGN.md section 11: wall-clock timing of the cascade chain.

Measures, on the synthetic fixtures of ``gen18proc.testsystems`` (no database access), the
median and quartiles in milliseconds of: one stage solve (``equilibrium.solve_stage``, cold and
warm-started), the Kremser initial solve, one residual evaluation and one Jacobian assembly of
the cascade Newton, one full cascade by the Newton path and by successive substitution, and one
LHS evaluation (``solve_cascade`` + ``compute_metrics``) for the reference cascade N = 6/3/3
(six extraction, three scrub, three strip stages), M = 2 metals, K = 1 ligand of the
cation-exchange fixture (84 unknowns), plus the same cascade on the solvating fixture.  From the
median LHS evaluation it projects the case-study budget of DESIGN.md section 11.3 (64 draws x 2
systems x 1000 LHS rows).

Writes ``results/bench/timing.json`` — **the only results file that may contain wall-clock
values** (DESIGN.md section 1.5) — and ``results/bench/manifest.json`` (no wall-clock values).
No test asserts a timing.

Usage (from the repository root):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_bench.py [--n 50] [--seed 18]
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import DEFAULT_SEED, paths, testsystems as ts  # noqa: E402
from gen18proc.cascade import (  # noqa: E402
    _kremser_u0,
    _Newton,
    _Problem,
    _refine_init,
    kremser_init,
    solve_cascade,
)
from gen18proc.equilibrium import solve_stage  # noqa: E402
from gen18proc.metrics import Prices, compute_metrics  # noqa: E402
from gen18proc.report import write_manifest  # noqa: E402
from gen18proc.types import AqStream, CascadeSpec  # noqa: E402

OUT_DIR = paths.RESULTS_BENCH_DIR
TIMING_JSON = OUT_DIR / "timing.json"
MANIFEST_JSON = OUT_DIR / "manifest.json"
REFERENCE = (6, 3, 3)


def reference_specs() -> dict[str, tuple[CascadeSpec, Any]]:
    """The reference cascades: cation-exchange and solvating fixtures, N = 6/3/3, M = 2, K = 1."""
    ce = ts.two_metal_cation_exchange()
    sv = ts.two_metal_solvating()
    scrub_ce = AqStream(0.3, {"Pr": 0.0, "Nd": 0.0}, 0.2, 0.2, 0.0, 0.0)
    strip_ce = AqStream(0.5, {"Pr": 0.0, "Nd": 0.0}, 3.0, 3.0, 0.0, 0.0)
    spec_ce = CascadeSpec(*REFERENCE, ts.feed_prnd(), scrub_ce, strip_ce, 1.0,
                          {ts.CE_LIGAND: ts.CE_LT_DIMER}, 0.0, target="Nd")
    scrub_sv = AqStream(0.3, {"Pr": 0.0, "Nd": 0.0}, 1.0, 1.0, 0.0, 0.0)
    strip_sv = AqStream(0.5, {"Pr": 0.0, "Nd": 0.0}, 0.01, 0.01, 0.0, 0.0)
    spec_sv = CascadeSpec(*REFERENCE, ts.feed_prnd_nitrate(), scrub_sv, strip_sv, 1.0,
                          {ts.SOLV_LIGAND: ts.SOLV_LT}, 0.0, target="Nd")
    return {"cation_exchange": (spec_ce, ce), "solvating": (spec_sv, sv)}


def timed(fn: Callable[[], Any], n: int) -> dict[str, float]:
    """Median / quartiles / min of ``n`` repeats of ``fn`` in milliseconds (one warm-up call)."""
    fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    q = statistics.quantiles(samples, n=4) if len(samples) >= 2 else [samples[0]] * 3
    return {"median_ms": float(statistics.median(samples)), "p25_ms": float(q[0]),
            "p75_ms": float(q[2]), "min_ms": float(min(samples)), "n": int(n)}


def bench_one(name: str, spec: CascadeSpec, system: Any, n: int, prices: Prices,
              ) -> dict[str, Any]:
    prob = _Problem(spec, system)
    out: dict[str, Any] = {
        "n_stages": prob.top.n_stages, "n_ext": spec.n_ext, "n_scr": spec.n_scr,
        "n_str": spec.n_str, "n_metals": prob.M, "n_ligands": prob.K, "n_unknowns": prob.n,
    }
    feed = spec.feed
    lean = ts.lean_organic(system)
    out["stage_solve_cold"] = timed(lambda: solve_stage(feed, lean, system), n)
    warm = solve_stage(feed, lean, system)[2]
    out["stage_solve_warm"] = timed(lambda: solve_stage(feed, lean, system, warm=warm), n)
    out["kremser_init"] = timed(lambda: kremser_init(spec, system), n)
    u0 = _kremser_u0(prob)
    out["init_refine"] = timed(lambda: _refine_init(prob, u0), n)
    u1 = _refine_init(prob, u0)
    nt = _Newton(prob, u1)
    w1 = nt.to_w(u1)
    ev = nt.evaluate(w1)
    out["newton_residual_eval"] = timed(lambda: nt.evaluate(w1), n)
    out["newton_jacobian_assembly"] = timed(lambda: nt.jacobian_scaled(ev), n)
    out["cascade_newton"] = timed(lambda: solve_cascade(spec, system, method="newton"), n)
    res = solve_cascade(spec, system)
    out["cascade_status"] = res.status
    out["cascade_newton_iterations"] = res.iterations
    out["cascade_balance_rel_max"] = res.balance_rel_max
    n_ss = max(3, n // 10)
    out["cascade_ss"] = timed(lambda: solve_cascade(spec, system, method="ss"), n_ss)
    res_ss = solve_cascade(spec, system, method="ss")
    out["cascade_ss_sweeps"] = res_ss.iterations
    out["cascade_ss_status"] = res_ss.status
    out["lhs_evaluation"] = timed(
        lambda: compute_metrics(solve_cascade(spec, system), spec, system, "Nd", ["Pr"], prices),
        n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50, help="repeats per timed item (default 50)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    if args.n < 1:
        ap.error("--n must be >= 1")
    prices = Prices.load()
    results: dict[str, Any] = {}
    for name, (spec, system) in reference_specs().items():
        results[name] = bench_one(name, spec, system, args.n, prices)
        r = results[name]
        print(f"{name}: stage {r['stage_solve_cold']['median_ms']:.2f} ms, "
              f"newton cascade {r['cascade_newton']['median_ms']:.1f} ms "
              f"({r['cascade_newton_iterations']} it, {r['cascade_status']}), "
              f"ss cascade {r['cascade_ss']['median_ms']:.0f} ms "
              f"({r['cascade_ss_sweeps']} sweeps), "
              f"lhs evaluation {r['lhs_evaluation']['median_ms']:.1f} ms")
    ref = results["cation_exchange"]["lhs_evaluation"]["median_ms"]
    projection = {
        "reference": "cation_exchange lhs_evaluation median",
        "per_1000_cascades_s": ref,
        "case_study_64_draws_x_2_systems_x_1000_lhs_min": 64 * 2 * 1000 * ref / 1e3 / 60.0,
        "design_expectation_ms": [5.0, 15.0],
        "within_design_expectation": bool(5.0 <= ref <= 15.0),
    }
    payload = {
        "schema": "gen18.bench.1",
        "note": ("wall-clock milliseconds; this is the only results file that may carry "
                 "wall-clock values (DESIGN.md section 1.5); no test asserts them"),
        "reference_cascade": {"n_ext": REFERENCE[0], "n_scr": REFERENCE[1],
                              "n_str": REFERENCE[2], "n_metals": 2, "n_ligands": 1},
        "seed": int(args.seed),
        "repeats": int(args.n),
        "platform": {"python": platform.python_version(), "numpy": np.__version__,
                     "machine": platform.machine(), "system": platform.system()},
        "results": results,
        "projection": projection,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TIMING_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8", newline="\n")
    write_manifest(MANIFEST_JSON, [TIMING_JSON],
                   [paths.CONFIG_DIR / "prices.json",
                    paths.G18_ROOT / "gen18proc" / "cascade.py",
                    paths.G18_ROOT / "gen18proc" / "equilibrium.py",
                    paths.G18_ROOT / "gen18proc" / "metrics.py"],
                   args.seed, arguments=vars(args))
    print(f"wrote {TIMING_JSON} and {MANIFEST_JSON}; projected case study "
          f"{projection['case_study_64_draws_x_2_systems_x_1000_lhs_min']:.0f} min at "
          f"{ref:.1f} ms per evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
