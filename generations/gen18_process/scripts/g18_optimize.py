"""Script 9 of DESIGN.md section 11: local regime optimisation of one system.

For ``--system-id`` with the feed and product specification of the case files, build the
system model (``params_source`` per the pre-registered decision by default), the design space of
section 9.1 (config defaults, the parameter set's applicability domain, ``--widen`` overrides),
run ``lhs_pareto`` (section 9.2) and write ``results/optimize/<system_id>/lhs.csv`` (every
candidate), ``front_all.csv``, ``front_in_domain_only.csv``, ``knees.csv`` (epsilon-constraint
knees over the spec grid), ``SUMMARY.md`` and ``manifest.json``.  ``--bo`` adds the gated GP-BO
of section 9.3 (``bo.csv``; a gated call writes ``bo_gate.txt`` and evaluates nothing).  Every
table carries its regime and the status of the parameters; a literature entry with
``ASSUMED_PLACEHOLDER`` parameters needs ``--allow-placeholders``.

Usage (from the repository root, one process):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_optimize.py \\
        --system-id <id> --feed cases/prnd_feed.json --spec cases/prnd_spec.json \\
        [--n-lhs 2000] [--seed 18] [--objective consumption|cost] [--bo] \\
        [--params-source auto|literature|fitted|nearest] [--widen strip_acid_M 0.5 6] ...

No wall-clock value is written.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g18_case_prnd import base_spec_for, feed_stream, load_json, sourced_value  # noqa: E402
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc import optimize as OPT  # noqa: E402
from gen18proc.dmodel import build_system_model  # noqa: E402
from gen18proc.metrics import Prices  # noqa: E402
from gen18proc.report import markdown_table, write_manifest, write_table  # noqa: E402
from gen18proc.systems import iter_sourced, load_system  # noqa: E402
from gen18proc.types import ModelBuildError, ProvStatus  # noqa: E402


def parameter_status(entry, system) -> str:
    placeholders = any(s.status == ProvStatus.ASSUMED and s.value is not None
                       for _, s in iter_sourced(entry))
    return (f"{system.params_source} (origin {entry.origin}"
            + ("; ASSUMED_PLACEHOLDER values" if placeholders else "") + ")")


def build_space(entry, system, args: argparse.Namespace) -> OPT.DesignSpace:
    ligand = system.ligands[0]
    dom = entry.applicability.get(f"{ligand}|{system.bands.get(ligand, '20-30C')}")
    widen: dict[str, tuple[float, float]] = {}
    if args.max_stages:
        for k in ("n_ext", "n_scr", "n_str"):
            widen[k] = (1.0, float(args.max_stages))
        widen["feed_stage_offset"] = (0.0, float(args.max_stages - 1))
        widen["scrub_return_offset"] = (0.0, float(args.max_stages - 1))
    for name, lo, hi in args.widen or []:
        widen[name] = (float(lo), float(hi))
    fixed = {"strip_anion_M": 0.0, "feed_dilution": 0.0, "f_bleed": 0.0}
    if system.complexant_model is None:
        fixed["scrub_complexant_M"] = 0.0
    for name, value in args.fix or []:
        fixed[name] = float(value)
    return OPT.DesignSpace.from_config(family=entry.family, domain=dom, ligand=ligand,
                                       widen=widen, fixed=fixed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--feed", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--n-lhs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--objective", choices=sorted(OPT.OBJECTIVES), default="consumption")
    ap.add_argument("--bo", action="store_true")
    ap.add_argument("--bo-n-init", type=int, default=64)
    ap.add_argument("--bo-n-iter", type=int, default=30)
    ap.add_argument("--bo-batch", type=int, default=4)
    ap.add_argument("--params-source", default="auto",
                    choices=("auto", "literature", "fitted", "nearest"))
    ap.add_argument("--widen", nargs=3, action="append", metavar=("NAME", "LO", "HI"))
    ap.add_argument("--fix", nargs=2, action="append", metavar=("NAME", "VALUE"))
    ap.add_argument("--max-stages", type=int, default=None)
    ap.add_argument("--solver-max-newton", type=int, default=40)
    ap.add_argument("--solver-max-sweeps", type=int, default=10)
    ap.add_argument("--allow-placeholders", action="store_true")
    ap.add_argument("--out-dir", default=str(paths.RESULTS_OPTIMIZE_DIR))
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    args = ap.parse_args()
    feed_path = paths.G18_ROOT / args.feed if not Path(args.feed).is_absolute() else Path(args.feed)
    spec_path = paths.G18_ROOT / args.spec if not Path(args.spec).is_absolute() else Path(args.spec)
    feed_json, spec_json = load_json(feed_path), load_json(spec_path)
    entry = load_system(Path(args.systems_dir) / f"{args.system_id}.json")
    target = str(spec_json["target"])
    impurities = [str(m) for m in spec_json["impurities"]]
    purity_grid = [float(v) for v in spec_json["purity_min_grid"]["values"]]
    recovery_grid = [float(v) for v in spec_json["recovery_min_grid"]["values"]]
    loosest = {"purity_min": min(purity_grid), "recovery_min": min(recovery_grid)}
    if any(s.status == ProvStatus.ASSUMED and s.value is not None
           for _, s in iter_sourced(entry)) and not args.allow_placeholders:
        raise SystemExit(f"refused: {entry.system_id} carries ASSUMED_PLACEHOLDER parameters; "
                         "pass --allow-placeholders")
    feed = feed_stream(feed_json)
    system = build_system_model(entry, feed_anion=feed_json["anion"],
                                temperature_C=sourced_value(feed_json["temperature_C"]),
                                params_source=args.params_source,
                                feed_metals=(target, *impurities))
    if isinstance(system, ModelBuildError):
        raise SystemExit(f"refused: {entry.system_id}: {system.reason}")
    ligand = system.ligands[0]
    scale = float(getattr(system.dmodels[ligand], "ligand_scale", 1.0) or 1.0)
    conc = entry.organic_ligands[0].concentration.value
    lt0 = (float(conc) if conc else 0.1) / scale
    base = base_spec_for(feed, target, ligand, lt0)
    space = build_space(entry, system, args)
    status = parameter_status(entry, system)
    regime = {"cohort": f"{entry.system_id} ({entry.name}); feed {feed_path.name} "
                        f"(feed_status {feed_json.get('feed_status')}); spec {spec_path.name}",
              "holdout": "none (computed regimes)", "averaging_unit": "candidate",
              "status_of_parameters": status}
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = paths.G18_ROOT / out_dir
    out_dir = out_dir / entry.system_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = Prices.load()
    solver_kwargs = {"max_newton": args.solver_max_newton, "max_sweeps": args.solver_max_sweeps}
    print(f"{entry.system_id}: D source {system.params_source}; space {space.variables}")
    df = OPT.lhs_pareto(space, base, system, target, impurities, loosest, prices, n=args.n_lhs,
                        seed=args.seed, objective=args.objective, solver_kwargs=solver_kwargs)
    outputs = [write_table(df, out_dir / "lhs.csv", regime=regime, regime_table=True)]
    fronts = OPT.pareto_fronts(df)
    for name, fr in fronts.items():
        outputs.append(write_table(fr, out_dir / f"front_{name}.csv", regime=regime,
                                   regime_table=True))
    knees = OPT.epsilon_knees(df, purity_grid, recovery_grid,
                              objective_col=OPT.OBJECTIVES[args.objective][2][0])
    outputs.append(write_table(knees, out_dir / "knees.csv", regime=regime, regime_table=True))
    bo_note = ""
    if args.bo:
        res = OPT.gp_bo(space, base, system, target, impurities, loosest, prices,
                        n_init=args.bo_n_init, n_iter=args.bo_n_iter, batch=args.bo_batch,
                        seed=args.seed, objective=args.objective, solver_kwargs=solver_kwargs)
        if res.status == "done":
            outputs.append(write_table(res.table, out_dir / "bo.csv", regime={
                **regime, "averaging_unit": "candidate (GP-BO, ParEGO; never a headline)"},
                regime_table=True))
            bo_note = f"GP-BO ran ({len(res.table)} evaluations): {res.reason}"
        else:
            gate = out_dir / "bo_gate.txt"
            gate.write_text(f"GP-BO {res.status}: {res.reason}\n", encoding="utf-8")
            outputs.append(gate)
            bo_note = f"GP-BO {res.status}: {res.reason}"
    # summary
    conv = df.loc[df["status"].str.startswith("converged")]
    on_spec = conv.loc[conv["on_spec"].fillna(False).astype(bool)]
    top = on_spec.sort_values("consumption_index").head(3)
    cols = ["front", "regime_status", "n_ext", "n_scr", "n_str", "oa_ext", "s_over_a", "w_over_a",
            "scrub_acid_M", "strip_acid_M", "ligand_total_M", "saponification_degree",
            "purity_mol", "recovery_from_feed", "consumption_index", "n_stages_total", "flags"]
    cols = [c for c in cols if c in df.columns]
    lines = [f"# Regime search for {entry.system_id} ({entry.name})", "",
             f"*regime: cohort={regime['cohort']}; holdout={regime['holdout']}; "
             f"averaging_unit=candidate; status_of_parameters={status}*", "",
             f"LHS n = {args.n_lhs}, seed {args.seed}, objective {args.objective}; converged "
             f"{df.attrs['n_converged']}, failed {df.attrs['n_failed']}, invalid_spec "
             f"{df.attrs['n_invalid_spec']}; on-spec (loosest cell {loosest}) {len(on_spec)}; "
             f"front sizes: all {len(fronts['all'])}, in_domain_only "
             f"{len(fronts['in_domain_only'])}.", "",
             "Design space: " + json.dumps(space.to_json(), sort_keys=True), "",
             "## Top three on-spec regimes by consumption index", "",
             markdown_table(top[cols]) if len(top) else "(none on spec)", "",
             "## Epsilon-constraint knees (min consumption index per spec cell)", "",
             markdown_table(knees[["subset", "purity_min", "recovery_min", "reachable",
                                   "n_feasible", "n_stages_total", "consumption_index",
                                   "regime_status"]]), "",
             "## First Pareto front, in-domain only", "",
             markdown_table(fronts["in_domain_only"][cols]) if len(fronts["in_domain_only"])
             else "(no in-domain candidate on the first front)", "",
             "## First Pareto front, all candidates (OOD rows carry their flags)", "",
             markdown_table(fronts["all"][cols]) if len(fronts["all"]) else "(empty)", ""]
    if bo_note:
        lines += ["## GP-BO", "", bo_note, ""]
    summ = out_dir / "SUMMARY.md"
    summ.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    outputs.append(summ)
    write_manifest(out_dir / "manifest.json", outputs,
                   [feed_path, spec_path, Path(args.systems_dir) / f"{entry.system_id}.json",
                    paths.CONFIG_DIR / "design_spaces.json", paths.CONFIG_DIR / "prices.json"],
                   args.seed, arguments=vars(args),
                   extra={"params_source": system.params_source, "space": space.to_json(),
                          "n_failed": df.attrs["n_failed"],
                          "n_invalid_spec": df.attrs["n_invalid_spec"]})
    best = top.iloc[0] if len(top) else None
    print(f"wrote {len(outputs)} files under {out_dir}; on-spec {len(on_spec)}/{len(df)}"
          + (f"; best consumption index {float(best['consumption_index']):.4g} at "
             f"{int(best['n_stages_total'])} stages" if best is not None else "")
          + (f"; {bo_note}" if bo_note else ""))
    return 0 if not math.isnan(float(len(df))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
