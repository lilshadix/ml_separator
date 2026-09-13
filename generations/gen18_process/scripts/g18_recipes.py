"""Script 10 of DESIGN.md section 11: the recipes driver (section 9.4).

For every database system whose ``medium.anion`` equals the feed anion and which has a usable
parameter set (``build_system_model`` is not an error: corpus systems through the
pre-registered D source -- fitted M1 when adopted, else ``NearestConditionD`` -- literature
systems through their placeholder blocks), run ``lhs_pareto`` and write
``results/recipes/<feed>/<system_id>.csv`` plus ``SUMMARY.md``: per system family the top three
on-spec regimes (by consumption index), the Pareto knee of the consistency cell, flags, and the
parameter status of that system; systems without parameters are listed as ``not_parameterised``
with the reason.  Every table carries the regime tag (section 10.6).

Usage (from the repository root, one process):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_recipes.py \\
        --feed cases/<feed>.json --spec cases/<spec>.json [--systems all|<ids>] \\
        [--n-lhs 1000] [--seed 18] [--max-stages 30]

No wall-clock value is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g18_case_prnd import base_spec_for, feed_stream, load_json, sourced_value  # noqa: E402
from g18_optimize import build_space, parameter_status  # noqa: E402
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc import optimize as OPT  # noqa: E402
from gen18proc.dmodel import build_system_model  # noqa: E402
from gen18proc.metrics import Prices  # noqa: E402
from gen18proc.report import markdown_table, write_manifest, write_table  # noqa: E402
from gen18proc.systems import load_registry, load_system  # noqa: E402
from gen18proc.types import ModelBuildError, SystemValidationError  # noqa: E402

SUMMARY_COLS = ["system_id", "front", "regime_status", "n_ext", "n_scr", "n_str", "oa_ext",
                "s_over_a", "w_over_a", "scrub_acid_M", "strip_acid_M", "ligand_total_M",
                "saponification_degree", "purity_mol", "recovery_from_feed",
                "consumption_index", "n_stages_total", "flags"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feed", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--systems", nargs="*", default=["all"])
    ap.add_argument("--n-lhs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--objective", choices=sorted(OPT.OBJECTIVES), default="consumption")
    ap.add_argument("--max-stages", type=int, default=None)
    ap.add_argument("--solver-max-newton", type=int, default=40)
    ap.add_argument("--solver-max-sweeps", type=int, default=10)
    ap.add_argument("--widen", nargs=3, action="append", metavar=("NAME", "LO", "HI"))
    ap.add_argument("--fix", nargs=2, action="append", metavar=("NAME", "VALUE"))
    ap.add_argument("--out-dir", default=str(paths.RESULTS_RECIPES_DIR))
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    args = ap.parse_args()
    feed_path = paths.G18_ROOT / args.feed if not Path(args.feed).is_absolute() else Path(args.feed)
    spec_path = paths.G18_ROOT / args.spec if not Path(args.spec).is_absolute() else Path(args.spec)
    feed_json, spec_json = load_json(feed_path), load_json(spec_path)
    anion = str(feed_json["anion"])
    target = str(spec_json["target"])
    impurities = [str(m) for m in spec_json["impurities"]]
    purity_grid = [float(v) for v in spec_json["purity_min_grid"]["values"]]
    recovery_grid = [float(v) for v in spec_json["recovery_min_grid"]["values"]]
    loosest = {"purity_min": min(purity_grid), "recovery_min": min(recovery_grid)}
    cons = spec_json.get("consistency_cell", {})
    cons_cell = (float(cons.get("purity_min", max(purity_grid))),
                 float(cons.get("recovery_min", max(recovery_grid))))
    systems_dir = Path(args.systems_dir)
    registry = load_registry(systems_dir)
    if args.systems != ["all"]:
        registry = registry.loc[registry["system_id"].isin(args.systems)]
    registry = registry.loc[registry["acid_class"] == anion]   # anion == acid class vocabulary
    feed = feed_stream(feed_json)
    temperature = sourced_value(feed_json["temperature_C"])
    prices = Prices.load()
    solver_kwargs = {"max_newton": args.solver_max_newton, "max_sweeps": args.solver_max_sweeps}
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = paths.G18_ROOT / out_dir
    out_dir = out_dir / feed_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    dec_path = paths.RESULTS_EVAL_DIR / "decision.json"
    adopted = bool(load_json(dec_path).get("m1_adopted", False)) if dec_path.exists() else False
    outputs: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[pd.DataFrame] = []
    knee_rows: list[dict[str, Any]] = []
    for r in registry.itertuples():
        sid = str(r.system_id)
        try:
            entry = load_system(systems_dir / f"{sid}.json")
        except SystemValidationError as exc:
            summary_rows.append({"system_id": sid, "family": r.family, "status":
                                 "not_parameterised", "reason": f"entry invalid: {exc}"[:200]})
            continue
        if entry.medium.anion != anion:
            continue
        system = build_system_model(entry, feed_anion=anion, temperature_C=temperature,
                                    params_source="auto", decision_adopted=adopted,
                                    feed_metals=(target, *impurities))
        if isinstance(system, ModelBuildError):
            summary_rows.append({"system_id": sid, "family": entry.family, "name": entry.name,
                                 "status": "not_parameterised", "reason": system.reason})
            print(f"{sid}: not_parameterised ({system.reason})")
            continue
        ligand = system.ligands[0]
        scale = float(getattr(system.dmodels[ligand], "ligand_scale", 1.0) or 1.0)
        conc = entry.organic_ligands[0].concentration.value
        base = base_spec_for(feed, target, ligand, (float(conc) if conc else 0.1) / scale)
        space = build_space(entry, system, args)
        status = parameter_status(entry, system)
        regime = {"cohort": f"{sid} ({entry.name}); feed {feed_path.name} (feed_status "
                            f"{feed_json.get('feed_status')})",
                  "holdout": "none (computed regimes)", "averaging_unit": "candidate",
                  "status_of_parameters": status}
        df = OPT.lhs_pareto(space, base, system, target, impurities, loosest, prices,
                            n=args.n_lhs, seed=args.seed, objective=args.objective,
                            solver_kwargs=solver_kwargs)
        outputs.append(write_table(df, out_dir / f"{sid}.csv", regime=regime, regime_table=True))
        conv = df.loc[df["status"].str.startswith("converged")]
        on_spec = conv.loc[conv["on_spec"].fillna(False).astype(bool)]
        top = on_spec.sort_values("consumption_index").head(3).copy()
        if len(top):
            top.insert(0, "family", entry.family)
            top_rows.append(top[["family", *[c for c in SUMMARY_COLS if c in top.columns]]])
        knees = OPT.epsilon_knees(df, [cons_cell[0]], [cons_cell[1]])
        for k in knees.itertuples():
            knee_rows.append({"family": entry.family, "system_id": sid, "subset": k.subset,
                              "purity_min": k.purity_min, "recovery_min": k.recovery_min,
                              "reachable": k.reachable, "n_feasible": k.n_feasible,
                              "n_stages_total": getattr(k, "n_stages_total", None),
                              "consumption_index": getattr(k, "consumption_index", None),
                              "regime_status": getattr(k, "regime_status", None),
                              "flags": getattr(k, "flags", None)})
        summary_rows.append({"system_id": sid, "family": entry.family, "name": entry.name,
                             "status": "parameterised", "params_source": system.params_source,
                             "status_of_parameters": status, "n_lhs": len(df),
                             "n_converged": df.attrs["n_converged"],
                             "n_failed": df.attrs["n_failed"],
                             "n_invalid_spec": df.attrs["n_invalid_spec"],
                             "n_on_spec_loosest": int(len(on_spec)),
                             "n_front_all": int((df["front"] == 1).sum()),
                             "n_front_in_domain": int((df["front_in_domain"] == 1).sum()),
                             "reason": ""})
        print(f"{sid} ({entry.family}, {system.params_source}): converged "
              f"{df.attrs['n_converged']}/{len(df)}, on-spec {len(on_spec)}")
    summary = pd.DataFrame(summary_rows)
    outputs.append(write_table(summary, out_dir / "systems_summary.csv", regime={
        "cohort": f"database systems with medium.anion == {anion}", "holdout": "none",
        "averaging_unit": "system", "status_of_parameters": "per row"}, regime_table=True))
    tops = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame(
        columns=["family", *SUMMARY_COLS])
    knees_df = pd.DataFrame(knee_rows)
    outputs.append(write_table(tops, out_dir / "top_regimes.csv", regime={
        "cohort": f"database systems with medium.anion == {anion}", "holdout": "none",
        "averaging_unit": "candidate", "status_of_parameters": "see systems_summary.csv"},
        regime_table=True))
    outputs.append(write_table(knees_df, out_dir / "knees.csv", regime={
        "cohort": f"database systems with medium.anion == {anion}", "holdout": "none",
        "averaging_unit": "system x consistency cell",
        "status_of_parameters": "see systems_summary.csv"}, regime_table=True))
    lines = [f"# Recipes for feed {feed_path.name} (anion {anion}), spec {spec_path.name}", "",
             f"*regime: cohort=database systems with medium.anion == {anion}; holdout=none "
             f"(computed regimes); averaging_unit=candidate; status_of_parameters=per system "
             f"(column status_of_parameters); LHS n = {args.n_lhs}, seed {args.seed}; on-spec = "
             f"loosest cell {loosest}; knee = consistency cell {cons_cell}*", "",
             f"Systems considered: {len(summary)}; parameterised: "
             f"{int((summary['status'] == 'parameterised').sum()) if len(summary) else 0}; "
             f"not_parameterised: "
             f"{int((summary['status'] == 'not_parameterised').sum()) if len(summary) else 0}. "
             f"Pre-registered decision m1_adopted = {adopted}.", ""]
    if len(summary):
        for family, g in summary.groupby("family", sort=True):
            lines += [f"## Family {family}", "",
                      markdown_table(g[[c for c in ("system_id", "name", "status", "params_source",
                                                    "status_of_parameters", "n_converged",
                                                    "n_failed", "n_on_spec_loosest",
                                                    "n_front_in_domain", "reason")
                                        if c in g.columns]]), ""]
            ft = tops.loc[tops["family"] == family] if len(tops) else tops
            lines += ["Top three on-spec regimes per system (by consumption index):", "",
                      markdown_table(ft) if len(ft) else "(none on spec)", ""]
            fk = knees_df.loc[knees_df["family"] == family] if len(knees_df) else knees_df
            lines += ["Pareto knee of the consistency cell:", "",
                      markdown_table(fk) if len(fk) else "(none)", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    outputs.append(out_dir / "SUMMARY.md")
    write_manifest(out_dir / "manifest.json", outputs,
                   [feed_path, spec_path, systems_dir / "registry.json",
                    paths.CONFIG_DIR / "design_spaces.json", paths.CONFIG_DIR / "prices.json"],
                   args.seed, arguments=vars(args),
                   extra={"m1_adopted": adopted, "n_systems": int(len(summary))})
    print(f"wrote {len(outputs)} files under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
