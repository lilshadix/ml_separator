"""Targeted search for the Pr/Nd spec cell, after the random LHS of the case found none.

**Exploratory, not pre-registered, and on the same PLACEHOLDER parameters as the case study.**

`scripts/g18_case_prnd.py` searches each parameter draw with a Latin hypercube over ~17 design
variables and reported *no* spec cell reached by either system in any of 64 draws -- not even the
loosest, (purity 0.95, recovery 0.80).  That verdict is not credible as chemistry: the Fenske
minimum at total reflux for that cell at SF 1.4 is only about **10 theoretical stages**, far inside
the 40 + 40 the search allowed, so a feasible region must exist and a 250-point random sample over
17 dimensions simply missed it.  This script tests that explanation directly by searching a small
structured grid instead of sampling randomly: it fixes the circuit shape a separations engineer
would choose (extraction sized so the target is well but not completely extracted, a modest scrub
that returns the impurity, a strong strip) and scans stage counts, O/A, scrub flow and scrub
acidity.

Whichever way it falls, it is reported: if a regime meets the cell the case's "not reachable" is a
search artefact; if nothing does across the structured grid either, that is evidence the window
really is closed for these placeholder parameters.

Run from the repository root (one process, a few minutes):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_case_targeted.py \\
        [--system sys_29976921e156a0a0] [--n-ext-max 30] [--seed 18]

Writes `results/case_prnd/targeted/<system_id>.csv` and `TARGETED.md`.  No wall-clock value is
written.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc.cascade import solve_cascade  # noqa: E402
from gen18proc.metrics import Prices, compute_metrics  # noqa: E402
from gen18proc.optimize import candidate_spec  # noqa: E402
from gen18proc.report import markdown_table, write_manifest, write_table  # noqa: E402
from gen18proc.systems import load_system  # noqa: E402
from gen18proc.types import AqStream, CascadeSpec, ModelBuildError  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g18_case_prnd import (  # noqa: E402
    build, draw_table, draw_to_parameters, feed_stream, load_json, median_draw,
)

CELLS = [(0.95, 0.80), (0.97, 0.85), (0.99, 0.90)]
REGIME = {
    "cohort": "Pr/Nd case, PC88A and Cyanex 272 literature placeholder entries",
    "holdout": "none (computed regimes; not validation)",
    "averaging_unit": "candidate regime",
    "status_of_parameters": "PLACEHOLDER_PARAMETERS (median draw); EXPLORATORY, not pre-registered",
}


def fenske_n_min(purity: float, recovery: float, alpha: float, nd: float, pr: float) -> float:
    """Theoretical stages at total reflux for the binary split (a bound, not a design)."""
    nd_p = recovery * nd
    pr_p = nd_p * (1.0 - purity) / purity
    nd_w, pr_w = nd - nd_p, pr - pr_p
    if min(nd_p, pr_p, nd_w, pr_w) <= 0:
        return math.nan
    x_d = nd_p / (nd_p + pr_p)
    x_w = nd_w / (nd_w + pr_w)
    return math.log((x_d / (1 - x_d)) * ((1 - x_w) / x_w)) / math.log(alpha)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", default="sys_29976921e156a0a0")
    ap.add_argument("--feed", default="cases/prnd_feed.json")
    ap.add_argument("--n-ext-max", type=int, default=30)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    args = ap.parse_args()

    feed_json = load_json(paths.G18_ROOT / args.feed)
    feed = feed_stream(feed_json)
    entry = load_system(Path(args.systems_dir) / f"{args.system}.json")
    ligand0 = entry.organic_ligands[0].name
    draws = draw_table(entry, ligand0, 64, args.seed)
    draw = draw_to_parameters(ligand0, median_draw(draws), entry)
    system = build(entry, feed_json["anion"], 25.0, draw, feed_metals=("Nd", "Pr"))
    if isinstance(system, ModelBuildError):
        raise SystemExit(f"{args.system}: {system.reason}")
    ligand = system.ligands[0]
    prices = Prices.load()

    zeros = {m: 0.0 for m in feed.metals}
    base = CascadeSpec(6, 3, 3, feed,
                       AqStream(0.3 * feed.flow_L_h, dict(zeros), 0.2, 0.2, 0.0, 0.0),
                       AqStream(0.5 * feed.flow_L_h, dict(zeros), 3.0, 3.0, 0.0, 0.0),
                       feed.flow_L_h, {ligand: 0.4}, 0.0, target="Nd")

    # A structured grid, not a random sample: the circuit shape is fixed by engineering judgement
    # and only the sizes are scanned.  About 3000 cascades.
    grid = {
        "n_ext": [6, 12, 20, min(24, args.n_ext_max)],
        "n_scr": [4, 10, 16],
        "n_str": [4],
        "oa_ext": [0.5, 1.0, 2.0],
        "s_over_a": [0.05, 0.15, 0.4],
        "scrub_acid_M": [0.05, 0.2, 0.5],
        "strip_acid_M": [4.0],
        "ligand_total_M": [0.8],
        "w_over_a": [0.3],
        # A cation-exchange extractant releases 3 H+ per Ln3+: a 0.1 M feed self-acidifies from
        # pH 2 to about 0.3 M H+ and extraction stops (log D falls by 3 log10(31) ~ 4.5).  Real
        # PC88A circuits are therefore partially saponified, and this is the variable that decides
        # whether the cell is reachable at all -- so it is scanned, not fixed.
        "saponification_degree": [0.0, 0.2, 0.35, 0.5, 0.65],
    }
    names = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in names)):
        values = dict(zip(names, combo))
        try:
            spec, extras = candidate_spec(values, base, system, "Nd")
            res = solve_cascade(spec, system, max_newton=40, max_sweeps=10)
            m = compute_metrics(res, spec, system, "Nd", ("Pr",), prices,
                                feed_dilution_L_h=float(extras.get("feed_dilution_L_h", 0.0)))
        except ValueError:
            continue
        if res.status.startswith("converged") and m.purity_mol is not None:
            rows.append({**values, "status": res.status, "purity_mol": m.purity_mol,
                         "recovery_from_feed": m.recovery_from_feed,
                         "n_stages_total": m.n_stages_total,
                         "regime_status": m.regime_status,
                         "flags": "|".join(sorted(f.value for f in m.flags))})
    df = pd.DataFrame(rows)
    out_dir = paths.RESULTS_CASE_PRND_DIR / "targeted"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{args.system}.csv"
    write_table(df.sort_values(["purity_mol", "recovery_from_feed"], ascending=False),
                csv, regime=REGIME, regime_table=True)

    alpha = 1.4
    nd = feed.metals.get("Nd", 0.075) / (feed.metals.get("Nd", 0.075)
                                         + feed.metals.get("Pr", 0.025))
    lines = ["# Targeted search for the Pr/Nd spec cells (exploratory)", "",
             f"*regime: {REGIME['cohort']}; system {args.system}; "
             f"{REGIME['status_of_parameters']}*", "",
             f"A structured grid of {len(rows)} converged regimes (of "
             f"{int(np.prod([len(v) for v in grid.values()]))} tried), against the random LHS of "
             f"`g18_case_prnd.py` which reached no cell in 64 draws.", "",
             "| purity_min | recovery_min | Fenske N_min (total reflux, SF 1.4) | reached here | "
             "best purity at that recovery | stages of the best |", "|---|---|---|---|---|---|"]
    best_rows = []
    for pur, rec in CELLS:
        n_min = fenske_n_min(pur, rec, alpha, nd, 1 - nd)
        ok = df[(df["purity_mol"] >= pur) & (df["recovery_from_feed"] >= rec)] if len(df) else df
        at_rec = df[df["recovery_from_feed"] >= rec] if len(df) else df
        best_p = at_rec["purity_mol"].max() if len(at_rec) else math.nan
        stages = (ok.sort_values("n_stages_total")["n_stages_total"].iloc[0] if len(ok)
                  else math.nan)
        lines.append(f"| {pur} | {rec} | {n_min:.1f} | **{'yes' if len(ok) else 'no'}** "
                     f"({len(ok)} regimes) | {best_p:.4f} | {stages if len(ok) else '-'} |")
        if len(ok):
            best_rows.append(ok.sort_values("n_stages_total").iloc[0])
    lines += ["", "## Verdict", ""]
    if best_rows:
        lines += ["The case study's \"not reachable\" is a **search artefact**: a structured grid "
                  "finds regimes meeting the cell that a 250-point Latin hypercube over 17 "
                  "variables missed. The cheapest such regimes:", "",
                  markdown_table(pd.DataFrame(best_rows)), "",
                  "These are still **placeholder-parameter** regimes and are not a "
                  "recommendation; what they establish is that the case's search, not the "
                  "chemistry, produced the empty result."]
    else:
        by_sap = df.groupby("saponification_degree").agg(
            max_recovery=("recovery_from_feed", "max"), max_purity=("purity_mol", "max"),
            n=("purity_mol", "size")).reset_index() if len(df) else pd.DataFrame()
        best_joint = df["purity_mol"].max() if len(df) else math.nan
        lines += ["No regime of the structured grid meets any cell either, but the grid locates "
                  "*why*, and it is not what the random search suggested.", "",
                  "**Saponification is decisive, as the acid balance predicts.** A cation-exchange "
                  "extractant releases 3 H+ per Ln3+, so an unsaponified 0.1 M feed self-acidifies "
                  "and extraction stalls; the grid shows exactly that, and shows it lifting:", "",
                  markdown_table(by_sap), "",
                  f"So **recovery near 1 is reachable** (from {by_sap['max_recovery'].iloc[0]:.3f} "
                  f"unsaponified to {by_sap['max_recovery'].max():.3f} saponified) and **purity "
                  f"near {by_sap['max_purity'].max():.3f} is reachable** -- but not at the same "
                  "time: the best purity anywhere on the grid at recovery >= 0.80 is "
                  f"**{best_p:.4f}**, short of the 0.95 the loosest cell needs. Over-saponification "
                  "closes it from the other side (at 0.65 the organic takes everything and purity "
                  f"falls to {by_sap['max_purity'].iloc[-1]:.3f}).", "",
                  "**What this does and does not establish.** It is *not* evidence that the "
                  "`log_k` windows are broken -- both ends of the trade-off are reachable with "
                  "them. It is evidence that the purity-recovery frontier of *this circuit shape* "
                  "stops below the cell. The Fenske minimum (about 10 stages) is a **total-reflux** "
                  "bound; at finite reflux a SF-1.4 separation needs both more stages and the "
                  "right internal reflux, and the variables that set it -- feed stage, scrub-return "
                  "stage, per-section O/A -- were held at their defaults here. Whether tuning "
                  "those closes the gap, or whether the placeholder parameters are simply too far "
                  "from real PC88A, is unresolved and needs the transcribed values (open item U4)."]
    md = out_dir / "TARGETED.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    write_manifest(out_dir / "manifest.json", [csv, md],
                   [paths.G18_ROOT / args.feed, Path(args.systems_dir) / f"{args.system}.json"],
                   args.seed, arguments=vars(args))
    print("\n".join(lines[6:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
