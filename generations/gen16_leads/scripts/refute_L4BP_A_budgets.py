"""Refuter lens A, step 7: where in the budget range the BP headline lives, and on which unit.

No refits.  Recomputed from `results/L4/_pe_keep/_pe_<design>.parquet`.

  a  ABC restricted to sub-ranges of the registered budget set (the registered endpoint averages
     six budgets; two of them are budgets at which the RANDOM model is worse than FLAT).
  b  the same contrast with the chemotype as the scoring unit instead of the extractant
     (`block_macro` in the lead's own output), to see whether the unit of analysis flips a design.
  c  per-budget AOPT-vs-FLAT and RANDOM-vs-FLAT, so the reader can see which arms are below the
     floor at each rung.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen15_curve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.inference import paired_contrasts  # noqa: E402

OUT = ROOT / "generations" / "gen16_leads" / "results" / "refutation" / "L4BP" / "A"
KEEP = ROOT / "generations" / "gen16_leads" / "results" / "L4" / "_pe_keep"
DESIGNS = ("B", "BR", "BQ", "A", "BP")
SUBSETS = {"6..24 (registered)": (6, 9, 12, 16, 20, 24), "9..24": (9, 12, 16, 20, 24),
           "12..24": (12, 16, 20, 24), "16..24": (16, 20, 24), "20..24": (20, 24)}
pd.set_option("display.width", 260)
LOG: list[str] = []


def say(m: str = "") -> None:
    print(m, flush=True)
    LOG.append(m)


def main() -> None:
    rows, unit_rows, floor_rows = [], [], []
    for design in DESIGNS:
        P = pd.read_parquet(KEEP / f"_pe_{design}.parquet")
        full = P[P.budget == "full"]
        flat = float(full[full.order == "FLAT"].groupby("split_seed")["mae_all"]
                     .mean().mean()) if len(full) else np.nan
        # macro FLAT the bench way: macro over extractants within seed, then over seeds
        fl = full[full.order == "FLAT"]
        flat = float(fl.groupby("split_seed")["mae_all"].mean().mean())
        P = P[P.budget != "full"].copy()
        P["budget"] = P["budget"].astype(int)
        da = P.groupby(["order", "budget", "split_seed", "extractant", "chemotype"])["mae_all"].mean()

        for lab, ks in SUBSETS.items():
            sub = da[da.index.get_level_values("budget").isin(ks)]
            abc = sub.groupby(level=["order", "split_seed", "extractant", "chemotype"]).mean().reset_index()
            c = paired_contrasts(abc.rename(columns={"order": "arm"}),
                                 {"AOPT_vs_RANDOM": ("RANDOM", "AOPT")}, value="mae_all")
            r = c.iloc[0]
            rows.append(dict(design=design, budgets=lab, n_budgets=len(ks), point=float(r.point),
                             ci_low=float(r.ci95_low), ci_high=float(r.ci95_high),
                             p=float(r.p_two_sided), seeds_positive=int(r.seeds_positive),
                             loco_stable=bool(r.loco_sign_stable), passes_P1=bool(r.passes_P1),
                             block_macro=float(r.block_macro)))

        # ---- chemotype as the scoring unit -------------------------------------------------
        abc = da.groupby(level=["order", "split_seed", "extractant", "chemotype"]).mean().reset_index()
        ch = (abc.groupby(["order", "split_seed", "chemotype"])["mae_all"].mean().reset_index()
              .rename(columns={"order": "arm", "chemotype": "extractant"}))
        ch["chemotype"] = ch["extractant"]
        c = paired_contrasts(ch, {"AOPT_vs_RANDOM": ("RANDOM", "AOPT"),
                                  "AOPT_vs_MAXMIN": ("MAXMIN", "AOPT")}, value="mae_all")
        for _, r in c.iterrows():
            unit_rows.append(dict(design=design, unit="chemotype", comparison=r.comparison,
                                  point=float(r.point), ci_low=float(r.ci95_low),
                                  ci_high=float(r.ci95_high), p=float(r.p_two_sided),
                                  seeds_positive=int(r.seeds_positive), n_units=int(r.n_units),
                                  passes_P1=bool(r.passes_P1)))

        # ---- which arms sit below the FLAT floor at each budget -----------------------------
        macro = (P.groupby(["order", "budget", "split_seed", "extractant"])["mae_all"].mean()
                 .groupby(level=["order", "budget", "split_seed"]).mean()
                 .groupby(level=["order", "budget"]).mean())
        for (o, k), v in macro.items():
            floor_rows.append(dict(design=design, order=o, budget=int(k), macro_mae=float(v),
                                   flat=flat, beats_flat=bool(v < flat)))

    R = pd.DataFrame(rows)
    U = pd.DataFrame(unit_rows)
    F = pd.DataFrame(floor_rows)
    R.to_csv(OUT / "abc_budget_subsets.csv", index=False)
    U.to_csv(OUT / "abc_unit_chemotype.csv", index=False)
    F.to_csv(OUT / "arms_vs_flat.csv", index=False)

    say("a. AOPT_vs_RANDOM ABC restricted to budget sub-ranges (point / p / seeds+ / P1)")
    say(R.pivot_table(index="design", columns="budgets", values="point").round(4).to_string())
    say()
    say(R[["design", "budgets", "point", "ci_low", "ci_high", "p", "seeds_positive",
           "loco_stable", "passes_P1"]].round(4).to_string(index=False))
    say()
    say("b. the same contrast with the CHEMOTYPE as the scoring unit (45 blocks, not 90 extractants)")
    say(U.round(4).to_string(index=False))
    say()
    say("c. macro MAE against the FLAT floor (arms below the floor are marked False)")
    say(F.pivot_table(index=["design", "order"], columns="budget", values="macro_mae")
        .round(4).to_string())
    say(f"   FLAT per design: "
        + ", ".join(f"{d} {F[F.design == d].flat.iloc[0]:.4f}" for d in DESIGNS))
    (OUT / "budget_sensitivity.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
